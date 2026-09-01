#include "docaligner_rknn.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <rknn_api.h>

namespace rk3568_camera {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

std::vector<unsigned char> ReadFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open RKNN model: " + path);
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0) {
        throw std::runtime_error("RKNN model is empty: " + path);
    }
    stream.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("cannot read RKNN model: " + path);
    }
    return data;
}

void CheckRknn(int code, const char* operation) {
    if (code != RKNN_SUCC) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(code));
    }
}

void Trace(const char* stage) {
    if (std::getenv("RK3568_CAMERA_TRACE") != nullptr) {
        std::fprintf(stderr, "docaligner:%s\n", stage);
        std::fflush(stderr);
    }
}

std::size_t FindOutput(const std::vector<rknn_tensor_attr>& attrs, const char* name) {
    for (std::size_t index = 0; index < attrs.size(); ++index) {
        if (std::strcmp(attrs[index].name, name) == 0) {
            return index;
        }
    }
    throw std::runtime_error(std::string("RKNN output is missing: ") + name);
}

std::uint16_t FloatToHalfBits(float value) {
    union {
        float value;
        std::uint32_t bits;
    } input;
    input.value = value;
    const std::uint32_t sign = input.bits & 0x80000000U;
    input.bits ^= sign;
    std::uint16_t result;
    if (input.bits >= 0x47800000U) {
        result = static_cast<std::uint16_t>(input.bits > 0x7f800000U ? 0x7e00U : 0x7c00U);
    } else if (input.bits < 0x38800000U) {
        input.value += 0.5F;
        result = static_cast<std::uint16_t>(input.bits - 0x3f000000U);
    } else {
        const std::uint32_t rounded = input.bits + 0xc8000fffU;
        result = static_cast<std::uint16_t>((rounded + ((input.bits >> 13U) & 1U)) >> 13U);
    }
    return static_cast<std::uint16_t>(result | (sign >> 16U));
}

const std::array<std::uint16_t, 256>& NormalizedHalfLut() {
    static const std::array<std::uint16_t, 256> values = [] {
        std::array<std::uint16_t, 256> table{};
        for (std::size_t index = 0; index < table.size(); ++index) {
            table[index] = FloatToHalfBits(static_cast<float>(index) / 255.0F);
        }
        return table;
    }();
    return values;
}

}  // namespace

class DocAlignerRknn::Impl {
public:
    Impl(const std::string& model_path, float threshold, DocAlignerInputMode input_mode)
        : threshold_(threshold), input_mode_(input_mode) {
        if (!(threshold > 0.0F && threshold < 1.0F)) {
            throw std::invalid_argument("DocAligner threshold must be between zero and one");
        }
        const std::vector<unsigned char> model = ReadFile(model_path);
        Trace("model_read");
        CheckRknn(rknn_init(&context_, const_cast<unsigned char*>(model.data()), model.size(), 0, nullptr),
                  "rknn_init");
        Trace("model_initialized");
        try {
            QueryContract();
        } catch (...) {
            ReleaseBoundIo();
            rknn_destroy(context_);
            context_ = 0;
            throw;
        }
    }

    ~Impl() {
        ReleaseBoundIo();
        if (context_ != 0) {
            rknn_destroy(context_);
        }
    }

    PaperDetection DetectBgr(const cv::Mat& bgr_image, int frame_width, int frame_height) {
        if (bgr_image.empty() || bgr_image.type() != CV_8UC3) {
            throw std::invalid_argument("DocAligner input must be a non-empty CV_8UC3 image");
        }

        PaperDetection detection;
        detection.frame_width = frame_width > 0 ? frame_width : bgr_image.cols;
        detection.frame_height = frame_height > 0 ? frame_height : bgr_image.rows;

        auto started = Clock::now();
        cv::Mat resized;
        if (bgr_image.cols == 256 && bgr_image.rows == 256 && bgr_image.isContinuous()) {
            resized = bgr_image;
        } else {
            cv::resize(bgr_image, resized, cv::Size(256, 256), 0.0, 0.0, cv::INTER_LINEAR);
        }
        std::vector<float> float_input;
        std::vector<std::uint16_t> half_input;
        rknn_input rknn_input_value;
        std::memset(&rknn_input_value, 0, sizeof(rknn_input_value));
        rknn_input_value.index = 0;
        if (input_mode_ == DocAlignerInputMode::kUint8) {
            const int width_stride = input_attr_.w_stride > 0
                                         ? static_cast<int>(input_attr_.w_stride)
                                         : 256;
            unsigned char* destination =
                static_cast<unsigned char*>(bound_input_mem_->virt_addr);
            if (width_stride == 256) {
                std::memcpy(destination, resized.data, resized.total() * resized.elemSize());
            } else {
                for (int row = 0; row < 256; ++row) {
                    std::memcpy(destination + row * width_stride * 3,
                                resized.ptr(row), 256 * 3);
                }
            }
        } else if (input_mode_ == DocAlignerInputMode::kFloat16) {
            const std::size_t value_count = resized.total() * resized.channels();
            half_input.resize(value_count);
            const auto& lookup = NormalizedHalfLut();
            for (std::size_t index = 0; index < value_count; ++index) {
                half_input[index] = lookup[resized.data[index]];
            }
            rknn_input_value.buf = half_input.data();
            rknn_input_value.size = static_cast<uint32_t>(half_input.size() * sizeof(std::uint16_t));
            rknn_input_value.type = RKNN_TENSOR_FLOAT16;
        } else {
            float_input.resize(3U * 256U * 256U);
            for (int y = 0; y < 256; ++y) {
                const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
                for (int x = 0; x < 256; ++x) {
                    const std::size_t offset = static_cast<std::size_t>((y * 256 + x) * 3);
                    float_input[offset] = static_cast<float>(row[x][0]) / 255.0F;
                    float_input[offset + 1] = static_cast<float>(row[x][1]) / 255.0F;
                    float_input[offset + 2] = static_cast<float>(row[x][2]) / 255.0F;
                }
            }
            rknn_input_value.buf = float_input.data();
            rknn_input_value.size = static_cast<uint32_t>(float_input.size() * sizeof(float));
            rknn_input_value.type = RKNN_TENSOR_FLOAT32;
        }
        if (input_mode_ != DocAlignerInputMode::kUint8) {
            rknn_input_value.fmt = RKNN_TENSOR_NHWC;
            rknn_input_value.pass_through = 0;
            CheckRknn(rknn_inputs_set(context_, 1, &rknn_input_value), "rknn_inputs_set");
        }
        detection.preprocess_ms = ElapsedMs(started);
        Trace("preprocess_complete");
        Trace("input_set");

        started = Clock::now();
        CheckRknn(rknn_run(context_, nullptr), "rknn_run");
        Trace("inference_complete");
        std::vector<rknn_output> outputs;
        if (input_mode_ != DocAlignerInputMode::kUint8) {
            outputs.resize(output_attrs_.size());
            std::memset(outputs.data(), 0, outputs.size() * sizeof(rknn_output));
            for (std::size_t index = 0; index < outputs.size(); ++index) {
                outputs[index].index = static_cast<uint32_t>(index);
                outputs[index].want_float = 1;
            }
            CheckRknn(rknn_outputs_get(context_, outputs.size(), outputs.data(), nullptr),
                      "rknn_outputs_get");
        }
        Trace("outputs_acquired");
        detection.inference_ms = ElapsedMs(started);

        started = Clock::now();
        const float* point_output = input_mode_ == DocAlignerInputMode::kUint8
                                        ? static_cast<const float*>(
                                              bound_output_mems_[points_output_]->virt_addr)
                                        : static_cast<const float*>(outputs[points_output_].buf);
        const float* object_output = input_mode_ == DocAlignerInputMode::kUint8
                                         ? static_cast<const float*>(
                                               bound_output_mems_[object_output_]->virt_addr)
                                         : static_cast<const float*>(outputs[object_output_].buf);
        detection.confidence = object_output[0];
        detection.detected = detection.confidence > threshold_;
        if (detection.detected) {
            for (std::size_t index = 0; index < detection.corners.size(); ++index) {
                detection.corners[index].x = std::max(
                    0.0F,
                    std::min(static_cast<float>(detection.frame_width),
                             point_output[index * 2] * detection.frame_width));
                detection.corners[index].y = std::max(
                    0.0F,
                    std::min(static_cast<float>(detection.frame_height),
                             point_output[index * 2 + 1] * detection.frame_height));
            }
        }
        if (input_mode_ != DocAlignerInputMode::kUint8) {
            CheckRknn(rknn_outputs_release(context_, outputs.size(), outputs.data()),
                      "rknn_outputs_release");
        }
        detection.postprocess_ms = ElapsedMs(started);
        Trace("postprocess_complete");
        Trace("outputs_released");
        return detection;
    }

    const std::string& runtime_version() const { return runtime_version_; }
    const std::string& driver_version() const { return driver_version_; }

private:
    void QueryContract() {
        rknn_input_output_num io_num;
        std::memset(&io_num, 0, sizeof(io_num));
        CheckRknn(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)),
                  "RKNN_QUERY_IN_OUT_NUM");
        Trace("io_count_queried");
        if (io_num.n_input != 1 || io_num.n_output != 2) {
            throw std::runtime_error("unexpected DocAligner RKNN input/output count");
        }

        std::memset(&input_attr_, 0, sizeof(input_attr_));
        input_attr_.index = 0;
        CheckRknn(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)),
                  "RKNN_QUERY_INPUT_ATTR");
        Trace("input_attr_queried");
        if (input_attr_.n_elems != 3U * 256U * 256U || input_attr_.fmt != RKNN_TENSOR_NHWC) {
            throw std::runtime_error("unexpected DocAligner RKNN input shape");
        }
        if (input_attr_.type != RKNN_TENSOR_UINT8 && input_attr_.type != RKNN_TENSOR_FLOAT16 &&
            input_attr_.type != RKNN_TENSOR_FLOAT32) {
            throw std::runtime_error("unexpected DocAligner RKNN input type");
        }

        output_attrs_.resize(io_num.n_output);
        for (uint32_t index = 0; index < io_num.n_output; ++index) {
            std::memset(&output_attrs_[index], 0, sizeof(output_attrs_[index]));
            output_attrs_[index].index = index;
            CheckRknn(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[index],
                                 sizeof(output_attrs_[index])),
                      "RKNN_QUERY_OUTPUT_ATTR");
        }
        Trace("output_attrs_queried");
        points_output_ = FindOutput(output_attrs_, "points");
        object_output_ = FindOutput(output_attrs_, "has_obj");
        if (output_attrs_[points_output_].n_elems != 8 ||
            output_attrs_[object_output_].n_elems != 1) {
            throw std::runtime_error("unexpected DocAligner RKNN output shape");
        }

        rknn_sdk_version version;
        std::memset(&version, 0, sizeof(version));
        CheckRknn(rknn_query(context_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)),
                  "RKNN_QUERY_SDK_VERSION");
        Trace("sdk_version_queried");
        runtime_version_ = version.api_version;
        driver_version_ = version.drv_version;
        if (input_mode_ == DocAlignerInputMode::kUint8) {
            BindUint8Io();
        }
    }

    void BindUint8Io() {
        input_attr_.type = RKNN_TENSOR_UINT8;
        input_attr_.fmt = RKNN_TENSOR_NHWC;
        bound_input_mem_ = rknn_create_mem(context_, input_attr_.size_with_stride);
        if (bound_input_mem_ == nullptr) {
            throw std::runtime_error("rknn_create_mem failed for DocAligner input");
        }
        CheckRknn(rknn_set_io_mem(context_, bound_input_mem_, &input_attr_),
                  "rknn_set_io_mem input");

        bound_output_mems_.resize(output_attrs_.size(), nullptr);
        for (std::size_t index = 0; index < output_attrs_.size(); ++index) {
            output_attrs_[index].type = RKNN_TENSOR_FLOAT32;
            bound_output_mems_[index] = rknn_create_mem(
                context_, output_attrs_[index].n_elems * sizeof(float));
            if (bound_output_mems_[index] == nullptr) {
                throw std::runtime_error("rknn_create_mem failed for DocAligner output");
            }
            CheckRknn(rknn_set_io_mem(context_, bound_output_mems_[index],
                                      &output_attrs_[index]),
                      "rknn_set_io_mem output");
        }
    }

    void ReleaseBoundIo() {
        for (rknn_tensor_mem* memory : bound_output_mems_) {
            if (memory != nullptr && context_ != 0) {
                rknn_destroy_mem(context_, memory);
            }
        }
        bound_output_mems_.clear();
        if (bound_input_mem_ != nullptr && context_ != 0) {
            rknn_destroy_mem(context_, bound_input_mem_);
            bound_input_mem_ = nullptr;
        }
    }

    rknn_context context_ = 0;
    float threshold_ = 0.5F;
    DocAlignerInputMode input_mode_ = DocAlignerInputMode::kFloat16;
    rknn_tensor_attr input_attr_{};
    std::vector<rknn_tensor_attr> output_attrs_;
    rknn_tensor_mem* bound_input_mem_ = nullptr;
    std::vector<rknn_tensor_mem*> bound_output_mems_;
    std::size_t points_output_ = 0;
    std::size_t object_output_ = 0;
    std::string runtime_version_;
    std::string driver_version_;
};

DocAlignerRknn::DocAlignerRknn(const std::string& model_path, float threshold,
                               DocAlignerInputMode input_mode)
    : impl_(new Impl(model_path, threshold, input_mode)) {}

DocAlignerRknn::~DocAlignerRknn() = default;

PaperDetection DocAlignerRknn::DetectBgr(const cv::Mat& bgr_image, int frame_width,
                                        int frame_height) {
    return impl_->DetectBgr(bgr_image, frame_width, frame_height);
}

const std::string& DocAlignerRknn::runtime_version() const {
    return impl_->runtime_version();
}

const std::string& DocAlignerRknn::driver_version() const {
    return impl_->driver_version();
}

}  // namespace rk3568_camera
