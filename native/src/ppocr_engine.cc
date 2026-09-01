#include "ppocr_engine.h"

#include <chrono>
#include <cstring>
#include <stdexcept>

#include <image_utils.h>
#include <ppocr_system.h>

namespace rk3568_camera {
namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

class PpOcrEngine::Impl {
public:
    Impl(const std::string& detection_model, const std::string& recognition_model) {
        std::memset(&context_, 0, sizeof(context_));
        if (init_ppocr_model(detection_model.c_str(), &context_.det_context) != 0) {
            throw std::runtime_error("cannot initialize PP-OCR detection model");
        }
        if (init_ppocr_model(recognition_model.c_str(), &context_.rec_context) != 0) {
            release_ppocr_model(&context_.det_context);
            throw std::runtime_error("cannot initialize PP-OCR recognition model");
        }
    }

    ~Impl() {
        release_ppocr_model(&context_.det_context);
        release_ppocr_model(&context_.rec_context);
    }

    OcrResult RecognizeRgb(const cv::Mat& rgb_image) {
        if (rgb_image.empty() || rgb_image.type() != CV_8UC3 || !rgb_image.isContinuous()) {
            throw std::invalid_argument("PP-OCR input must be a contiguous CV_8UC3 RGB image");
        }
        image_buffer_t input;
        std::memset(&input, 0, sizeof(input));
        input.width = rgb_image.cols;
        input.height = rgb_image.rows;
        input.width_stride = rgb_image.cols;
        input.height_stride = rgb_image.rows;
        input.format = IMAGE_FORMAT_RGB888;
        input.size = static_cast<int>(rgb_image.total() * rgb_image.elemSize());
        input.virt_addr = const_cast<unsigned char*>(rgb_image.data);

        ppocr_det_postprocess_params parameters;
        parameters.threshold = 0.3F;
        parameters.box_threshold = 0.6F;
        parameters.use_dilate = false;
        parameters.db_score_mode = const_cast<char*>("slow");
        parameters.db_box_type = const_cast<char*>("poly");
        parameters.db_unclip_ratio = 1.5F;

        ppocr_text_recog_array_result_t native_result;
        std::memset(&native_result, 0, sizeof(native_result));
        const auto started = Clock::now();
        if (inference_ppocr_system_model(&context_, &input, &parameters, &native_result) != 0) {
            throw std::runtime_error("PP-OCR inference failed");
        }

        OcrResult result;
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        ppocr_runtime_timing_t timing;
        std::memset(&timing, 0, sizeof(timing));
        ppocr_get_last_timing(&timing);
        result.detection_ms = timing.detection_ms;
        result.crop_ms = timing.crop_ms;
        result.recognition_preprocess_ms = timing.recognition_preprocess_ms;
        result.recognition_inference_ms = timing.recognition_inference_ms;
        result.recognition_postprocess_ms = timing.recognition_postprocess_ms;
        result.recognition_count = timing.recognition_count;
        result.items.reserve(native_result.count);
        for (int index = 0; index < native_result.count; ++index) {
            const ppocr_text_recog_result_t& item = native_result.text_result[index];
            OcrItem converted;
            converted.polygon = {{{item.box.left_top.x, item.box.left_top.y},
                                  {item.box.right_top.x, item.box.right_top.y},
                                  {item.box.right_bottom.x, item.box.right_bottom.y},
                                  {item.box.left_bottom.x, item.box.left_bottom.y}}};
            converted.text = item.text.str;
            converted.score = item.text.score;
            result.items.push_back(std::move(converted));
        }
        return result;
    }

private:
    ppocr_system_app_context context_;
};

PpOcrEngine::PpOcrEngine(const std::string& detection_model,
                         const std::string& recognition_model)
    : impl_(new Impl(detection_model, recognition_model)) {}

PpOcrEngine::~PpOcrEngine() = default;

OcrResult PpOcrEngine::RecognizeRgb(const cv::Mat& rgb_image) {
    return impl_->RecognizeRgb(rgb_image);
}

}  // namespace rk3568_camera
