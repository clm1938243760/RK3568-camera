#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

#include "opencv2/opencv.hpp"
#include "ppocr_system.h"
#include "common.h"
#include "dict.h"
#include "file_utils.h"
#include "image_utils.h"

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

thread_local ppocr_runtime_timing_t g_runtime_timing = {};

void Trace(const char* stage)
{
    if (getenv("RK3568_PPOCR_TRACE") != NULL) {
        fprintf(stderr, "ppocr:%s\n", stage);
        fflush(stderr);
    }
}

std::uint16_t FloatToHalfBits(float value)
{
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

const std::array<std::uint16_t, 256>& RecognitionHalfLut()
{
    static const std::array<std::uint16_t, 256> values = [] {
        std::array<std::uint16_t, 256> table{};
        for (std::size_t index = 0; index < table.size(); ++index) {
            table[index] = FloatToHalfBits(
                (static_cast<float>(index) - 127.5F) / 127.5F);
        }
        return table;
    }();
    return values;
}

float HalfBitsToFloat(std::uint16_t value)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    std::uint32_t exponent = (value >> 10U) & 0x1fU;
    std::uint32_t mantissa = value & 0x03ffU;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127U - 15U + 1U;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign | (exponent << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
    }
    union {
        std::uint32_t bits;
        float value;
    } output;
    output.bits = bits;
    return output.value;
}

int RecognitionPostprocessFp16(const std::uint16_t* output, int channels,
                               int sequence_length, ppocr_rec_result* text)
{
    std::string result;
    float score = 0.0F;
    int previous_index = 0;
    int count = 0;
    for (int step = 0; step < sequence_length; ++step) {
        const std::uint16_t* row = output + static_cast<size_t>(step) * channels;
        int maximum_index = 0;
        std::uint16_t maximum = row[0];
        for (int channel = 1; channel < channels; ++channel) {
            if (row[channel] > maximum) {
                maximum = row[channel];
                maximum_index = channel;
            }
        }
        if (maximum_index > 0 && !(step > 0 && maximum_index == previous_index)) {
            if (maximum_index >= MODEL_OUT_CHANNEL) {
                return -1;
            }
            result += ocr_dict[maximum_index];
            score += HalfBitsToFloat(maximum);
            ++count;
        }
        previous_index = maximum_index;
    }
    text->score = count > 0 ? score / static_cast<float>(count) : 0.0F;
    text->str_size = count;
    std::strncpy(text->str, result.c_str(), sizeof(text->str) - 1U);
    text->str[sizeof(text->str) - 1U] = '\0';
    return 0;
}

}  // namespace

bool CompareBox(const std::array<int, 8>& result1, const std::array<int, 8>& result2)
{
    if (result1[1] < result2[1]) 
    {
        return true;
    } else if (result1[1] == result2[1]) 
    {
        return result1[0] < result2[0];
    } else 
    {
        return false;
    }
}

void SortBoxes(std::vector<std::array<int, 8>>* boxes)
{
    std::sort(boxes->begin(), boxes->end(), CompareBox);

    if (boxes->size() == 0)
    {
        return;
    }
    
    for (std::size_t i = 0; i + 1 < boxes->size(); ++i) {
        for (int j = static_cast<int>(i); j >= 0; --j) {
            if (std::abs((*boxes)[j + 1][1] - (*boxes)[j][1]) < 10 &&
                (*boxes)[j + 1][0] < (*boxes)[j][0]) {
                std::swap((*boxes)[j], (*boxes)[j + 1]);
            }
        }
    }

}

cv::Mat GetRotateCropImage(const cv::Mat& srcimage, const std::array<int, 8>& box)
{
    std::vector<std::vector<int>> points;

    for (int i = 0; i < 4; ++i) {
        std::vector<int> tmp;
        tmp.push_back(box[2 * i]);
        tmp.push_back(box[2 * i + 1]);
        points.push_back(tmp);
    }
    int x_collect[4] = {box[0], box[2], box[4], box[6]};
    int y_collect[4] = {box[1], box[3], box[5], box[7]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    const cv::Mat img_crop = srcimage(cv::Rect(left, top, right - left, bottom - top));

    for (int i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
    }

    int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                    pow(points[0][1] - points[1][1], 2)));
    int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                    pow(points[0][1] - points[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
        cv::Mat srcCopy;
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return dst_img;
    }
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, size_stride=%d, w_stride=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, attr->size_with_stride, attr->w_stride,
            get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_ppocr_model(const char* model_path, rknn_app_context_t* app_ctx)
{
    int ret;
    int model_len = 0;
    char* model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
        app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    app_ctx->bound_input_mem = NULL;
    app_ctx->bound_output_mem = NULL;
    app_ctx->staging_input = NULL;
    app_ctx->staging_input_size = 0;
    const bool is_recognition_model =
        app_ctx->model_height == 48 && app_ctx->model_width == 320 &&
        app_ctx->output_attrs[0].n_elems == 40U * MODEL_OUT_CHANNEL;
    if (is_recognition_model) {
        app_ctx->input_attrs[0].type = RKNN_TENSOR_FLOAT16;
        app_ctx->input_attrs[0].fmt = RKNN_TENSOR_NHWC;
        const size_t input_bytes = app_ctx->input_attrs[0].size_with_stride > 0
                                       ? app_ctx->input_attrs[0].size_with_stride
                                       : app_ctx->input_attrs[0].size;
        app_ctx->bound_input_mem = rknn_create_mem(ctx, input_bytes);
        if (app_ctx->bound_input_mem == NULL ||
            app_ctx->bound_input_mem->virt_addr == NULL ||
            rknn_set_io_mem(ctx, app_ctx->bound_input_mem,
                            &app_ctx->input_attrs[0]) != RKNN_SUCC) {
            printf("bind recognition input memory failed\n");
            return -1;
        }
        app_ctx->output_attrs[0].type = RKNN_TENSOR_FLOAT16;
        const size_t output_bytes = app_ctx->output_attrs[0].size_with_stride > 0
                                        ? app_ctx->output_attrs[0].size_with_stride
                                        : app_ctx->output_attrs[0].size;
        app_ctx->bound_output_mem = rknn_create_mem(ctx, output_bytes);
        if (app_ctx->bound_output_mem == NULL ||
            app_ctx->bound_output_mem->virt_addr == NULL ||
            rknn_set_io_mem(ctx, app_ctx->bound_output_mem,
                            &app_ctx->output_attrs[0]) != RKNN_SUCC) {
            printf("bind recognition output memory failed\n");
            return -1;
        }
    } else {
        app_ctx->staging_input_size = app_ctx->input_attrs[0].size;
        app_ctx->staging_input = static_cast<unsigned char*>(
            malloc(app_ctx->staging_input_size));
        if (app_ctx->staging_input == NULL) {
            printf("allocate detection staging input failed\n");
            return -1;
        }
    }

    return 0;
}

int release_ppocr_model(rknn_app_context_t* app_ctx)
{
    if (app_ctx->bound_output_mem != NULL && app_ctx->rknn_ctx != 0) {
        rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->bound_output_mem);
        app_ctx->bound_output_mem = NULL;
    }
    if (app_ctx->bound_input_mem != NULL && app_ctx->rknn_ctx != 0) {
        rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->bound_input_mem);
        app_ctx->bound_input_mem = NULL;
    }
    if (app_ctx->staging_input != NULL) {
        free(app_ctx->staging_input);
        app_ctx->staging_input = NULL;
        app_ctx->staging_input_size = 0;
    }
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_ppocr_det_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, ppocr_det_postprocess_params* params, ppocr_det_result* out_result)
{
    int ret;
    image_buffer_t img;
    rknn_input inputs[1];
    rknn_output outputs[1];

    memset(&img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    img.width = app_ctx->model_width;
    img.height = app_ctx->model_height;
    img.format = IMAGE_FORMAT_RGB888;
    img.size = get_image_size(&img);
    img.virt_addr = app_ctx->staging_input;
    if (img.virt_addr == NULL || app_ctx->staging_input_size < static_cast<size_t>(img.size)) {
        printf("malloc buffer size:%d fail!\n", img.size);
        return -1;
    }

    ret = convert_image(src_img, &img, NULL, NULL, 0);
    if (ret < 0) {
        printf("convert_image fail! ret=%d\n", ret);
        return -1;
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf   = img.virt_addr;

    float scale_w = (float)src_img->width / (float)img.width;
    float scale_h = (float)src_img->height / (float)img.height;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    // Post Process
    ret = dbnet_postprocess((float*)outputs[0].buf, app_ctx->model_width, app_ctx->model_height, 
                                                params->threshold, params->box_threshold, params->use_dilate, params->db_score_mode, 
                                                params->db_unclip_ratio, params->db_box_type,
                                                scale_w, scale_h, out_result);
    
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);

out:
    return ret;
}

int inference_ppocr_rec_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, ppocr_rec_result* out_result)
{
    int ret;
    const auto preprocess_started = Clock::now();
    // Pre Process
    float ratio = src_img->width / float(src_img->height);
    int resized_w;
    int imgW = app_ctx->model_width, imgH = app_ctx->model_height;
    if (std::ceil(imgH*ratio) > imgW) {
        resized_w = imgW;
    }
    else {
        resized_w = std::ceil(imgH*ratio);
    }

    cv::Mat img_M = cv::Mat(src_img->height, src_img->width, CV_8UC3,(uint8_t*)src_img->virt_addr);
    cv::resize(img_M, img_M, cv::Size(resized_w, imgH));
    if (resized_w < imgW) {
        copyMakeBorder(img_M, img_M, 0, 0, 0, imgW- resized_w, cv::BORDER_CONSTANT, 0);
    }

    if (app_ctx->bound_input_mem == NULL || app_ctx->bound_output_mem == NULL) {
        printf("recognition zero-copy buffers are not bound\n");
        return -1;
    }
    const auto& lookup = RecognitionHalfLut();
    std::uint16_t* input_half = static_cast<std::uint16_t*>(
        app_ctx->bound_input_mem->virt_addr);
    const int width_stride = app_ctx->input_attrs[0].w_stride > 0
                                 ? app_ctx->input_attrs[0].w_stride
                                 : imgW;
    for (int row = 0; row < imgH; ++row) {
        const unsigned char* source = img_M.ptr<unsigned char>(row);
        std::uint16_t* destination =
            input_half + static_cast<size_t>(row) * width_stride * app_ctx->model_channel;
        for (int column = 0; column < resized_w; ++column) {
            for (int channel = 0; channel < app_ctx->model_channel; ++channel) {
                const size_t source_index =
                    static_cast<size_t>(column) * app_ctx->model_channel + channel;
                destination[source_index] = lookup[source[source_index]];
            }
        }
        std::fill(destination + resized_w * app_ctx->model_channel,
                  destination + width_stride * app_ctx->model_channel,
                  static_cast<std::uint16_t>(0));
    }
    g_runtime_timing.recognition_preprocess_ms += ElapsedMs(preprocess_started);
    Trace("recognition_input_ready");

    // Run
    // printf("rknn_run\n");
    const auto inference_started = Clock::now();
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    g_runtime_timing.recognition_inference_ms += ElapsedMs(inference_started);
    Trace("recognition_inference_complete");

    // Get Output
    // Post Process
    const int out_len_seq = app_ctx->model_width / 8;
    const auto postprocess_started = Clock::now();
    ret = RecognitionPostprocessFp16(
        static_cast<const std::uint16_t*>(app_ctx->bound_output_mem->virt_addr),
        MODEL_OUT_CHANNEL, out_len_seq, out_result);
    g_runtime_timing.recognition_postprocess_ms += ElapsedMs(postprocess_started);
    Trace("recognition_postprocess_complete");
    return ret;
}

int inference_ppocr_system_model(ppocr_system_app_context* sys_app_ctx, image_buffer_t* src_img, ppocr_det_postprocess_params* params, ppocr_text_recog_array_result_t* out_result)
{
    int ret;
    memset(&g_runtime_timing, 0, sizeof(g_runtime_timing));
    // Detect Text
    ppocr_det_result det_results;
    const auto detection_started = Clock::now();
    ret = inference_ppocr_det_model(&sys_app_ctx->det_context, src_img, params, &det_results);
    g_runtime_timing.detection_ms = ElapsedMs(detection_started);
    if (ret != 0) {
        printf("inference_ppocr_det_model fail! ret=%d\n", ret);
        return -1;
    }
    Trace("detection_complete");

    // Recogize Text
    out_result->count = 0;
    if (det_results.count == 0) {           // detect nothing
        return 0;
    }

    // boxes to boxes_result
    std::vector<std::array<int, 8>> boxes_result;
    for (int i=0; i < det_results.count; i++) {
        std::array<int, 8> new_box;
        new_box[0] = det_results.box[i].left_top.x;
        new_box[1] = det_results.box[i].left_top.y;
        new_box[2] = det_results.box[i].right_top.x;
        new_box[3] = det_results.box[i].right_top.y;
        new_box[4] = det_results.box[i].right_bottom.x;
        new_box[5] = det_results.box[i].right_bottom.y;
        new_box[6] = det_results.box[i].left_bottom.x;
        new_box[7] = det_results.box[i].left_bottom.y;
        boxes_result.emplace_back(new_box);
    }

    // Sort text boxes in order from top to bottom, left to right for speeding up
    SortBoxes(&boxes_result);

    const cv::Mat in_image(src_img->height, src_img->width, CV_8UC3,
                           static_cast<uint8_t*>(src_img->virt_addr));

    // text recognize
    g_runtime_timing.recognition_count = static_cast<int>(boxes_result.size());
    for (int i=0; i < boxes_result.size(); i++) {
        Trace("recognition_crop_begin");
        const auto crop_started = Clock::now();
        cv::Mat crop_image = GetRotateCropImage(in_image, boxes_result[i]);
        if (!crop_image.isContinuous()) {
            crop_image = crop_image.clone();
        }
        g_runtime_timing.crop_ms += ElapsedMs(crop_started);
        image_buffer_t text_img;
        memset(&text_img, 0, sizeof(image_buffer_t));
        text_img.width = crop_image.cols;
        text_img.height = crop_image.rows;
        text_img.format = IMAGE_FORMAT_RGB888;
        text_img.size = get_image_size(&text_img);
        text_img.virt_addr = crop_image.data;
        Trace("recognition_crop_complete");
        
        ppocr_rec_result text_result;
        text_result.score = 1.0;
        ret = inference_ppocr_rec_model(&sys_app_ctx->rec_context, &text_img, &text_result);
        if (ret != 0) {
            printf("inference_ppocr_rec_model fail! ret=%d\n", ret);
            return -1;
        }
        if (text_result.score < TEXT_SCORE) {
            continue;
        }
        out_result->text_result[out_result->count].box.left_top.x = boxes_result[i][0];
        out_result->text_result[out_result->count].box.left_top.y = boxes_result[i][1];
        out_result->text_result[out_result->count].box.right_top.x = boxes_result[i][2];
        out_result->text_result[out_result->count].box.right_top.y = boxes_result[i][3];
        out_result->text_result[out_result->count].box.right_bottom.x = boxes_result[i][4];
        out_result->text_result[out_result->count].box.right_bottom.y = boxes_result[i][5];
        out_result->text_result[out_result->count].box.left_bottom.x = boxes_result[i][6];
        out_result->text_result[out_result->count].box.left_bottom.y = boxes_result[i][7];
        out_result->text_result[out_result->count].text = text_result;
        out_result->count ++;
    }

    return ret;
}

void ppocr_get_last_timing(ppocr_runtime_timing_t* timing)
{
    if (timing != NULL) {
        *timing = g_runtime_timing;
    }
}
