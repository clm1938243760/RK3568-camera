#include "docaligner_rknn.h"
#include "document_transform.h"
#include "frame_quality.h"
#include "rga_preprocessor.h"
#include "v4l2_capture.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

void PrintSummary(const char* name, std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const std::size_t middle = values.size() / 2;
    const double median = values.size() % 2 == 0
                              ? (values[middle - 1] + values[middle]) / 2.0
                              : values[middle];
    std::cout << "\"" << name << "\":{";
    std::cout << "\"mean_ms\":" << sum / values.size() << ",";
    std::cout << "\"median_ms\":" << median << ",";
    std::cout << "\"min_ms\":" << values.front() << ",";
    std::cout << "\"max_ms\":" << values.back() << "}";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: " << argv[0] << " MODEL.rknn [DEVICE] [FRAMES] [DETECT_EVERY]\n";
        return 2;
    }
    try {
        const std::string device = argc >= 3 ? argv[2] : "/dev/video0";
        const int frame_count = argc >= 4 ? std::atoi(argv[3]) : 120;
        const int detect_every = argc >= 5 ? std::atoi(argv[4]) : 2;
        if (frame_count < 1 || detect_every < 1) {
            throw std::invalid_argument("frames and detect-every must be positive");
        }

        rk3568_camera::DocAlignerRknn detector(
            argv[1], 0.5F, rk3568_camera::DocAlignerInputMode::kUint8);
        rk3568_camera::RgaPreprocessor preprocessor;
        rk3568_camera::V4l2Nv12Capture capture(device, 3840, 2160, 4);
        std::vector<double> rga_ms;
        std::vector<double> inference_ms;
        std::vector<double> input_prepare_ms;
        std::vector<double> postprocess_ms;
        std::vector<double> detection_total_ms;
        unsigned int last_sequence = 0;
        int detections = 0;
        int processed = 0;
        bool final_frame_processed = false;
        double final_crop_ms = 0.0;
        double final_warp_ms = 0.0;
        double final_quality_rga_ms = 0.0;
        double final_quality_ms = 0.0;
        cv::Size final_region_size;
        rk3568_camera::FrameQualityResult final_quality;
        const auto probe_started = Clock::now();

        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            rk3568_camera::Nv12Frame frame;
            if (!capture.Dequeue(&frame, 2000)) {
                throw std::runtime_error("camera frame timeout");
            }
            last_sequence = frame.sequence;
            try {
                if (frame_index % detect_every == 0) {
                    const auto detection_started = Clock::now();
                    const auto rga_started = Clock::now();
                    cv::Mat input = preprocessor.ResizeNv12ToBgr(frame, 256, 256);
                    rga_ms.push_back(ElapsedMs(rga_started));
                    rk3568_camera::PaperDetection result =
                        detector.DetectBgr(input, frame.width, frame.height);
                    input_prepare_ms.push_back(result.preprocess_ms);
                    inference_ms.push_back(result.inference_ms);
                    postprocess_ms.push_back(result.postprocess_ms);
                    detection_total_ms.push_back(ElapsedMs(detection_started));
                    detections += result.detected ? 1 : 0;
                    ++processed;
                    if (result.detected && !final_frame_processed) {
                        auto stage_started = Clock::now();
                        cv::Mat quality_input = preprocessor.ResizeNv12ToBgr(frame, 960, 540);
                        final_quality_rga_ms = ElapsedMs(stage_started);
                        stage_started = Clock::now();
                        final_quality = rk3568_camera::EvaluateFinalFrame(quality_input, result);
                        final_quality_ms = ElapsedMs(stage_started);
                        const rk3568_camera::DocumentTransformPlan transform =
                            rk3568_camera::BuildDocumentTransform(result);
                        if (final_quality.accepted) {
                            stage_started = Clock::now();
                            cv::Mat paper =
                                preprocessor.CropNv12ToBgr(frame, transform.source_bounds);
                            final_crop_ms = ElapsedMs(stage_started);
                            stage_started = Clock::now();
                            cv::Mat region =
                                rk3568_camera::WarpDocumentRegion(paper, transform);
                            final_warp_ms = ElapsedMs(stage_started);
                            final_region_size = region.size();
                        }
                        final_frame_processed = true;
                    }
                }
            } catch (...) {
                capture.Requeue(frame);
                throw;
            }
            capture.Requeue(frame);
        }

        const double elapsed_ms = ElapsedMs(probe_started);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "{\"ok\":true,";
        std::cout << "\"frame_size\":[" << capture.width() << "," << capture.height() << "],";
        std::cout << "\"width_stride\":" << capture.width_stride() << ",";
        std::cout << "\"height_stride\":" << capture.height_stride() << ",";
        std::cout << "\"frames\":" << frame_count << ",";
        std::cout << "\"last_sequence\":" << last_sequence << ",";
        std::cout << "\"capture_fps\":" << frame_count * 1000.0 / elapsed_ms << ",";
        std::cout << "\"processed\":" << processed << ",";
        std::cout << "\"paper_detections\":" << detections << ",";
        std::cout << "\"final_frame\":{";
        std::cout << "\"processed\":" << (final_frame_processed ? "true" : "false") << ",";
        std::cout << "\"region_size\":[" << final_region_size.width << ","
                  << final_region_size.height << "],";
        std::cout << "\"crop_ms\":" << final_crop_ms << ",";
        std::cout << "\"warp_ms\":" << final_warp_ms << ",";
        std::cout << "\"quality_rga_ms\":" << final_quality_rga_ms << ",";
        std::cout << "\"quality_ms\":" << final_quality_ms << ",";
        std::cout << "\"quality_accepted\":" << (final_quality.accepted ? "true" : "false") << ",";
        std::cout << "\"sharpness\":" << final_quality.sharpness << ",";
        std::cout << "\"contrast\":" << final_quality.contrast << ",";
        std::cout << "\"glare_ratio\":" << final_quality.glare_ratio << "},";
        std::cout << "\"timings\":{";
        PrintSummary("rga_nv12_to_bgr_256", rga_ms);
        std::cout << ",";
        PrintSummary("npu_input_prepare", input_prepare_ms);
        std::cout << ",";
        PrintSummary("npu_inference", inference_ms);
        std::cout << ",";
        PrintSummary("npu_postprocess", postprocess_ms);
        std::cout << ",";
        PrintSummary("detection_total", detection_total_ms);
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"ok\":false,\"error\":\"" << error.what() << "\"}\n";
        return 1;
    }
}
