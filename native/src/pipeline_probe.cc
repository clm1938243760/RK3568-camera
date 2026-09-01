#include "docaligner_rknn.h"
#include "document_transform.h"
#include "frame_quality.h"
#include "ppocr_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

std::string JsonEscape(const std::string& value) {
    std::ostringstream output;
    for (unsigned char character : value) {
        if (character == '"') output << "\\\"";
        else if (character == '\\') output << "\\\\";
        else if (character == '\n') output << "\\n";
        else if (character == '\r') output << "\\r";
        else if (character == '\t') output << "\\t";
        else if (character >= 0x20U) output << static_cast<char>(character);
    }
    return output.str();
}

void WritePrivateOcr(const char* path, const rk3568_camera::OcrResult& result,
                     const cv::Size& image_size,
                     const rk3568_camera::PaperDetection& detection,
                     double averaged_ocr_ms) {
    if (path == nullptr || path[0] == '\0') return;
    const std::string temporary = std::string(path) + ".tmp." + std::to_string(getpid());
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create private probe OCR output");
    output << std::fixed << std::setprecision(4);
    output << "{\"capture_id\":\"00000000000000000000000000000001\","
           << "\"image_size\":[" << image_size.width << ',' << image_size.height
           << "],\"source\":{\"frame_size\":{\"width\":" << detection.frame_width
           << ",\"height\":" << detection.frame_height << "},\"paper_corners\":[";
    for (std::size_t index = 0; index < detection.corners.size(); ++index) {
        if (index > 0) output << ',';
        output << '[' << detection.corners[index].x << ',' << detection.corners[index].y
               << ']';
    }
    output << "],\"ocr_rotation\":90,\"ocr_document_long_side\":3200,"
           << "\"recognition_mode\":\"fixed_document_region\"},"
           << "\"timings\":{\"ocr_ms\":" << averaged_ocr_ms << "},\"ocr\":[";
    for (std::size_t index = 0; index < result.items.size(); ++index) {
        if (index > 0) output << ',';
        const auto& item = result.items[index];
        int left = image_size.width;
        int top = image_size.height;
        int right = 0;
        int bottom = 0;
        for (const cv::Point& point : item.polygon) {
            left = std::min(left, point.x);
            top = std::min(top, point.y);
            right = std::max(right, point.x);
            bottom = std::max(bottom, point.y);
        }
        output << "{\"text\":\"" << JsonEscape(item.text) << "\",\"score\":"
               << item.score << ",\"box\":[" << left << ',' << top << ',' << right << ','
               << bottom << "]}";
    }
    output << "]}\n";
    output.close();
    if (chmod(temporary.c_str(), 0600) != 0 ||
        std::rename(temporary.c_str(), path) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error("cannot publish private probe OCR output");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::cerr << "usage: " << argv[0]
                  << " DOCALIGNER.rknn DET.rknn REC.rknn IMAGE [MEASURED_RUNS]\n";
        return 2;
    }
    try {
        const int measured_runs = argc == 6 ? std::atoi(argv[5]) : 1;
        if (measured_runs < 1 || measured_runs > 100) {
            throw std::invalid_argument("measured runs must be in range 1..100");
        }
        cv::Mat source = cv::imread(argv[4], cv::IMREAD_COLOR);
        if (source.empty()) {
            throw std::runtime_error("cannot decode probe image");
        }
        rk3568_camera::DocAlignerRknn detector(
            argv[1], 0.5F, rk3568_camera::DocAlignerInputMode::kUint8);
        rk3568_camera::PpOcrEngine ocr(argv[2], argv[3]);
        double detection_ms = 0.0;
        double quality_ms = 0.0;
        double transform_ms = 0.0;
        double ocr_ms = 0.0;
        double ocr_detection_ms = 0.0;
        double ocr_crop_ms = 0.0;
        double recognition_preprocess_ms = 0.0;
        double recognition_inference_ms = 0.0;
        double recognition_postprocess_ms = 0.0;
        std::size_t expected_items = 0;
        int expected_recognitions = 0;
        double mean_score = 0.0;
        cv::Size region_size;
        rk3568_camera::OcrResult last_result;
        rk3568_camera::PaperDetection last_detection;

        const int warmup_runs = measured_runs > 1 ? 1 : 0;
        for (int iteration = -warmup_runs; iteration < measured_runs; ++iteration) {
            auto started = Clock::now();
            const rk3568_camera::PaperDetection detection = detector.DetectBgr(source);
            last_detection = detection;
            const double current_detection_ms = ElapsedMs(started);
            if (!detection.detected) {
                throw std::runtime_error("DocAligner did not detect the probe document");
            }

            started = Clock::now();
            const auto quality = rk3568_camera::EvaluateFinalFrame(source, detection);
            const double current_quality_ms = ElapsedMs(started);
            if (!quality.accepted) {
                throw std::runtime_error("probe image failed final-frame quality checks");
            }

            started = Clock::now();
            const auto transform = rk3568_camera::BuildDocumentTransform(detection);
            cv::Mat paper = source(transform.source_bounds).clone();
            cv::Mat region_bgr = rk3568_camera::WarpDocumentRegion(paper, transform);
            cv::Mat region_rgb;
            cv::cvtColor(region_bgr, region_rgb, cv::COLOR_BGR2RGB);
            const double current_transform_ms = ElapsedMs(started);
            const rk3568_camera::OcrResult result = ocr.RecognizeRgb(region_rgb);
            last_result = result;
            double score_sum = 0.0;
            for (const auto& item : result.items) score_sum += item.score;
            const double current_mean =
                result.items.empty() ? 0.0 : score_sum / result.items.size();
            if (iteration < 0) {
                expected_items = result.items.size();
                expected_recognitions = result.recognition_count;
                mean_score = current_mean;
                region_size = region_rgb.size();
                continue;
            }
            if (iteration == 0 && warmup_runs == 0) {
                expected_items = result.items.size();
                expected_recognitions = result.recognition_count;
                mean_score = current_mean;
                region_size = region_rgb.size();
            }
            if (result.items.size() != expected_items ||
                result.recognition_count != expected_recognitions ||
                std::abs(current_mean - mean_score) > 0.0001) {
                throw std::runtime_error("probe OCR result changed between warm runs");
            }
            detection_ms += current_detection_ms;
            quality_ms += current_quality_ms;
            transform_ms += current_transform_ms;
            ocr_ms += result.elapsed_ms;
            ocr_detection_ms += result.detection_ms;
            ocr_crop_ms += result.crop_ms;
            recognition_preprocess_ms += result.recognition_preprocess_ms;
            recognition_inference_ms += result.recognition_inference_ms;
            recognition_postprocess_ms += result.recognition_postprocess_ms;
        }
        const double divisor = static_cast<double>(measured_runs);
        detection_ms /= divisor;
        quality_ms /= divisor;
        transform_ms /= divisor;
        ocr_ms /= divisor;
        ocr_detection_ms /= divisor;
        ocr_crop_ms /= divisor;
        recognition_preprocess_ms /= divisor;
        recognition_inference_ms /= divisor;
        recognition_postprocess_ms /= divisor;
        WritePrivateOcr(std::getenv("RK3568_PRIVATE_PROBE_OCR"), last_result,
                        region_size, last_detection, ocr_ms);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "{\"ok\":true,";
        std::cout << "\"image_size\":[" << source.cols << "," << source.rows << "],";
        std::cout << "\"region_size\":[" << region_size.width << "," << region_size.height << "],";
        std::cout << "\"measured_runs\":" << measured_runs << ",";
        std::cout << "\"warmup_runs\":" << warmup_runs << ",";
        std::cout << "\"ocr_item_count\":" << expected_items << ",";
        std::cout << "\"mean_confidence\":" << mean_score << ",";
        std::cout << "\"timings\":{";
        std::cout << "\"detection_ms\":" << detection_ms << ",";
        std::cout << "\"quality_ms\":" << quality_ms << ",";
        std::cout << "\"transform_ms\":" << transform_ms << ",";
        std::cout << "\"ocr_ms\":" << ocr_ms << "},";
        std::cout << "\"ocr_breakdown\":{";
        std::cout << "\"detection_ms\":" << ocr_detection_ms << ",";
        std::cout << "\"crop_ms\":" << ocr_crop_ms << ",";
        std::cout << "\"recognition_preprocess_ms\":"
                  << recognition_preprocess_ms << ",";
        std::cout << "\"recognition_inference_ms\":"
                  << recognition_inference_ms << ",";
        std::cout << "\"recognition_postprocess_ms\":"
                  << recognition_postprocess_ms << ",";
        std::cout << "\"recognition_count\":" << expected_recognitions << "},";
        std::cout << "\"privacy\":{";
        std::cout << "\"ocr_text_emitted\":false,\"image_saved\":false}}\n";
        return expected_items == 0 ? 3 : 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"ok\":false,\"error\":\"" << error.what() << "\"}\n";
        return 1;
    }
}
