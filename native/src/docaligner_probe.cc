#include "docaligner_rknn.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace {

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }
    return values[middle];
}

void PrintTimings(const char* name, const std::vector<double>& values) {
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    std::cout << "\"" << name << "\":{";
    std::cout << "\"mean_ms\":" << sum / values.size() << ",";
    std::cout << "\"median_ms\":" << Median(values) << ",";
    std::cout << "\"min_ms\":" << *std::min_element(values.begin(), values.end()) << ",";
    std::cout << "\"max_ms\":" << *std::max_element(values.begin(), values.end()) << "}";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL.rknn IMAGE [ITERATIONS] [WARMUP] [float32|fp16|uint8]\n";
        return 2;
    }
    try {
        const int iterations = argc >= 4 ? std::atoi(argv[3]) : 20;
        const int warmup = argc >= 5 ? std::atoi(argv[4]) : 3;
        const std::string input_mode = argc >= 6 ? argv[5] : "fp16";
        if (iterations < 1 || warmup < 0) {
            throw std::invalid_argument("iterations must be positive and warmup non-negative");
        }
        if (input_mode != "float32" && input_mode != "fp16" && input_mode != "uint8") {
            throw std::invalid_argument("input mode must be float32, fp16 or uint8");
        }
        cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("cannot decode input image");
        }

        rk3568_camera::DocAlignerInputMode native_input_mode =
            rk3568_camera::DocAlignerInputMode::kFloat16;
        if (input_mode == "float32") {
            native_input_mode = rk3568_camera::DocAlignerInputMode::kFloat32;
        } else if (input_mode == "uint8") {
            native_input_mode = rk3568_camera::DocAlignerInputMode::kUint8;
        }
        rk3568_camera::DocAlignerRknn detector(argv[1], 0.5F, native_input_mode);
        for (int index = 0; index < warmup; ++index) {
            detector.DetectBgr(image);
        }

        std::vector<double> preprocess;
        std::vector<double> inference;
        std::vector<double> total;
        rk3568_camera::PaperDetection detection;
        for (int index = 0; index < iterations; ++index) {
            const auto started = std::chrono::steady_clock::now();
            detection = detector.DetectBgr(image);
            total.push_back(std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count());
            preprocess.push_back(detection.preprocess_ms);
            inference.push_back(detection.inference_ms);
        }

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "{\"ok\":true,";
        std::cout << "\"runtime_version\":\"" << detector.runtime_version() << "\",";
        std::cout << "\"driver_version\":\"" << detector.driver_version() << "\",";
        std::cout << "\"image_size\":[" << image.cols << "," << image.rows << "],";
        std::cout << "\"iterations\":" << iterations << ",";
        std::cout << "\"input_mode\":\"" << input_mode << "\",";
        std::cout << "\"detected\":" << (detection.detected ? "true" : "false") << ",";
        std::cout << "\"confidence\":" << detection.confidence << ",";
        std::cout << "\"corners\":[";
        for (std::size_t index = 0; index < detection.corners.size(); ++index) {
            if (index != 0) {
                std::cout << ",";
            }
            std::cout << "[" << detection.corners[index].x << ","
                      << detection.corners[index].y << "]";
        }
        std::cout << "],\"timings\":{";
        PrintTimings("preprocess", preprocess);
        std::cout << ",";
        PrintTimings("inference", inference);
        std::cout << ",";
        PrintTimings("total", total);
        std::cout << "}}\n";
        return detection.detected ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "{\"ok\":false,\"error\":\"" << error.what() << "\"}\n";
        return 1;
    }
}
