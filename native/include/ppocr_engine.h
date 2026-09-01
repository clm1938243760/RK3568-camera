#ifndef RK3568_CAMERA_PPOCR_ENGINE_H_
#define RK3568_CAMERA_PPOCR_ENGINE_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace rk3568_camera {

struct OcrItem {
    std::array<cv::Point, 4> polygon{};
    std::string text;
    double score = 0.0;
};

struct OcrResult {
    std::vector<OcrItem> items;
    double elapsed_ms = 0.0;
    double detection_ms = 0.0;
    double crop_ms = 0.0;
    double recognition_preprocess_ms = 0.0;
    double recognition_inference_ms = 0.0;
    double recognition_postprocess_ms = 0.0;
    int recognition_count = 0;
};

class PpOcrEngine {
public:
    PpOcrEngine(const std::string& detection_model, const std::string& recognition_model);
    ~PpOcrEngine();

    PpOcrEngine(const PpOcrEngine&) = delete;
    PpOcrEngine& operator=(const PpOcrEngine&) = delete;

    OcrResult RecognizeRgb(const cv::Mat& rgb_image);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rk3568_camera

#endif  // RK3568_CAMERA_PPOCR_ENGINE_H_
