#ifndef RK3568_CAMERA_DOCALIGNER_RKNN_H_
#define RK3568_CAMERA_DOCALIGNER_RKNN_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace rk3568_camera {

struct PointF {
    float x = 0.0F;
    float y = 0.0F;
};

struct PaperDetection {
    bool detected = false;
    float confidence = 0.0F;
    std::array<PointF, 4> corners{};
    int frame_width = 0;
    int frame_height = 0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
};

enum class DocAlignerInputMode {
    kFloat32,
    kFloat16,
    kUint8,
};

class DocAlignerRknn {
public:
    explicit DocAlignerRknn(const std::string& model_path, float threshold = 0.5F,
                            DocAlignerInputMode input_mode = DocAlignerInputMode::kFloat16);
    ~DocAlignerRknn();

    DocAlignerRknn(const DocAlignerRknn&) = delete;
    DocAlignerRknn& operator=(const DocAlignerRknn&) = delete;

    PaperDetection DetectBgr(const cv::Mat& bgr_image, int frame_width = 0,
                             int frame_height = 0);
    const std::string& runtime_version() const;
    const std::string& driver_version() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rk3568_camera

#endif  // RK3568_CAMERA_DOCALIGNER_RKNN_H_
