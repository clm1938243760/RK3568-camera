#ifndef RK3568_CAMERA_RGA_PREPROCESSOR_H_
#define RK3568_CAMERA_RGA_PREPROCESSOR_H_

#include <opencv2/core.hpp>

#include "v4l2_capture.h"

namespace rk3568_camera {

class RgaPreprocessor {
public:
    cv::Mat ResizeNv12ToBgr(const Nv12Frame& frame, int width, int height) const;
    cv::Mat CropNv12ToBgr(const Nv12Frame& frame, const cv::Rect& crop) const;
};

}  // namespace rk3568_camera

#endif  // RK3568_CAMERA_RGA_PREPROCESSOR_H_
