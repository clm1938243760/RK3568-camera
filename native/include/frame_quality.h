#ifndef RK3568_CAMERA_FRAME_QUALITY_H_
#define RK3568_CAMERA_FRAME_QUALITY_H_

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "docaligner_rknn.h"

namespace rk3568_camera {

struct Nv12Frame;

struct FrameQualityConfig {
    int analysis_long_side = 384;
    double minimum_sharpness = 20.0;
    double maximum_glare_ratio = 0.85;
    int glare_threshold = 250;
    int clipped_edge_margin = 2;
};

struct FrameQualityResult {
    bool accepted = false;
    double sharpness = 0.0;
    double contrast = 0.0;
    double glare_ratio = 0.0;
    std::vector<std::string> reasons;
};

FrameQualityResult EvaluateFinalFrame(const cv::Mat& document_region,
                                      const PaperDetection& detection,
                                      const FrameQualityConfig& config = {});
FrameQualityResult EvaluateFinalFrameNv12(const Nv12Frame& frame,
                                          const PaperDetection& detection,
                                          const FrameQualityConfig& config = {});

}  // namespace rk3568_camera

#endif  // RK3568_CAMERA_FRAME_QUALITY_H_
