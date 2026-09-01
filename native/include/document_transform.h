#ifndef RK3568_CAMERA_DOCUMENT_TRANSFORM_H_
#define RK3568_CAMERA_DOCUMENT_TRANSFORM_H_

#include <opencv2/core.hpp>

#include "docaligner_rknn.h"

namespace rk3568_camera {

struct DocumentTransformConfig {
    int canonical_long_side = 3200;
    int canonical_short_side = 1176;
    double recognition_top = 0.13;
    double recognition_bottom = 0.60;
    int rotation_degrees = 90;
};

struct DocumentTransformPlan {
    cv::Rect source_bounds;
    cv::Size canonical_size;
    cv::Rect recognition_region;
    cv::Mat homography;
};

DocumentTransformPlan BuildDocumentTransform(const PaperDetection& detection,
                                             const DocumentTransformConfig& config = {});
cv::Mat WarpDocumentRegion(const cv::Mat& cropped_bgr,
                           const DocumentTransformPlan& plan);

}  // namespace rk3568_camera

#endif  // RK3568_CAMERA_DOCUMENT_TRANSFORM_H_
