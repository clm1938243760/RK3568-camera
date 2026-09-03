#ifndef RK3568_CAMERA_PAPER_DETECTION_ORIENTATION_H_
#define RK3568_CAMERA_PAPER_DETECTION_ORIENTATION_H_

#include "docaligner_rknn.h"

namespace rk3568_camera {

PaperDetection RemapDetectionFromClockwiseRotation(
    const PaperDetection& rotated_detection, int original_width,
    int original_height, int clockwise_degrees);

}  // namespace rk3568_camera

#endif  // RK3568_CAMERA_PAPER_DETECTION_ORIENTATION_H_
