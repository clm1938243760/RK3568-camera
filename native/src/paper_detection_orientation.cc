#include "paper_detection_orientation.h"

#include <algorithm>
#include <stdexcept>

namespace rk3568_camera {
namespace {

PointF InverseRotate(const PointF& point, int rotated_width,
                     int rotated_height, int original_width,
                     int original_height, int clockwise_degrees) {
    const float rotated_x = point.x / static_cast<float>(rotated_width);
    const float rotated_y = point.y / static_cast<float>(rotated_height);
    float original_x = rotated_x;
    float original_y = rotated_y;
    if (clockwise_degrees == 90) {
        original_x = rotated_y;
        original_y = 1.0F - rotated_x;
    } else if (clockwise_degrees == 180) {
        original_x = 1.0F - rotated_x;
        original_y = 1.0F - rotated_y;
    } else if (clockwise_degrees == 270) {
        original_x = 1.0F - rotated_y;
        original_y = rotated_x;
    }
    PointF result;
    result.x = std::max(0.0F, std::min(static_cast<float>(original_width),
                                      original_x * original_width));
    result.y = std::max(0.0F, std::min(static_cast<float>(original_height),
                                      original_y * original_height));
    return result;
}

}  // namespace

PaperDetection RemapDetectionFromClockwiseRotation(
    const PaperDetection& rotated_detection, int original_width,
    int original_height, int clockwise_degrees) {
    if (original_width <= 0 || original_height <= 0 ||
        rotated_detection.frame_width <= 0 ||
        rotated_detection.frame_height <= 0 || clockwise_degrees < 0 ||
        clockwise_degrees > 270 || clockwise_degrees % 90 != 0) {
        throw std::invalid_argument("invalid paper detection rotation");
    }

    PaperDetection result = rotated_detection;
    result.frame_width = original_width;
    result.frame_height = original_height;
    if (!result.detected) return result;

    const int corner_offset = clockwise_degrees / 90;
    for (std::size_t index = 0; index < result.corners.size(); ++index) {
        const std::size_t rotated_index =
            (index + static_cast<std::size_t>(corner_offset)) % result.corners.size();
        result.corners[index] = InverseRotate(
            rotated_detection.corners[rotated_index],
            rotated_detection.frame_width, rotated_detection.frame_height,
            original_width, original_height, clockwise_degrees);
    }
    return result;
}

}  // namespace rk3568_camera
