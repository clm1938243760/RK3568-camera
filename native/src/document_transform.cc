#include "document_transform.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace rk3568_camera {
namespace {

double Distance(const PointF& first, const PointF& second) {
    return std::hypot(static_cast<double>(first.x - second.x),
                      static_cast<double>(first.y - second.y));
}

int EvenFloor(double value) {
    const int rounded = static_cast<int>(std::floor(value));
    return rounded & ~1;
}

int EvenCeil(double value) {
    const int rounded = static_cast<int>(std::ceil(value));
    return (rounded + 1) & ~1;
}

}  // namespace

DocumentTransformPlan BuildDocumentTransform(const PaperDetection& detection,
                                             const DocumentTransformConfig& config) {
    if (!detection.detected || detection.frame_width <= 0 || detection.frame_height <= 0) {
        throw std::invalid_argument("document transform requires a detected paper");
    }
    if (config.canonical_long_side < 256 || config.canonical_short_side < 256 ||
        config.canonical_short_side > config.canonical_long_side ||
        config.recognition_top < 0.0 ||
        config.recognition_bottom > 1.0 || config.recognition_top >= config.recognition_bottom ||
        (config.rotation_degrees != 0 && config.rotation_degrees != 90 &&
         config.rotation_degrees != 180 && config.rotation_degrees != 270)) {
        throw std::invalid_argument("invalid document transform configuration");
    }

    double minimum_x = detection.frame_width;
    double minimum_y = detection.frame_height;
    double maximum_x = 0.0;
    double maximum_y = 0.0;
    for (const PointF& point : detection.corners) {
        minimum_x = std::min(minimum_x, static_cast<double>(point.x));
        minimum_y = std::min(minimum_y, static_cast<double>(point.y));
        maximum_x = std::max(maximum_x, static_cast<double>(point.x));
        maximum_y = std::max(maximum_y, static_cast<double>(point.y));
    }
    const int left = std::max(0, EvenFloor(minimum_x));
    const int top = std::max(0, EvenFloor(minimum_y));
    const int right = std::min(detection.frame_width, EvenCeil(maximum_x));
    const int bottom = std::min(detection.frame_height, EvenCeil(maximum_y));
    if (right - left < 32 || bottom - top < 32) {
        throw std::runtime_error("detected paper bounds are too small");
    }

    const double paper_width = std::max(Distance(detection.corners[0], detection.corners[1]),
                                        Distance(detection.corners[3], detection.corners[2]));
    const double paper_height = std::max(Distance(detection.corners[0], detection.corners[3]),
                                         Distance(detection.corners[1], detection.corners[2]));
    if (paper_width < 32.0 || paper_height < 32.0) {
        throw std::runtime_error("detected paper geometry is degenerate");
    }

    const bool swaps_axes = config.rotation_degrees == 90 || config.rotation_degrees == 270;
    const int canonical_width = swaps_axes ? config.canonical_short_side
                                           : config.canonical_long_side;
    const int canonical_height = swaps_axes ? config.canonical_long_side
                                            : config.canonical_short_side;
    const int region_top = std::max(
        0, std::min(canonical_height - 1,
                    static_cast<int>(std::floor(canonical_height * config.recognition_top))));
    const int region_bottom = std::max(
        region_top + 1,
        std::min(canonical_height,
                 static_cast<int>(std::ceil(canonical_height * config.recognition_bottom))));

    std::array<cv::Point2f, 4> source;
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = cv::Point2f(detection.corners[index].x - left,
                                    detection.corners[index].y - top);
    }
    const float output_bottom = static_cast<float>(canonical_height - 1 - region_top);
    const float output_top = static_cast<float>(-region_top);
    const float output_right = static_cast<float>(canonical_width - 1);
    std::array<cv::Point2f, 4> destination;
    if (config.rotation_degrees == 0) {
        destination = {{{0.0F, output_top}, {output_right, output_top},
                        {output_right, output_bottom}, {0.0F, output_bottom}}};
    } else if (config.rotation_degrees == 90) {
        destination = {{{0.0F, output_bottom}, {0.0F, output_top},
                        {output_right, output_top}, {output_right, output_bottom}}};
    } else if (config.rotation_degrees == 180) {
        destination = {{{output_right, output_bottom}, {0.0F, output_bottom},
                        {0.0F, output_top}, {output_right, output_top}}};
    } else {
        destination = {{{output_right, output_top}, {output_right, output_bottom},
                        {0.0F, output_bottom}, {0.0F, output_top}}};
    }

    DocumentTransformPlan plan;
    plan.source_bounds = cv::Rect(left, top, right - left, bottom - top);
    plan.canonical_size = cv::Size(canonical_width, canonical_height);
    plan.recognition_region = cv::Rect(0, region_top, canonical_width,
                                       region_bottom - region_top);
    plan.homography = cv::getPerspectiveTransform(source.data(), destination.data());
    return plan;
}

cv::Mat WarpDocumentRegion(const cv::Mat& cropped_bgr,
                           const DocumentTransformPlan& plan) {
    if (cropped_bgr.empty() || cropped_bgr.type() != CV_8UC3 ||
        cropped_bgr.size() != plan.source_bounds.size() || plan.homography.empty() ||
        plan.recognition_region.width <= 0 || plan.recognition_region.height <= 0) {
        throw std::invalid_argument("invalid direct document warp input");
    }
    cv::Mat output;
    cv::warpPerspective(cropped_bgr, output, plan.homography,
                        plan.recognition_region.size(), cv::INTER_CUBIC,
                        cv::BORDER_REPLICATE);
    return output;
}

}  // namespace rk3568_camera
