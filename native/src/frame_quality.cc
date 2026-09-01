#include "frame_quality.h"

#include "v4l2_capture.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace rk3568_camera {
namespace {

cv::Mat DownsampleLuma(const Nv12Frame& frame, int analysis_long_side) {
    if (frame.virtual_address == nullptr || frame.width <= 0 || frame.height <= 0 ||
        frame.width_stride < frame.width || frame.height_stride < frame.height ||
        analysis_long_side < 256) {
        throw std::invalid_argument("invalid NV12 quality frame");
    }
    const int longest_side = std::max(frame.width, frame.height);
    const double scale = std::min(1.0, static_cast<double>(analysis_long_side) / longest_side);
    const int width = std::max(2, static_cast<int>(std::lround(frame.width * scale)));
    const int height = std::max(2, static_cast<int>(std::lround(frame.height * scale)));
    cv::Mat gray(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        const int source_y = std::min(frame.height - 1, y * frame.height / height);
        const unsigned char* source =
            frame.virtual_address + static_cast<std::size_t>(source_y) * frame.width_stride;
        unsigned char* destination = gray.ptr<unsigned char>(y);
        for (int x = 0; x < width; ++x) {
            destination[x] = source[std::min(frame.width - 1, x * frame.width / width)];
        }
    }
    return gray;
}

FrameQualityResult EvaluateGray(const cv::Mat& gray,
                                const PaperDetection& detection,
                                const FrameQualityConfig& config) {
    if (gray.empty() || gray.type() != CV_8UC1 || !detection.detected ||
        detection.frame_width <= 0 || detection.frame_height <= 0 ||
        config.analysis_long_side < 256 || config.minimum_sharpness < 0.0 ||
        config.maximum_glare_ratio < 0.0 || config.maximum_glare_ratio > 1.0 ||
        config.glare_threshold < 0 || config.glare_threshold > 255) {
        throw std::invalid_argument("invalid final-frame quality input");
    }

    FrameQualityResult result;
    for (const PointF& point : detection.corners) {
        if (point.x <= config.clipped_edge_margin || point.y <= config.clipped_edge_margin ||
            point.x >= detection.frame_width - 1 - config.clipped_edge_margin ||
            point.y >= detection.frame_height - 1 - config.clipped_edge_margin) {
            result.reasons.push_back("paper_clipped_by_frame");
            break;
        }
    }

    std::vector<cv::Point> polygon;
    polygon.reserve(detection.corners.size());
    const double scale_x = static_cast<double>(gray.cols) / detection.frame_width;
    const double scale_y = static_cast<double>(gray.rows) / detection.frame_height;
    for (const PointF& point : detection.corners) {
        polygon.emplace_back(static_cast<int>(std::lround(point.x * scale_x)),
                             static_cast<int>(std::lround(point.y * scale_y)));
    }
    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(0));
    const std::vector<std::vector<cv::Point>> polygons = {polygon};
    cv::fillPoly(mask, polygons, cv::Scalar(255));
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
    const int selected_pixels = cv::countNonZero(mask);
    if (selected_pixels < 1024) {
        result.reasons.push_back("paper_area_too_small");
        result.accepted = false;
        return result;
    }
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_16S, 3);
    cv::Scalar laplacian_mean;
    cv::Scalar laplacian_deviation;
    cv::meanStdDev(laplacian, laplacian_mean, laplacian_deviation, mask);
    result.sharpness = laplacian_deviation[0] * laplacian_deviation[0];

    cv::Scalar gray_mean;
    cv::Scalar gray_deviation;
    cv::meanStdDev(gray, gray_mean, gray_deviation, mask);
    result.contrast = gray_deviation[0];

    cv::Mat glare_mask;
    cv::compare(gray, cv::Scalar(config.glare_threshold), glare_mask, cv::CMP_GE);
    cv::bitwise_and(glare_mask, mask, glare_mask);
    result.glare_ratio = static_cast<double>(cv::countNonZero(glare_mask)) /
                         static_cast<double>(selected_pixels);
    if (result.sharpness < config.minimum_sharpness) {
        result.reasons.push_back("frame_blurry");
    }
    if (result.glare_ratio > config.maximum_glare_ratio) {
        result.reasons.push_back("frame_overexposed");
    }
    result.accepted = result.reasons.empty();
    return result;
}

}  // namespace

FrameQualityResult EvaluateFinalFrame(const cv::Mat& document_region,
                                      const PaperDetection& detection,
                                      const FrameQualityConfig& config) {
    if (document_region.empty() || document_region.type() != CV_8UC3) {
        throw std::invalid_argument("invalid BGR quality frame");
    }
    cv::Mat analysis = document_region;
    const int longest_side = std::max(analysis.cols, analysis.rows);
    if (longest_side > config.analysis_long_side) {
        const double scale = static_cast<double>(config.analysis_long_side) / longest_side;
        cv::resize(document_region, analysis,
                   cv::Size(std::max(2, static_cast<int>(std::lround(document_region.cols * scale))),
                            std::max(2, static_cast<int>(std::lround(document_region.rows * scale)))),
                   0.0, 0.0, cv::INTER_AREA);
    }
    cv::Mat gray;
    cv::cvtColor(analysis, gray, cv::COLOR_BGR2GRAY);
    return EvaluateGray(gray, detection, config);
}

FrameQualityResult EvaluateFinalFrameNv12(const Nv12Frame& frame,
                                          const PaperDetection& detection,
                                          const FrameQualityConfig& config) {
    return EvaluateGray(DownsampleLuma(frame, config.analysis_long_side), detection, config);
}

}  // namespace rk3568_camera
