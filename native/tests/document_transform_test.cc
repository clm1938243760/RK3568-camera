#include "document_transform.h"
#include "frame_quality.h"
#include "v4l2_capture.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>

namespace {

rk3568_camera::PaperDetection SampleDetection() {
    rk3568_camera::PaperDetection detection;
    detection.detected = true;
    detection.confidence = 0.92F;
    detection.frame_width = 3840;
    detection.frame_height = 2160;
    detection.corners = {{{1220.636F, 419.407F},
                          {3088.503F, 390.387F},
                          {3129.373F, 1099.627F},
                          {1197.505F, 1090.424F}}};
    return detection;
}

}  // namespace

void RunDocumentTransformTests() {
    const auto detection = SampleDetection();
    const auto plan = rk3568_camera::BuildDocumentTransform(detection);
    assert(plan.source_bounds.x % 2 == 0);
    assert(plan.source_bounds.y % 2 == 0);
    assert(plan.source_bounds.width % 2 == 0);
    assert(plan.source_bounds.height % 2 == 0);
    assert(plan.canonical_size.height == 3200);
    assert(plan.canonical_size.width == 1176);
    assert(plan.recognition_region.y == 416);
    assert(plan.recognition_region.height == 1504);
    assert(plan.recognition_region.width == 1176);

    cv::Mat source(plan.source_bounds.height, plan.source_bounds.width, CV_8UC3,
                   cv::Scalar(128, 128, 128));
    cv::Mat output = rk3568_camera::WarpDocumentRegion(source, plan);
    assert(output.size() == plan.recognition_region.size());

    rk3568_camera::DocumentTransformConfig unrotated;
    unrotated.rotation_degrees = 0;
    unrotated.recognition_top = 0.0;
    unrotated.recognition_bottom = 1.0;
    const auto unrotated_plan = rk3568_camera::BuildDocumentTransform(detection, unrotated);
    assert(unrotated_plan.canonical_size.width == 3200);
    assert(unrotated_plan.canonical_size.height == 1176);

    rk3568_camera::DocumentTransformConfig invalid = unrotated;
    invalid.rotation_degrees = 45;
    bool rejected_invalid_rotation = false;
    try {
        rk3568_camera::BuildDocumentTransform(detection, invalid);
    } catch (const std::invalid_argument&) {
        rejected_invalid_rotation = true;
    }
    assert(rejected_invalid_rotation);

    cv::Mat analysis(540, 960, CV_8UC3, cv::Scalar(128, 128, 128));
    const auto quality = rk3568_camera::EvaluateFinalFrame(analysis, detection);
    assert(!quality.accepted);
    assert(quality.sharpness < 20.0);

    std::vector<unsigned char> nv12(3840U * 2160U * 3U / 2U, 128U);
    for (int y = 0; y < 2160; ++y) {
        unsigned char* row = nv12.data() + static_cast<std::size_t>(y) * 3840U;
        for (int x = 0; x < 3840; ++x) {
            row[x] = ((x / 32 + y / 32) % 2 == 0) ? 64U : 192U;
        }
    }
    rk3568_camera::Nv12Frame nv12_frame;
    nv12_frame.virtual_address = nv12.data();
    nv12_frame.bytes_used = nv12.size();
    nv12_frame.width = 3840;
    nv12_frame.height = 2160;
    nv12_frame.width_stride = 3840;
    nv12_frame.height_stride = 2160;
    const auto nv12_quality =
        rk3568_camera::EvaluateFinalFrameNv12(nv12_frame, detection);
    if (!nv12_quality.accepted) {
        std::cerr << "NV12 quality rejected: sharpness=" << nv12_quality.sharpness
                  << " contrast=" << nv12_quality.contrast
                  << " glare=" << nv12_quality.glare_ratio;
        for (const auto& reason : nv12_quality.reasons) std::cerr << ' ' << reason;
        std::cerr << std::endl;
    }
    assert(nv12_quality.accepted);
    assert(nv12_quality.sharpness >= 20.0);
}
