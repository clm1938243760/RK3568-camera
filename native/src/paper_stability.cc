#include "paper_stability.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rk3568_camera {

PaperStabilityTracker::PaperStabilityTracker(StabilityConfig config) : config_(config) {
    if (config_.minimum_observations < 2 || config_.minimum_span_ns == 0 ||
        !(config_.minimum_iou > 0.0 && config_.minimum_iou <= 1.0) ||
        config_.maximum_center_shift_ratio < 0.0 || config_.maximum_area_change_ratio < 0.0) {
        throw std::invalid_argument("invalid paper stability configuration");
    }
}

StabilityUpdate PaperStabilityTracker::UpdatePresent(unsigned int sequence,
                                                      std::uint64_t timestamp_ns,
                                                      const PaperDetection& detection) {
    if (!detection.detected || detection.frame_width <= 0 || detection.frame_height <= 0) {
        throw std::invalid_argument("present observation requires a valid paper detection");
    }
    ValidateOrder(sequence, timestamp_ns);
    missing_started_ = false;
    if (state_ == PaperState::kLocked) {
        return Result("waiting_for_paper_removal");
    }

    const Geometry current = Normalize(detection);
    if (!has_anchor_) {
        StartTracking(sequence, timestamp_ns, current);
        return Result("paper_acquired");
    }
    if (has_last_sequence_ && sequence == last_sequence_) {
        return Result("duplicate_frame_ignored");
    }

    const double intersection_width =
        std::max(0.0, std::min(anchor_.right, current.right) - std::max(anchor_.left, current.left));
    const double intersection_height =
        std::max(0.0, std::min(anchor_.bottom, current.bottom) - std::max(anchor_.top, current.top));
    const double intersection = intersection_width * intersection_height;
    const double anchor_bbox_area =
        std::max(0.0, anchor_.right - anchor_.left) * std::max(0.0, anchor_.bottom - anchor_.top);
    const double current_bbox_area =
        std::max(0.0, current.right - current.left) * std::max(0.0, current.bottom - current.top);
    const double union_area = anchor_bbox_area + current_bbox_area - intersection;
    last_iou_ = union_area > 0.0 ? intersection / union_area : 0.0;
    last_center_shift_ = std::hypot(current.center_x - anchor_.center_x,
                                    current.center_y - anchor_.center_y);
    last_area_change_ = anchor_.area > 0.0
                            ? std::abs(current.area - anchor_.area) / anchor_.area
                            : std::numeric_limits<double>::infinity();
    const bool stable_geometry =
        last_iou_ >= config_.minimum_iou &&
        last_center_shift_ <= config_.maximum_center_shift_ratio &&
        last_area_change_ <= config_.maximum_area_change_ratio;
    if (!stable_geometry) {
        StartTracking(sequence, timestamp_ns, current);
        return Result("paper_moved");
    }

    last_sequence_ = sequence;
    last_timestamp_ns_ = timestamp_ns;
    ++observations_;
    const std::uint64_t span = timestamp_ns - tracking_since_ns_;
    if (observations_ >= config_.minimum_observations && span >= config_.minimum_span_ns) {
        state_ = PaperState::kLocked;
        StabilityUpdate update = Result("paper_stable");
        update.triggered = true;
        return update;
    }
    return Result(observations_ >= 2 ? "prestable" : "stability_pending");
}

StabilityUpdate PaperStabilityTracker::UpdateMissing(unsigned int sequence,
                                                      std::uint64_t timestamp_ns) {
    ValidateOrder(sequence, timestamp_ns);
    if (state_ != PaperState::kLocked) {
        Reset();
        has_last_sequence_ = true;
        last_sequence_ = sequence;
        last_timestamp_ns_ = timestamp_ns;
        return Result("paper_not_detected");
    }
    if (!missing_started_) {
        missing_started_ = true;
        missing_since_ns_ = timestamp_ns;
        return Result("paper_removal_pending");
    }
    if (timestamp_ns - missing_since_ns_ < config_.removal_span_ns) {
        return Result("paper_removal_pending");
    }
    Reset();
    has_last_sequence_ = true;
    last_sequence_ = sequence;
    last_timestamp_ns_ = timestamp_ns;
    StabilityUpdate update = Result("paper_removed");
    update.rearmed = true;
    return update;
}

void PaperStabilityTracker::Reset() {
    state_ = PaperState::kAbsent;
    has_anchor_ = false;
    has_last_sequence_ = false;
    last_sequence_ = 0;
    last_timestamp_ns_ = 0;
    tracking_since_ns_ = 0;
    missing_since_ns_ = 0;
    missing_started_ = false;
    observations_ = 0;
    last_iou_ = 0.0;
    last_center_shift_ = 0.0;
    last_area_change_ = 0.0;
}

PaperState PaperStabilityTracker::state() const { return state_; }

PaperStabilityTracker::Geometry PaperStabilityTracker::Normalize(
    const PaperDetection& detection) {
    Geometry geometry;
    geometry.left = 1.0;
    geometry.top = 1.0;
    for (const PointF& point : detection.corners) {
        const double x = point.x / detection.frame_width;
        const double y = point.y / detection.frame_height;
        if (!std::isfinite(x) || !std::isfinite(y)) {
            throw std::invalid_argument("paper corner coordinates must be finite");
        }
        geometry.left = std::min(geometry.left, x);
        geometry.top = std::min(geometry.top, y);
        geometry.right = std::max(geometry.right, x);
        geometry.bottom = std::max(geometry.bottom, y);
    }
    geometry.center_x = (geometry.left + geometry.right) / 2.0;
    geometry.center_y = (geometry.top + geometry.bottom) / 2.0;
    double twice_area = 0.0;
    for (std::size_t index = 0; index < detection.corners.size(); ++index) {
        const PointF& first = detection.corners[index];
        const PointF& second = detection.corners[(index + 1) % detection.corners.size()];
        twice_area += static_cast<double>(first.x) * second.y -
                      static_cast<double>(second.x) * first.y;
    }
    geometry.area = std::abs(twice_area) /
                    (2.0 * detection.frame_width * detection.frame_height);
    return geometry;
}

StabilityUpdate PaperStabilityTracker::Result(const std::string& reason) const {
    StabilityUpdate update;
    update.state = state_;
    update.prestable = state_ == PaperState::kTracking && observations_ >= 2;
    update.observations = observations_;
    update.stable_span_ns = has_anchor_ && last_timestamp_ns_ >= tracking_since_ns_
                                ? last_timestamp_ns_ - tracking_since_ns_
                                : 0;
    update.iou = last_iou_;
    update.center_shift_ratio = last_center_shift_;
    update.area_change_ratio = last_area_change_;
    update.reason = reason;
    return update;
}

void PaperStabilityTracker::StartTracking(unsigned int sequence, std::uint64_t timestamp_ns,
                                          const Geometry& geometry) {
    state_ = PaperState::kTracking;
    anchor_ = geometry;
    has_anchor_ = true;
    last_sequence_ = sequence;
    has_last_sequence_ = true;
    last_timestamp_ns_ = timestamp_ns;
    tracking_since_ns_ = timestamp_ns;
    observations_ = 1;
    last_iou_ = 1.0;
    last_center_shift_ = 0.0;
    last_area_change_ = 0.0;
}

void PaperStabilityTracker::ValidateOrder(unsigned int sequence, std::uint64_t timestamp_ns) {
    if (has_last_sequence_ && timestamp_ns < last_timestamp_ns_) {
        throw std::invalid_argument("paper observations must be timestamp ordered");
    }
    if (has_last_sequence_ && sequence < last_sequence_ &&
        last_sequence_ - sequence < std::numeric_limits<unsigned int>::max() / 2U) {
        throw std::invalid_argument("paper observations must be sequence ordered");
    }
}

}  // namespace rk3568_camera

