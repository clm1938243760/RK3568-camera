#ifndef RK3568_CAMERA_PAPER_STABILITY_H_
#define RK3568_CAMERA_PAPER_STABILITY_H_

#include <array>
#include <cstdint>
#include <string>

#include "docaligner_rknn.h"

namespace rk3568_camera {

enum class PaperState {
    kAbsent,
    kTracking,
    kLocked,
};

struct StabilityConfig {
    int minimum_observations = 4;
    std::uint64_t minimum_span_ns = 180000000ULL;
    double minimum_iou = 0.90;
    double maximum_center_shift_ratio = 0.03;
    double maximum_area_change_ratio = 0.15;
    std::uint64_t removal_span_ns = 1500000000ULL;
};

struct StabilityUpdate {
    PaperState state = PaperState::kAbsent;
    bool prestable = false;
    bool triggered = false;
    bool rearmed = false;
    int observations = 0;
    std::uint64_t stable_span_ns = 0;
    double iou = 0.0;
    double center_shift_ratio = 0.0;
    double area_change_ratio = 0.0;
    std::string reason;
};

class PaperStabilityTracker {
public:
    explicit PaperStabilityTracker(StabilityConfig config = StabilityConfig());

    StabilityUpdate UpdatePresent(unsigned int sequence, std::uint64_t timestamp_ns,
                                  const PaperDetection& detection);
    StabilityUpdate UpdateMissing(unsigned int sequence, std::uint64_t timestamp_ns);
    void Reset();
    PaperState state() const;

private:
    struct Geometry {
        double left = 0.0;
        double top = 0.0;
        double right = 0.0;
        double bottom = 0.0;
        double center_x = 0.0;
        double center_y = 0.0;
        double area = 0.0;
    };

    static Geometry Normalize(const PaperDetection& detection);
    StabilityUpdate Result(const std::string& reason) const;
    void StartTracking(unsigned int sequence, std::uint64_t timestamp_ns,
                       const Geometry& geometry);
    void ValidateOrder(unsigned int sequence, std::uint64_t timestamp_ns);

    StabilityConfig config_;
    PaperState state_ = PaperState::kAbsent;
    Geometry anchor_;
    bool has_anchor_ = false;
    unsigned int last_sequence_ = 0;
    bool has_last_sequence_ = false;
    std::uint64_t last_timestamp_ns_ = 0;
    std::uint64_t tracking_since_ns_ = 0;
    std::uint64_t missing_since_ns_ = 0;
    bool missing_started_ = false;
    int observations_ = 0;
    double last_iou_ = 0.0;
    double last_center_shift_ = 0.0;
    double last_area_change_ = 0.0;
};

}  // namespace rk3568_camera

#endif  // RK3568_CAMERA_PAPER_STABILITY_H_
