#ifndef RK3568_CAMERA_DETECTION_CADENCE_H_
#define RK3568_CAMERA_DETECTION_CADENCE_H_

#include <cstdint>

namespace rk3568_camera {

class DetectionCadence {
public:
    bool Due(std::uint64_t timestamp_ns) const {
        // V4L2 timestamps are rounded to microseconds; tolerate 1 ms of jitter.
        return !scheduled_ || timestamp_ns + 1000000ULL >= next_ns_;
    }

    void AfterDetection(std::uint64_t timestamp_ns, bool paper_present) {
        const std::uint64_t interval = paper_present ? 66666667ULL : 200000000ULL;
        if (!scheduled_ || interval != interval_ns_) {
            next_ns_ = timestamp_ns + interval;
        } else {
            next_ns_ += interval;
            if (next_ns_ <= timestamp_ns + 1000000ULL) {
                next_ns_ += ((timestamp_ns + 1000000ULL - next_ns_) / interval + 1) * interval;
            }
        }
        scheduled_ = true;
        interval_ns_ = interval;
    }

private:
    bool scheduled_ = false;
    std::uint64_t next_ns_ = 0;
    std::uint64_t interval_ns_ = 0;
};

}  // namespace rk3568_camera
#endif
