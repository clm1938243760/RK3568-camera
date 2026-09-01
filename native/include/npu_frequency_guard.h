#ifndef RK3568_CAMERA_NPU_FREQUENCY_GUARD_H_
#define RK3568_CAMERA_NPU_FREQUENCY_GUARD_H_

#include <string>

namespace rk3568_camera {

class NpuFrequencyGuard {
public:
    explicit NpuFrequencyGuard(unsigned long requested_minimum_hz,
                               std::string sysfs_directory =
                                   "/sys/class/devfreq/fde40000.npu");
    ~NpuFrequencyGuard();

    NpuFrequencyGuard(const NpuFrequencyGuard&) = delete;
    NpuFrequencyGuard& operator=(const NpuFrequencyGuard&) = delete;

    unsigned long previous_minimum_hz() const;
    unsigned long requested_minimum_hz() const;

private:
    static unsigned long ReadFrequency(const std::string& path);
    static void WriteFrequency(const std::string& path, unsigned long value);

    std::string minimum_path_;
    unsigned long previous_minimum_hz_ = 0;
    unsigned long requested_minimum_hz_ = 0;
    bool armed_ = false;
};

}  // namespace rk3568_camera

#endif  // RK3568_CAMERA_NPU_FREQUENCY_GUARD_H_
