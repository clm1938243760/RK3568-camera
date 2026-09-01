#include "npu_frequency_guard.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace rk3568_camera {

NpuFrequencyGuard::NpuFrequencyGuard(unsigned long requested_minimum_hz,
                                     std::string sysfs_directory)
    : minimum_path_(std::move(sysfs_directory) + "/min_freq"),
      requested_minimum_hz_(requested_minimum_hz) {
    if (requested_minimum_hz_ == 0) {
        throw std::invalid_argument("NPU minimum frequency must be positive");
    }
    previous_minimum_hz_ = ReadFrequency(minimum_path_);
    WriteFrequency(minimum_path_, requested_minimum_hz_);
    armed_ = true;
}

NpuFrequencyGuard::~NpuFrequencyGuard() {
    if (!armed_) {
        return;
    }
    try {
        WriteFrequency(minimum_path_, previous_minimum_hz_);
    } catch (...) {
    }
}

unsigned long NpuFrequencyGuard::previous_minimum_hz() const {
    return previous_minimum_hz_;
}

unsigned long NpuFrequencyGuard::requested_minimum_hz() const {
    return requested_minimum_hz_;
}

unsigned long NpuFrequencyGuard::ReadFrequency(const std::string& path) {
    std::ifstream stream(path);
    unsigned long value = 0;
    if (!(stream >> value) || value == 0) {
        throw std::runtime_error("cannot read NPU frequency from " + path);
    }
    return value;
}

void NpuFrequencyGuard::WriteFrequency(const std::string& path, unsigned long value) {
    std::ofstream stream(path);
    if (!stream || !(stream << value << '\n')) {
        throw std::runtime_error("cannot write NPU frequency to " + path + ": " +
                                 std::strerror(errno));
    }
}

}  // namespace rk3568_camera
