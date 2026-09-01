#include "structured_feedback.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace rk3568_camera {

bool ParseStructuredFeedback(const std::string& message, StructuredFeedback* result) {
    if (result == nullptr || message.size() > 256) return false;
    StructuredFeedback parsed;
    std::string extra;
    std::istringstream stream(message);
    if (!(stream >> parsed.capture_id >> parsed.status >> parsed.field_count >>
          parsed.structured_ms) || (stream >> extra)) return false;
    if (parsed.capture_id.size() != 32 ||
        !std::all_of(parsed.capture_id.begin(), parsed.capture_id.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) return false;
    if (parsed.status != "accepted" && parsed.status != "review_required" &&
        parsed.status != "rejected" && parsed.status != "error") return false;
    if (parsed.field_count < 0 || parsed.field_count > 256 ||
        !std::isfinite(parsed.structured_ms) || parsed.structured_ms < 0.0 ||
        parsed.structured_ms > 60000.0) return false;
    *result = parsed;
    return true;
}

StructuredFeedbackReceiver::StructuredFeedbackReceiver(const std::string& path)
    : path_(path) {
    sockaddr_un address{};
    if (path.empty() || path.size() >= sizeof(address.sun_path)) {
        throw std::invalid_argument("invalid structured feedback socket path");
    }
    address.sun_family = AF_UNIX;
    std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
    descriptor_ = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (descriptor_ < 0) throw std::runtime_error("cannot create structured feedback socket");
    unlink(path.c_str());
    if (bind(descriptor_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
        chmod(path.c_str(), 0600) < 0) {
        close(descriptor_);
        unlink(path.c_str());
        throw std::runtime_error("cannot bind structured feedback socket");
    }
}

StructuredFeedbackReceiver::~StructuredFeedbackReceiver() {
    if (descriptor_ >= 0) close(descriptor_);
    unlink(path_.c_str());
}

bool StructuredFeedbackReceiver::Receive(const std::string& capture_id,
                                         StructuredFeedback* result) {
    for (int index = 0; index < 32; ++index) {
        char buffer[257];
        const ssize_t count = recv(descriptor_, buffer, sizeof(buffer), 0);
        if (count < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) return false;
            throw std::runtime_error("structured feedback receive failed");
        }
        StructuredFeedback parsed;
        if (ParseStructuredFeedback(std::string(buffer, count), &parsed) &&
            parsed.capture_id == capture_id) {
            *result = parsed;
            return true;
        }
    }
    return false;
}

}  // namespace rk3568_camera
