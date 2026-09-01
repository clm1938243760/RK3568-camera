#ifndef RK3568_CAMERA_STRUCTURED_FEEDBACK_H_
#define RK3568_CAMERA_STRUCTURED_FEEDBACK_H_

#include <string>

namespace rk3568_camera {

struct StructuredFeedback {
    std::string capture_id;
    std::string status;
    int field_count = 0;
    double structured_ms = 0.0;
};

bool ParseStructuredFeedback(const std::string& message, StructuredFeedback* result);

class StructuredFeedbackReceiver {
public:
    explicit StructuredFeedbackReceiver(const std::string& path);
    ~StructuredFeedbackReceiver();
    StructuredFeedbackReceiver(const StructuredFeedbackReceiver&) = delete;
    StructuredFeedbackReceiver& operator=(const StructuredFeedbackReceiver&) = delete;
    bool Receive(const std::string& capture_id, StructuredFeedback* result);

private:
    int descriptor_ = -1;
    std::string path_;
};

}  // namespace rk3568_camera
#endif
