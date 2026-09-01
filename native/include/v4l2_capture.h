#ifndef RK3568_CAMERA_V4L2_CAPTURE_H_
#define RK3568_CAMERA_V4L2_CAPTURE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rk3568_camera {

struct Nv12Frame {
    unsigned int buffer_index = 0;
    unsigned int sequence = 0;
    std::uint64_t timestamp_ns = 0;
    bool monotonic_timestamp = false;
    bool valid = false;
    unsigned int discarded_buffers = 0;
    int dma_fd = -1;
    unsigned char* virtual_address = nullptr;
    std::size_t bytes_used = 0;
    int width = 0;
    int height = 0;
    int width_stride = 0;
    int height_stride = 0;
};

class V4l2Nv12Capture {
public:
    V4l2Nv12Capture(const std::string& device, int width, int height, unsigned int buffers = 4);
    ~V4l2Nv12Capture();

    V4l2Nv12Capture(const V4l2Nv12Capture&) = delete;
    V4l2Nv12Capture& operator=(const V4l2Nv12Capture&) = delete;

    bool Dequeue(Nv12Frame* frame, int timeout_ms);
    bool DequeueLatest(Nv12Frame* frame, int timeout_ms);
    void Requeue(const Nv12Frame& frame);

    int width() const;
    int height() const;
    int width_stride() const;
    int height_stride() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rk3568_camera

#endif  // RK3568_CAMERA_V4L2_CAPTURE_H_
