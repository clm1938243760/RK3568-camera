#include "v4l2_capture.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace rk3568_camera {
namespace {

int Ioctl(int fd, unsigned long request, void* argument) {
    int result;
    do {
        result = ioctl(fd, request, argument);
    } while (result < 0 && errno == EINTR);
    return result;
}

std::runtime_error SystemError(const std::string& operation) {
    return std::runtime_error(operation + ": " + std::strerror(errno));
}

struct Buffer {
    unsigned char* address = nullptr;
    std::size_t length = 0;
    int dma_fd = -1;
    bool dequeued = false;
};

}  // namespace

class V4l2Nv12Capture::Impl {
public:
    Impl(const std::string& device, int requested_width, int requested_height,
         unsigned int requested_buffers) {
        if (requested_width <= 0 || requested_height <= 0 || requested_buffers < 2) {
            throw std::invalid_argument("invalid V4L2 capture dimensions or buffer count");
        }
        fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd_ < 0) {
            throw SystemError("open " + device);
        }
        try {
            Configure(requested_width, requested_height, requested_buffers);
        } catch (...) {
            Cleanup();
            throw;
        }
    }

    ~Impl() { Cleanup(); }

    bool Dequeue(Nv12Frame* frame, int timeout_ms) {
        if (frame == nullptr || timeout_ms < 0) {
            throw std::invalid_argument("invalid dequeue arguments");
        }
        pollfd descriptor;
        std::memset(&descriptor, 0, sizeof(descriptor));
        descriptor.fd = fd_;
        descriptor.events = POLLIN | POLLPRI;
        int poll_result;
        do {
            poll_result = poll(&descriptor, 1, timeout_ms);
        } while (poll_result < 0 && errno == EINTR);
        if (poll_result == 0) {
            return false;
        }
        if (poll_result < 0) {
            throw SystemError("poll V4L2 frame");
        }

        v4l2_plane plane;
        std::memset(&plane, 0, sizeof(plane));
        v4l2_buffer buffer;
        std::memset(&buffer, 0, sizeof(buffer));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.length = 1;
        buffer.m.planes = &plane;
        if (Ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
            if (errno == EAGAIN) {
                return false;
            }
            throw SystemError("VIDIOC_DQBUF");
        }
        if (buffer.index >= buffers_.size() || buffers_[buffer.index].dequeued) {
            throw std::runtime_error("V4L2 returned an invalid buffer index");
        }
        Buffer& mapped = buffers_[buffer.index];
        mapped.dequeued = true;
        frame->buffer_index = buffer.index;
        frame->sequence = buffer.sequence;
        frame->timestamp_ns = static_cast<std::uint64_t>(buffer.timestamp.tv_sec) * 1000000000ULL +
                              static_cast<std::uint64_t>(buffer.timestamp.tv_usec) * 1000ULL;
        frame->monotonic_timestamp =
            (buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
        frame->valid = (buffer.flags & V4L2_BUF_FLAG_ERROR) == 0 &&
                       plane.data_offset == 0 && plane.bytesused <= mapped.length &&
                       plane.bytesused >= static_cast<std::size_t>(width_stride_) *
                                              height_stride_ * 3U / 2U;
        frame->discarded_buffers = 0;
        frame->dma_fd = mapped.dma_fd;
        frame->virtual_address = mapped.address;
        frame->bytes_used = plane.bytesused;
        frame->width = width_;
        frame->height = height_;
        frame->width_stride = width_stride_;
        frame->height_stride = height_stride_;
        return true;
    }

    bool DequeueLatest(Nv12Frame* frame, int timeout_ms) {
        if (!Dequeue(frame, timeout_ms)) return false;
        unsigned int discarded = 0;
        try {
            // Bound draining even if the sensor produces frames during this loop.
            for (std::size_t index = 1; index < buffers_.size(); ++index) {
                Nv12Frame newer;
                if (!Dequeue(&newer, 0)) break;
                try {
                    Requeue(*frame);
                } catch (...) {
                    Requeue(newer);
                    throw;
                }
                *frame = newer;
                ++discarded;
            }
        } catch (...) {
            if (buffers_[frame->buffer_index].dequeued) Requeue(*frame);
            throw;
        }
        frame->discarded_buffers = discarded;
        return true;
    }

    void Requeue(const Nv12Frame& frame) {
        if (frame.buffer_index >= buffers_.size() || !buffers_[frame.buffer_index].dequeued) {
            throw std::runtime_error("attempted to requeue a buffer that is not dequeued");
        }
        Queue(frame.buffer_index);
        buffers_[frame.buffer_index].dequeued = false;
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int width_stride() const { return width_stride_; }
    int height_stride() const { return height_stride_; }

private:
    void Configure(int requested_width, int requested_height, unsigned int requested_buffers) {
        v4l2_capability capability;
        std::memset(&capability, 0, sizeof(capability));
        if (Ioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
            throw SystemError("VIDIOC_QUERYCAP");
        }
        if ((capability.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) == 0 ||
            (capability.device_caps & V4L2_CAP_STREAMING) == 0) {
            throw std::runtime_error("device does not support multiplanar streaming capture");
        }

        v4l2_format format;
        std::memset(&format, 0, sizeof(format));
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        format.fmt.pix_mp.width = requested_width;
        format.fmt.pix_mp.height = requested_height;
        format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
        format.fmt.pix_mp.field = V4L2_FIELD_NONE;
        format.fmt.pix_mp.num_planes = 1;
        if (Ioctl(fd_, VIDIOC_S_FMT, &format) < 0) {
            throw SystemError("VIDIOC_S_FMT");
        }
        if (format.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12 ||
            format.fmt.pix_mp.num_planes != 1) {
            throw std::runtime_error("camera did not accept single-plane NV12");
        }
        width_ = static_cast<int>(format.fmt.pix_mp.width);
        height_ = static_cast<int>(format.fmt.pix_mp.height);
        width_stride_ = static_cast<int>(format.fmt.pix_mp.plane_fmt[0].bytesperline);
        height_stride_ = static_cast<int>(
            format.fmt.pix_mp.plane_fmt[0].sizeimage /
            std::max(1, width_stride_ * 3 / 2));
        if (height_stride_ < height_) {
            height_stride_ = height_;
        }

        v4l2_requestbuffers request;
        std::memset(&request, 0, sizeof(request));
        request.count = requested_buffers;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        request.memory = V4L2_MEMORY_MMAP;
        if (Ioctl(fd_, VIDIOC_REQBUFS, &request) < 0) {
            throw SystemError("VIDIOC_REQBUFS");
        }
        if (request.count < 2) {
            throw std::runtime_error("camera allocated fewer than two capture buffers");
        }
        buffers_.resize(request.count);
        for (unsigned int index = 0; index < request.count; ++index) {
            MapBuffer(index);
            Queue(index);
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (Ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            throw SystemError("VIDIOC_STREAMON");
        }
        streaming_ = true;
    }

    void MapBuffer(unsigned int index) {
        v4l2_plane plane;
        std::memset(&plane, 0, sizeof(plane));
        v4l2_buffer buffer;
        std::memset(&buffer, 0, sizeof(buffer));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        buffer.length = 1;
        buffer.m.planes = &plane;
        if (Ioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
            throw SystemError("VIDIOC_QUERYBUF");
        }
        void* address = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                             plane.m.mem_offset);
        if (address == MAP_FAILED) {
            throw SystemError("mmap V4L2 buffer");
        }
        buffers_[index].address = static_cast<unsigned char*>(address);
        buffers_[index].length = plane.length;

        v4l2_exportbuffer exported;
        std::memset(&exported, 0, sizeof(exported));
        exported.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        exported.index = index;
        exported.plane = 0;
        exported.flags = O_CLOEXEC;
        if (Ioctl(fd_, VIDIOC_EXPBUF, &exported) < 0) {
            throw SystemError("VIDIOC_EXPBUF");
        }
        buffers_[index].dma_fd = exported.fd;
    }

    void Queue(unsigned int index) {
        v4l2_plane plane;
        std::memset(&plane, 0, sizeof(plane));
        v4l2_buffer buffer;
        std::memset(&buffer, 0, sizeof(buffer));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        buffer.length = 1;
        buffer.m.planes = &plane;
        if (Ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
            throw SystemError("VIDIOC_QBUF");
        }
    }

    void Cleanup() {
        if (fd_ >= 0 && streaming_) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            Ioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        for (Buffer& buffer : buffers_) {
            if (buffer.dma_fd >= 0) {
                close(buffer.dma_fd);
            }
            if (buffer.address != nullptr) {
                munmap(buffer.address, buffer.length);
            }
        }
        buffers_.clear();
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    int fd_ = -1;
    bool streaming_ = false;
    int width_ = 0;
    int height_ = 0;
    int width_stride_ = 0;
    int height_stride_ = 0;
    std::vector<Buffer> buffers_;
};

V4l2Nv12Capture::V4l2Nv12Capture(const std::string& device, int width, int height,
                                 unsigned int buffers)
    : impl_(new Impl(device, width, height, buffers)) {}

V4l2Nv12Capture::~V4l2Nv12Capture() = default;

bool V4l2Nv12Capture::Dequeue(Nv12Frame* frame, int timeout_ms) {
    return impl_->Dequeue(frame, timeout_ms);
}

bool V4l2Nv12Capture::DequeueLatest(Nv12Frame* frame, int timeout_ms) {
    return impl_->DequeueLatest(frame, timeout_ms);
}

void V4l2Nv12Capture::Requeue(const Nv12Frame& frame) { impl_->Requeue(frame); }
int V4l2Nv12Capture::width() const { return impl_->width(); }
int V4l2Nv12Capture::height() const { return impl_->height(); }
int V4l2Nv12Capture::width_stride() const { return impl_->width_stride(); }
int V4l2Nv12Capture::height_stride() const { return impl_->height_stride(); }

}  // namespace rk3568_camera
