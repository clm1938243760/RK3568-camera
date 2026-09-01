#include "rga_preprocessor.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include <im2d.h>
#include <rga.h>

namespace rk3568_camera {

cv::Mat RgaPreprocessor::ResizeNv12ToBgr(const Nv12Frame& frame, int width, int height) const {
    if (frame.dma_fd < 0 || frame.width <= 0 || frame.height <= 0 || width <= 0 || height <= 0) {
        throw std::invalid_argument("invalid RGA preprocessing frame");
    }
    cv::Mat output(height, width, CV_8UC3);
    rga_buffer_t source = wrapbuffer_fd(frame.dma_fd, frame.width, frame.height,
                                        RK_FORMAT_YCbCr_420_SP, frame.width_stride,
                                        frame.height_stride);
    rga_buffer_t destination = wrapbuffer_virtualaddr(output.data, width, height,
                                                       RK_FORMAT_BGR_888, width, height);
    im_rect source_rect = {0, 0, frame.width, frame.height};
    im_rect destination_rect = {0, 0, width, height};
    im_rect pattern_rect = {0, 0, 0, 0};
    rga_buffer_t pattern;
    std::memset(&pattern, 0, sizeof(pattern));
    const IM_STATUS status = improcess(source, destination, pattern, source_rect,
                                       destination_rect, pattern_rect, 0);
    if (status <= 0) {
        throw std::runtime_error(std::string("RGA NV12 preprocessing failed: ") +
                                 imStrError(status));
    }
    return output;
}

cv::Mat RgaPreprocessor::CropNv12ToBgr(const Nv12Frame& frame, const cv::Rect& crop) const {
    if (frame.dma_fd < 0 || frame.width <= 0 || frame.height <= 0 || crop.width <= 0 ||
        crop.height <= 0 || crop.x < 0 || crop.y < 0 || crop.x + crop.width > frame.width ||
        crop.y + crop.height > frame.height || (crop.x | crop.y | crop.width | crop.height) % 2 != 0) {
        throw std::invalid_argument("invalid even-aligned RGA NV12 crop");
    }
    const int output_stride = (crop.width + 3) & ~3;
    cv::Mat storage(crop.height, output_stride, CV_8UC3);
    cv::Mat output = storage(cv::Rect(0, 0, crop.width, crop.height));
    rga_buffer_t source = wrapbuffer_fd(frame.dma_fd, frame.width, frame.height,
                                        RK_FORMAT_YCbCr_420_SP, frame.width_stride,
                                        frame.height_stride);
    rga_buffer_t destination = wrapbuffer_virtualaddr(
        output.data, crop.width, crop.height, RK_FORMAT_BGR_888, output_stride, crop.height);
    im_rect source_rect = {crop.x, crop.y, crop.width, crop.height};
    im_rect destination_rect = {0, 0, crop.width, crop.height};
    im_rect pattern_rect = {0, 0, 0, 0};
    rga_buffer_t pattern;
    std::memset(&pattern, 0, sizeof(pattern));
    const IM_STATUS status = improcess(source, destination, pattern, source_rect,
                                       destination_rect, pattern_rect, 0);
    if (status <= 0) {
        throw std::runtime_error(std::string("RGA NV12 crop failed: ") + imStrError(status));
    }
    return output;
}

}  // namespace rk3568_camera
