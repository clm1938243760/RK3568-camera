#include "rga_preprocessor.h"
#include "v4l2_capture.h"

#include <iostream>
#include <stdexcept>
#include <opencv2/core.hpp>

int main(int argc, char** argv) {
    try {
        rk3568_camera::V4l2Nv12Capture capture(argc > 1 ? argv[1] : "/dev/video0", 3840, 2160);
        rk3568_camera::Nv12Frame frame;
        if (!capture.DequeueLatest(&frame, 2000) || !frame.valid) {
            throw std::runtime_error("no valid test frame");
        }
        rk3568_camera::RgaPreprocessor rga;
        for (int width : {1770, 1918, 1920}) {
            cv::Mat actual = rga.CropNv12ToBgr(frame, cv::Rect(100, 100, width, 400));
            const int stride = (width + 3) & ~3;
            if (actual.cols != width || actual.rows != 400 || actual.step != stride * 3U) {
                throw std::runtime_error("crop shape or padded stride mismatch");
            }
            cv::Mat reference = rga.CropNv12ToBgr(frame, cv::Rect(100, 100, stride, 400));
            cv::Mat difference;
            cv::absdiff(actual, reference(cv::Rect(0, 0, width, 400)), difference);
            double maximum = 0.0;
            cv::minMaxLoc(difference.reshape(1), nullptr, &maximum);
            if (maximum > 1.0) throw std::runtime_error("padding changed crop pixels");
        }
        capture.Requeue(frame);
        std::cout << "{\"ok\":true,\"tested_widths\":[1770,1918,1920],"
                     "\"pixel_identity\":true,\"image_saved\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RGA crop probe failed: " << error.what() << '\n';
        return 1;
    }
}
