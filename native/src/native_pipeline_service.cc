#include "docaligner_rknn.h"
#include "detection_cadence.h"
#include "document_transform.h"
#include "frame_quality.h"
#include "npu_frequency_guard.h"
#include "paper_stability.h"
#include "ppocr_engine.h"
#include "rga_preprocessor.h"
#include "v4l2_capture.h"
#include "structured_feedback.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <time.h>

namespace {

using Clock = std::chrono::steady_clock;

std::atomic<bool> g_running(true);
rk3568_camera::DocumentTransformConfig g_transform_config;

void HandleSignal(int) { g_running.store(false); }

double ElapsedMs(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

std::uint64_t MonotonicNs() {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL + now.tv_nsec;
}

int EnvironmentInteger(const char* name, int fallback, int minimum, int maximum) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return static_cast<int>(parsed);
}

double EnvironmentDouble(const char* name, double fallback,
                         double minimum, double maximum) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed) ||
        parsed < minimum || parsed > maximum) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return parsed;
}

rk3568_camera::DocumentTransformConfig LoadTransformConfig() {
    rk3568_camera::DocumentTransformConfig config;
    config.canonical_long_side = EnvironmentInteger(
        "OCR_DOCUMENT_LONG_SIDE", config.canonical_long_side, 256, 8192);
    config.recognition_top = EnvironmentDouble(
        "OCR_REGION_TOP", config.recognition_top, 0.0, 0.999999);
    config.recognition_bottom = EnvironmentDouble(
        "OCR_REGION_BOTTOM", config.recognition_bottom, 0.000001, 1.0);
    config.rotation_degrees = EnvironmentInteger(
        "OCR_ROTATION", config.rotation_degrees, 0, 270);
    if (config.rotation_degrees % 90 != 0 ||
        config.recognition_top >= config.recognition_bottom) {
        throw std::invalid_argument("invalid document transform configuration");
    }
    return config;
}

void AppendRecognitionConfig(std::ostringstream& output) {
    output << ",\"ocr_rotation\":" << g_transform_config.rotation_degrees
           << ",\"ocr_document_long_side\":" << g_transform_config.canonical_long_side
           << ",\"ocr_document_short_side\":" << g_transform_config.canonical_short_side
           << ",\"recognition_region\":{\"enabled\":true,"
           << "\"coordinate_space\":\"rectified_document_normalized\","
           << "\"crop_normalized\":[0.0," << g_transform_config.recognition_top
           << ",1.0," << g_transform_config.recognition_bottom << ']'
           << ",\"accept_normalized\":[0.0," << g_transform_config.recognition_top
           << ",1.0," << g_transform_config.recognition_bottom << "]}";
}

std::string JsonEscape(const std::string& value) {
    std::ostringstream output;
    for (unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

void AtomicWrite(const std::string& path, const std::string& payload, mode_t mode) {
    const std::string temporary = path + ".tmp." + std::to_string(getpid());
    std::remove(temporary.c_str());
    const int descriptor = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                                mode);
    if (descriptor < 0) {
        throw std::runtime_error("cannot create " + temporary);
    }
    const std::string content = payload + '\n';
    std::size_t written = 0;
    while (written < content.size()) {
        const ssize_t count = write(descriptor, content.data() + written,
                                    content.size() - written);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            close(descriptor);
            std::remove(temporary.c_str());
            throw std::runtime_error("cannot write " + temporary);
        }
        written += static_cast<std::size_t>(count);
    }
    bool publish_failed = fchmod(descriptor, mode) != 0;
    if (fsync(descriptor) != 0) publish_failed = true;
    if (close(descriptor) != 0) publish_failed = true;
    if (!publish_failed && std::rename(temporary.c_str(), path.c_str()) != 0) {
        publish_failed = true;
    }
    if (publish_failed) {
        std::remove(temporary.c_str());
        throw std::runtime_error("cannot publish " + path);
    }
}

void NotifyStructuredWorker(const std::string& socket_path,
                            const std::string& capture_id) {
    if (socket_path.empty()) return;
    const int descriptor = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (descriptor < 0) return;
    sockaddr_un address;
    std::memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(address.sun_path)) {
        close(descriptor);
        return;
    }
    std::strncpy(address.sun_path, socket_path.c_str(), sizeof(address.sun_path) - 1U);
    sendto(descriptor, capture_id.data(), capture_id.size(), MSG_DONTWAIT,
           reinterpret_cast<const sockaddr*>(&address), sizeof(address));
    close(descriptor);
}

std::string CaptureId(unsigned int sequence) {
    const auto ticks = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    std::ostringstream output;
    output << std::hex << std::setfill('0')
           << std::setw(16) << static_cast<unsigned long long>(ticks)
           << std::setw(8) << sequence
           << std::setw(8) << static_cast<unsigned int>(getpid());
    return output.str();
}

const char* LegacyState(const std::string& stage) {
    if (stage == "waiting_paper" || stage == "stopped") return "absent";
    if (stage == "detecting_stability") return "tracking";
    return "locked";
}

const char* LegacyCaptureStage(const std::string& stage) {
    if (stage == "waiting_paper" || stage == "stopped") return "absent";
    if (stage == "detecting_stability") return "tracking";
    if (stage == "ocr_running") return "ocr_primary";
    if (stage == "structuring") return "structuring";
    if (stage == "structured_complete") return "completed";
    if (stage == "structured_rejected") return "reposition_required";
    if (stage == "capture_error") return "ocr_error";
    if (stage == "quality_rejected") return "burst_rejected";
    if (stage == "ocr_rejected") return "reposition_required";
    if (stage == "ocr_complete") return "completed";
    return "locked";
}

void AppendDetection(std::ostringstream& output,
                     const rk3568_camera::PaperDetection* detection) {
    const bool present = detection != nullptr && detection->detected;
    output << ",\"paper_detected\":" << (present ? "true" : "false")
           << ",\"paper_confidence\":" << (present ? detection->confidence : 0.0F)
           << ",\"paper_inference_ms\":" << (present ? detection->inference_ms : 0.0)
           << ",\"frame_size\":{\"width\":"
           << (present ? detection->frame_width : 3840)
           << ",\"height\":" << (present ? detection->frame_height : 2160)
           << "},\"paper_corners\":[";
    if (present) {
        for (std::size_t index = 0; index < detection->corners.size(); ++index) {
            if (index > 0) output << ',';
            output << '[' << detection->corners[index].x << ','
                   << detection->corners[index].y << ']';
        }
    }
    output << ']';
}

std::string StagePayload(const std::string& stage, const std::string& reason,
                         unsigned int sequence, int observations,
                         double stage_elapsed_ms = 0.0,
                         const rk3568_camera::PaperDetection* detection = nullptr,
                         const std::string& capture_id = std::string()) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    output << "{\"ok\":true,\"stage\":\"" << JsonEscape(stage)
           << "\",\"reason\":\"" << JsonEscape(reason)
           << "\",\"sequence\":" << sequence
           << ",\"state\":\"" << LegacyState(stage)
           << "\",\"capture_stage\":\"" << LegacyCaptureStage(stage)
           << "\",\"capture_id\":\"" << JsonEscape(capture_id) << "\""
           << ",\"stable_observations\":" << observations
           << ",\"stable_for\":" << stage_elapsed_ms / 1000.0
           << ",\"stable_target_seconds\":0.180"
           << ",\"burst_target_frames\":1"
           << ",\"stage_elapsed_ms\":" << stage_elapsed_ms
           << ",\"text_only\":true";
    AppendRecognitionConfig(output);
    AppendDetection(output, detection);
    output << ",\"full_text\":{\"available\":false,\"status\":\"waiting\"}"
           << ",\"updated_at_ms\":"
           << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count()
           << '}';
    return output.str();
}

std::string OcrPrivatePayload(const std::string& capture_id,
                              const rk3568_camera::OcrResult& result,
                              const cv::Size& image_size, double mean_confidence,
                              const rk3568_camera::PaperDetection& detection,
                              double stable_ms, double quality_ms, double crop_ms,
                              double transform_ms, double post_stable_ms) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(4);
    output << "{\"schema_version\":1,\"status\":\"ocr_complete\",\"capture_id\":\""
           << JsonEscape(capture_id) << "\",\"image_size\":[" << image_size.width << ','
           << image_size.height << "],\"item_count\":" << result.items.size()
           << ",\"mean_confidence\":" << mean_confidence
           << ",\"source\":{\"frame_size\":{\"width\":" << detection.frame_width
           << ",\"height\":" << detection.frame_height << "},\"paper_corners\":[";
    for (std::size_t index = 0; index < detection.corners.size(); ++index) {
        if (index > 0) output << ',';
        output << '[' << detection.corners[index].x << ',' << detection.corners[index].y << ']';
    }
    output << "],\"recognition_mode\":\"fixed_document_region\"";
    AppendRecognitionConfig(output);
    output << "},\"ocr\":[";
    for (std::size_t index = 0; index < result.items.size(); ++index) {
        if (index > 0) output << ',';
        const auto& item = result.items[index];
        int left = image_size.width;
        int top = image_size.height;
        int right = 0;
        int bottom = 0;
        for (const cv::Point& point : item.polygon) {
            left = std::min(left, point.x);
            top = std::min(top, point.y);
            right = std::max(right, point.x);
            bottom = std::max(bottom, point.y);
        }
        output << "{\"text\":\"" << JsonEscape(item.text) << "\",\"score\":"
               << item.score << ",\"box\":[" << left << ',' << top << ',' << right << ','
               << bottom << "],\"polygon\":[";
        for (std::size_t point_index = 0; point_index < item.polygon.size(); ++point_index) {
            if (point_index > 0) output << ',';
            output << '[' << item.polygon[point_index].x << ','
                   << item.polygon[point_index].y << ']';
        }
        output << "],\"recognition_source\":\"primary\"}";
    }
    output << "],\"timings\":{\"stability_ms\":" << stable_ms
           << ",\"quality_ms\":" << quality_ms
           << ",\"crop_ms\":" << crop_ms
           << ",\"transform_ms\":" << transform_ms
           << ",\"ocr_ms\":" << result.elapsed_ms
           << ",\"ocr_detection_ms\":" << result.detection_ms
           << ",\"ocr_crop_ms\":" << result.crop_ms
           << ",\"ocr_recognition_preprocess_ms\":"
           << result.recognition_preprocess_ms
           << ",\"ocr_recognition_inference_ms\":"
           << result.recognition_inference_ms
           << ",\"ocr_recognition_postprocess_ms\":"
           << result.recognition_postprocess_ms
           << ",\"post_stable_ms\":" << post_stable_ms
           << ",\"paper_to_ocr_ms\":" << stable_ms + post_stable_ms << "}}";
    return output.str();
}

std::string CompletionPayload(const std::string& stage, const std::string& reason,
                              const std::string& capture_id, unsigned int sequence,
                              const rk3568_camera::FrameQualityResult& quality,
                              const rk3568_camera::OcrResult* result,
                              double stable_ms, double quality_ms, double crop_ms,
                              double transform_ms, double end_to_end_ms,
                              double mean_confidence,
                              const rk3568_camera::PaperDetection& detection,
                              const rk3568_camera::StructuredFeedback* feedback = nullptr) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    output << "{\"ok\":true,\"stage\":\"" << JsonEscape(stage)
           << "\",\"reason\":\"" << JsonEscape(reason)
           << "\",\"capture_id\":\"" << JsonEscape(capture_id)
           << "\",\"sequence\":" << sequence
           << ",\"state\":\"locked\",\"capture_stage\":\""
           << LegacyCaptureStage(stage) << "\",\"stable_for\":" << stable_ms / 1000.0
           << ",\"stable_target_seconds\":0.180,\"burst_target_frames\":1"
           << ",\"text_only\":true";
    AppendRecognitionConfig(output);
    AppendDetection(output, &detection);
    output << ",\"full_text\":{\"available\":"
           << (result != nullptr && !result->items.empty() ? "true" : "false")
           << ",\"status\":\""
           << (result == nullptr ? "rejected" : result->items.empty() ? "rejected" : "accepted")
           << "\",\"item_count\":" << (result == nullptr ? 0 : result->items.size())
           << ",\"mean_confidence\":" << mean_confidence
           << ",\"elapsed_ms\":" << (result == nullptr ? 0.0 : result->elapsed_ms) << '}'
           << ",\"quality\":{\"accepted\":" << (quality.accepted ? "true" : "false")
           << ",\"sharpness\":" << quality.sharpness
           << ",\"contrast\":" << quality.contrast
           << ",\"glare_ratio\":" << quality.glare_ratio << "},\"timings\":{"
           << "\"stability_ms\":" << stable_ms << ",\"quality_ms\":" << quality_ms
           << ",\"crop_ms\":" << crop_ms << ",\"transform_ms\":" << transform_ms;
    if (result != nullptr) {
        output << ",\"ocr_total_ms\":" << result->elapsed_ms
               << ",\"ocr_detection_ms\":" << result->detection_ms
               << ",\"ocr_crop_ms\":" << result->crop_ms
               << ",\"ocr_recognition_preprocess_ms\":"
               << result->recognition_preprocess_ms
               << ",\"ocr_recognition_inference_ms\":"
               << result->recognition_inference_ms
               << ",\"ocr_recognition_postprocess_ms\":"
               << result->recognition_postprocess_ms;
    }
    if (feedback != nullptr) {
        output << ",\"structured_ms\":" << feedback->structured_ms;
    }
    output << ",\"post_stable_ms\":" << end_to_end_ms
           << ",\"paper_to_ocr_ms\":" << stable_ms + end_to_end_ms
           << ",\"end_to_end_ms\":" << stable_ms + end_to_end_ms +
                (feedback == nullptr ? 0.0 : feedback->structured_ms)
           << "},\"structured\":{\"status\":\""
           << (feedback == nullptr ? "pending" : feedback->status)
           << "\",\"field_count\":" << (feedback == nullptr ? 0 : feedback->field_count)
           << "},\"ocr\":{"
           << "\"item_count\":" << (result == nullptr ? 0 : result->items.size())
           << ",\"recognition_count\":"
           << (result == nullptr ? 0 : result->recognition_count)
           << ",\"mean_confidence\":" << mean_confidence
           << "},\"privacy\":{\"ocr_text_emitted\":false,\"image_saved\":false}}";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 8) {
        std::cerr << "usage: " << argv[0]
                  << " DOCALIGNER.rknn DET.rknn REC.rknn [DEVICE] [STATUS_JSON] [OCR_JSON] [STRUCTURED_SOCKET]\n";
        return 2;
    }
    const std::string device = argc >= 5 ? argv[4] : "/dev/video0";
    const std::string status_path = argc >= 6
                                        ? argv[5]
                                        : "/run/rk3568-camera/native-status.json";
    const std::string ocr_path = argc >= 7
                                     ? argv[6]
                                     : "/run/rk3568-camera/native-ocr.json";
    const std::string structured_socket =
        argc >= 8 ? argv[7] : "/run/rk3568-camera/structured.sock";
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    try {
        g_transform_config = LoadTransformConfig();
        rk3568_camera::DocAlignerRknn detector(
            argv[1], 0.5F, rk3568_camera::DocAlignerInputMode::kUint8);
        rk3568_camera::PpOcrEngine ocr(argv[2], argv[3]);
        rk3568_camera::RgaPreprocessor rga;
        rk3568_camera::V4l2Nv12Capture capture(device, 3840, 2160, 4);
        rk3568_camera::PaperStabilityTracker tracker;
        rk3568_camera::DetectionCadence cadence;
        rk3568_camera::StructuredFeedbackReceiver feedback_receiver(structured_socket + ".result");
        std::unique_ptr<rk3568_camera::NpuFrequencyGuard> frequency_guard;
        std::uint64_t first_present_ns = 0;
        bool first_present_recorded = false;
        bool terminal_status_visible = false;
        std::string active_capture_id;
        std::string terminal_payload;
        std::uint64_t next_terminal_heartbeat_ns = 0;
        std::function<std::string(const rk3568_camera::StructuredFeedback&)> finish_structured;
        Clock::time_point structured_started;

        AtomicWrite(status_path, StagePayload("waiting_paper", "service_started", 0, 0), 0644);
        while (g_running.load()) {
            rk3568_camera::Nv12Frame frame;
            if (!capture.DequeueLatest(&frame, 1000)) {
                continue;
            }
            bool requeued = false;
            bool processing_capture = false;
            std::string capture_operation;
            rk3568_camera::PaperDetection detection;
            try {
                if (finish_structured) {
                    rk3568_camera::StructuredFeedback feedback;
                    bool ready = feedback_receiver.Receive(active_capture_id, &feedback);
                    if (!ready && ElapsedMs(structured_started) > 5000.0) {
                        feedback.capture_id = active_capture_id;
                        feedback.status = "error";
                        ready = true;
                    }
                    if (ready) {
                        terminal_payload = finish_structured(feedback);
                        finish_structured = {};
                        AtomicWrite(status_path, terminal_payload, 0644);
                    }
                }
                const std::uint64_t now_ns = MonotonicNs();
                const bool stale_frame = frame.monotonic_timestamp && now_ns > frame.timestamp_ns &&
                                         now_ns - frame.timestamp_ns > 200000000ULL;
                if (!frame.valid || stale_frame || !cadence.Due(frame.timestamp_ns)) {
                    capture.Requeue(frame);
                    requeued = true;
                    continue;
                }
                cv::Mat detector_input = rga.ResizeNv12ToBgr(frame, 256, 256);
                detection = detector.DetectBgr(detector_input, frame.width, frame.height);
                rk3568_camera::StabilityUpdate stability;
                if (detection.detected) {
                    stability = tracker.UpdatePresent(frame.sequence, frame.timestamp_ns, detection);
                    if (!first_present_recorded) {
                        first_present_ns = frame.timestamp_ns;
                        first_present_recorded = true;
                    }
                    if (!frequency_guard &&
                        tracker.state() != rk3568_camera::PaperState::kLocked) {
                        frequency_guard.reset(new rk3568_camera::NpuFrequencyGuard(900000000UL));
                    }
                } else {
                    stability = tracker.UpdateMissing(frame.sequence, frame.timestamp_ns);
                    if (tracker.state() == rk3568_camera::PaperState::kAbsent) {
                        frequency_guard.reset();
                        first_present_recorded = false;
                        first_present_ns = 0;
                        terminal_status_visible = false;
                        active_capture_id.clear();
                        terminal_payload.clear();
                        next_terminal_heartbeat_ns = 0;
                        finish_structured = {};
                    }
                }
                cadence.AfterDetection(frame.timestamp_ns,
                    tracker.state() != rk3568_camera::PaperState::kAbsent);

                if (!stability.triggered) {
                    if (terminal_status_visible &&
                        tracker.state() == rk3568_camera::PaperState::kLocked) {
                        if (!terminal_payload.empty() &&
                            frame.timestamp_ns >= next_terminal_heartbeat_ns) {
                            AtomicWrite(status_path, terminal_payload, 0644);
                            next_terminal_heartbeat_ns = frame.timestamp_ns + 1000000000ULL;
                        }
                        capture.Requeue(frame);
                        requeued = true;
                        continue;
                    }
                    const std::string stage =
                        tracker.state() == rk3568_camera::PaperState::kAbsent
                            ? "waiting_paper"
                            : tracker.state() == rk3568_camera::PaperState::kLocked
                                  ? "waiting_removal"
                                  : "detecting_stability";
                    AtomicWrite(status_path,
                                StagePayload(stage, stability.reason, frame.sequence,
                                             stability.observations,
                                             stability.stable_span_ns / 1000000.0,
                                             detection.detected ? &detection : nullptr,
                                             active_capture_id),
                                0644);
                    capture.Requeue(frame);
                    requeued = true;
                    continue;
                }

                const std::string capture_id = CaptureId(frame.sequence);
                active_capture_id = capture_id;
                processing_capture = true;
                const auto workflow_started = Clock::now();
                AtomicWrite(status_path,
                            StagePayload("checking_quality", "paper_stable", frame.sequence,
                                         stability.observations,
                                         stability.stable_span_ns / 1000000.0,
                                         &detection, capture_id),
                            0644);

                auto stage_started = Clock::now();
                capture_operation = "quality";
                const rk3568_camera::FrameQualityResult quality =
                    rk3568_camera::EvaluateFinalFrameNv12(frame, detection);
                const double quality_ms = ElapsedMs(stage_started);
                const double stable_ms = first_present_ns > 0
                                             ? (frame.timestamp_ns - first_present_ns) / 1000000.0
                                             : stability.stable_span_ns / 1000000.0;
                if (!quality.accepted) {
                    capture.Requeue(frame);
                    requeued = true;
                    frequency_guard.reset();
                    terminal_payload = CompletionPayload(
                        "quality_rejected", "final_frame_quality", capture_id,
                        frame.sequence, quality, nullptr, stable_ms, quality_ms, 0.0,
                        0.0, ElapsedMs(workflow_started), 0.0, detection);
                    AtomicWrite(status_path, terminal_payload, 0644);
                    terminal_status_visible = true;
                    next_terminal_heartbeat_ns = frame.timestamp_ns + 1000000000ULL;
                    continue;
                }

                capture_operation = "transform_plan";
                const rk3568_camera::DocumentTransformPlan transform =
                    rk3568_camera::BuildDocumentTransform(detection, g_transform_config);
                stage_started = Clock::now();
                capture_operation = "rga_crop";
                cv::Mat paper = rga.CropNv12ToBgr(frame, transform.source_bounds);
                const double crop_ms = ElapsedMs(stage_started);
                capture.Requeue(frame);
                requeued = true;

                stage_started = Clock::now();
                capture_operation = "perspective_warp";
                cv::Mat region_bgr = rk3568_camera::WarpDocumentRegion(paper, transform);
                cv::Mat region_rgb;
                cv::cvtColor(region_bgr, region_rgb, cv::COLOR_BGR2RGB);
                const double transform_ms = ElapsedMs(stage_started);
                AtomicWrite(status_path,
                            StagePayload("ocr_running", "final_frame_accepted", frame.sequence,
                                         stability.observations, ElapsedMs(workflow_started),
                                         &detection, capture_id),
                            0644);

                capture_operation = "ocr";
                const rk3568_camera::OcrResult result = ocr.RecognizeRgb(region_rgb);
                double score_sum = 0.0;
                for (const auto& item : result.items) score_sum += item.score;
                const double mean_confidence = result.items.empty()
                                                   ? 0.0
                                                   : score_sum / result.items.size();
                const double post_stable_ms = ElapsedMs(workflow_started);
                capture_operation = "publish_ocr";
                AtomicWrite(
                    ocr_path,
                    OcrPrivatePayload(capture_id, result, region_rgb.size(),
                                      mean_confidence, detection, stable_ms, quality_ms,
                                      crop_ms, transform_ms, post_stable_ms),
                    0600);
                NotifyStructuredWorker(structured_socket, capture_id);
                frequency_guard.reset();
                const std::string final_stage = result.items.empty()
                                                    ? "ocr_rejected"
                                                    : "structuring";
                terminal_payload = CompletionPayload(
                    final_stage, result.items.empty() ? "no_ocr_text" : "field_validation_pending",
                    capture_id, frame.sequence, quality, &result, stable_ms, quality_ms,
                    crop_ms, transform_ms, post_stable_ms, mean_confidence,
                    detection);
                if (!result.items.empty()) {
                    structured_started = Clock::now();
                    finish_structured = [=](const rk3568_camera::StructuredFeedback& feedback) {
                        const bool accepted = feedback.status == "accepted";
                        return CompletionPayload(
                            accepted ? "structured_complete" : "structured_rejected",
                            accepted ? "accepted" : "remove_and_reposition",
                            capture_id, frame.sequence, quality, &result, stable_ms, quality_ms,
                            crop_ms, transform_ms, post_stable_ms, mean_confidence,
                            detection, &feedback);
                    };
                }
                AtomicWrite(status_path, terminal_payload, 0644);
                terminal_status_visible = true;
                next_terminal_heartbeat_ns = frame.timestamp_ns + 1000000000ULL;
            } catch (const std::exception& error) {
                if (!requeued) capture.Requeue(frame);
                if (!processing_capture) throw;
                // A failed final frame is terminal; restarting here would retry the same paper.
                frequency_guard.reset();
                finish_structured = {};
                std::cerr << "capture failed at " << capture_operation << ": "
                          << error.what() << std::endl;
                terminal_payload = StagePayload("capture_error", capture_operation + "_failed",
                    frame.sequence, 0, 0.0, &detection, active_capture_id);
                AtomicWrite(status_path, terminal_payload, 0644);
                terminal_status_visible = true;
                next_terminal_heartbeat_ns = frame.timestamp_ns + 1000000000ULL;
            }
        }
        frequency_guard.reset();
        AtomicWrite(status_path, StagePayload("stopped", "signal", 0, 0), 0644);
        return 0;
    } catch (const std::exception& error) {
        try {
            AtomicWrite(status_path,
                        "{\"ok\":false,\"stage\":\"error\",\"error\":\"" +
                            JsonEscape(error.what()) + "\"}",
                        0644);
        } catch (...) {
        }
        std::cerr << "native pipeline failed: " << error.what() << '\n';
        return 1;
    }
}
