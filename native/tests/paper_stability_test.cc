#include "paper_stability.h"
#include "detection_cadence.h"
#include "paper_detection_orientation.h"
#include "structured_feedback.h"

#include <cassert>
#include <cmath>
#include <iostream>

void RunDocumentTransformTests();

namespace {

rk3568_camera::PaperDetection Detection(float shift_x = 0.0F) {
    rk3568_camera::PaperDetection value;
    value.detected = true;
    value.confidence = 0.9F;
    value.frame_width = 1000;
    value.frame_height = 1000;
    value.corners = {{{100.0F + shift_x, 100.0F},
                      {900.0F + shift_x, 100.0F},
                      {900.0F + shift_x, 900.0F},
                      {100.0F + shift_x, 900.0F}}};
    return value;
}

void TestFourUniqueFramesAndSpan() {
    rk3568_camera::PaperStabilityTracker tracker;
    assert(!tracker.UpdatePresent(1, 0, Detection()).triggered);
    assert(tracker.UpdatePresent(2, 60000000ULL, Detection()).prestable);
    assert(!tracker.UpdatePresent(3, 120000000ULL, Detection()).triggered);
    assert(!tracker.UpdatePresent(3, 190000000ULL, Detection()).triggered);
    const auto stable = tracker.UpdatePresent(4, 190000000ULL, Detection());
    assert(stable.triggered);
    assert(stable.observations == 4);
    assert(stable.stable_span_ns == 190000000ULL);
}

void TestMovementRestartsTracking() {
    rk3568_camera::PaperStabilityTracker tracker;
    tracker.UpdatePresent(1, 0, Detection());
    tracker.UpdatePresent(2, 70000000ULL, Detection());
    const auto moved = tracker.UpdatePresent(3, 140000000ULL, Detection(150.0F));
    assert(!moved.triggered);
    assert(moved.observations == 1);
    assert(moved.reason == "paper_moved");
}

void TestLockedUntilRemoval() {
    rk3568_camera::PaperStabilityTracker tracker;
    tracker.UpdatePresent(1, 0, Detection());
    tracker.UpdatePresent(2, 70000000ULL, Detection());
    tracker.UpdatePresent(3, 140000000ULL, Detection());
    assert(tracker.UpdatePresent(4, 210000000ULL, Detection()).triggered);
    assert(!tracker.UpdatePresent(5, 280000000ULL, Detection()).triggered);
    assert(!tracker.UpdateMissing(6, 350000000ULL).rearmed);
    assert(!tracker.UpdateMissing(7, 1800000000ULL).rearmed);
    assert(tracker.UpdateMissing(8, 1850000000ULL).rearmed);
    assert(tracker.state() == rk3568_camera::PaperState::kAbsent);
}

void TestCadenceWithMicrosecondCameraTimestamps() {
    rk3568_camera::DetectionCadence cadence;
    assert(cadence.Due(0));
    cadence.AfterDetection(0, false);
    assert(!cadence.Due(33333000));
    assert(cadence.Due(199998000));
    cadence.AfterDetection(199998000, true);
    // First paper observation immediately changes the next deadline to 15 FPS.
    assert(!cadence.Due(233331000));
    assert(cadence.Due(266664000));
    cadence.AfterDetection(266664000, true);
    assert(cadence.Due(333330000));
    cadence.AfterDetection(333330000, true);
    assert(cadence.Due(399996000));
    // No burst of scheduled work after a long OCR call.
    cadence.AfterDetection(3000000000ULL, true);
    assert(!cadence.Due(3000000000ULL));
    assert(cadence.Due(3066666000ULL));
}

void TestReappearanceCancelsRemovalCountdown() {
    rk3568_camera::PaperStabilityTracker tracker;
    tracker.UpdatePresent(1, 0, Detection());
    tracker.UpdatePresent(2, 70000000ULL, Detection());
    tracker.UpdatePresent(3, 140000000ULL, Detection());
    assert(tracker.UpdatePresent(4, 210000000ULL, Detection()).triggered);
    tracker.UpdateMissing(5, 300000000ULL);
    tracker.UpdatePresent(6, 750000000ULL, Detection());
    assert(!tracker.UpdateMissing(7, 850000000ULL).rearmed);
    assert(!tracker.UpdateMissing(8, 2300000000ULL).rearmed);
    assert(tracker.UpdateMissing(9, 2350000000ULL).rearmed);
}

void TestClockwiseDetectionRemap() {
    rk3568_camera::PaperDetection rotated;
    rotated.detected = true;
    rotated.confidence = 0.84F;
    rotated.frame_width = 600;
    rotated.frame_height = 1000;
    rotated.corners = {{{100.0F, 100.0F},
                        {500.0F, 100.0F},
                        {500.0F, 900.0F},
                        {100.0F, 900.0F}}};
    const auto original = rk3568_camera::RemapDetectionFromClockwiseRotation(
        rotated, 1000, 600, 90);
    assert(original.frame_width == 1000);
    assert(original.frame_height == 600);
    assert(original.confidence == rotated.confidence);
    const auto near = [](float actual, float expected) {
        return std::abs(actual - expected) < 0.01F;
    };
    assert(near(original.corners[0].x, 100.0F) &&
           near(original.corners[0].y, 100.0F));
    assert(near(original.corners[1].x, 900.0F) &&
           near(original.corners[1].y, 100.0F));
    assert(near(original.corners[2].x, 900.0F) &&
           near(original.corners[2].y, 500.0F));
    assert(near(original.corners[3].x, 100.0F) &&
           near(original.corners[3].y, 500.0F));
}

void TestStructuredFeedbackProtocol() {
    rk3568_camera::StructuredFeedback feedback;
    const std::string id = "0123456789abcdef0123456789abcdef";
    assert(rk3568_camera::ParseStructuredFeedback(id + " accepted 8 16.5", &feedback));
    assert(feedback.field_count == 8 && feedback.status == "accepted");
    assert(rk3568_camera::ParseStructuredFeedback(id + " review_required 7 20", &feedback));
    assert(!rk3568_camera::ParseStructuredFeedback(id + " unknown 7 20", &feedback));
    assert(!rk3568_camera::ParseStructuredFeedback(id + " accepted -1 20", &feedback));
    assert(!rk3568_camera::ParseStructuredFeedback("wrong accepted 8 20", &feedback));
    assert(!rk3568_camera::ParseStructuredFeedback(id + " accepted 8 nan", &feedback));
    assert(!rk3568_camera::ParseStructuredFeedback(id + " accepted 8 20 extra", &feedback));
}

}  // namespace

int main() {
    RunDocumentTransformTests();
    TestFourUniqueFramesAndSpan();
    TestMovementRestartsTracking();
    TestLockedUntilRemoval();
    TestCadenceWithMicrosecondCameraTimestamps();
    TestReappearanceCancelsRemovalCountdown();
    TestClockwiseDetectionRemap();
    TestStructuredFeedbackProtocol();
    std::cout << "paper stability tests passed\n";
    return 0;
}
