#!/usr/bin/env python3
"""Serve a low-rate JPEG preview from the RKISP secondary NV12 path."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import threading
import time
from typing import Any, Dict, Optional

_DEVICE_PATTERN = re.compile(r"/dev/video[0-9]+\Z")


def build_pipeline_description(
    device: str,
    width: int,
    height: int,
    source_fps: int,
    preview_fps: int,
    jpeg_quant: int,
) -> str:
    if _DEVICE_PATTERN.fullmatch(device) is None:
        raise ValueError("device must be a /dev/videoN path")
    if width < 16 or height < 16 or width > 7680 or height > 4320:
        raise ValueError("preview dimensions are out of range")
    if source_fps < 1 or preview_fps < 1 or preview_fps > source_fps:
        raise ValueError("preview frame rates are invalid")
    if jpeg_quant < 1 or jpeg_quant > 10:
        raise ValueError("jpeg quant must be from 1 to 10")
    return (
        "v4l2src device={device} io-mode=2 ! "
        "video/x-raw,format=NV12,width={width},height={height},framerate={source_fps}/1 ! "
        "videorate drop-only=true ! "
        "video/x-raw,format=NV12,framerate={preview_fps}/1 ! "
        "mppjpegenc quant={jpeg_quant} ! "
        "appsink name=preview_sink emit-signals=true sync=false max-buffers=1 drop=true"
    ).format(
        device=device,
        width=width,
        height=height,
        source_fps=source_fps,
        preview_fps=preview_fps,
        jpeg_quant=jpeg_quant,
    )


class LatestJpeg:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._data = b""
        self._updated_at = 0.0
        self._sequence = 0

    def publish(self, data: bytes, now: Optional[float] = None) -> bool:
        if len(data) < 4 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
            return False
        with self._condition:
            self._data = bytes(data)
            self._updated_at = time.monotonic() if now is None else now
            self._sequence += 1
            self._condition.notify_all()
        return True

    def snapshot(self, now: Optional[float] = None) -> Dict[str, Any]:
        with self._condition:
            current = time.monotonic() if now is None else now
            age_ms = max(0.0, (current - self._updated_at) * 1000.0) if self._updated_at else None
            return {
                "data": self._data,
                "age_ms": age_ms,
                "sequence": self._sequence,
            }

    def wait_ready(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while not self._data:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True


def notify_systemd_ready() -> None:
    address = os.environ.get("NOTIFY_SOCKET", "")
    if not address:
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    notifier = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        notifier.connect(address)
        notifier.sendall(b"READY=1\nSTATUS=RKISP preview frame ready")
    finally:
        notifier.close()


class GstPreview:
    def __init__(self, description: str, latest: LatestJpeg) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self.Gst = Gst
        self.latest = latest
        self.pipeline = Gst.parse_launch(description)
        self.sink = self.pipeline.get_by_name("preview_sink")
        if self.sink is None:
            raise RuntimeError("preview appsink is missing")
        self.sink.connect("new-sample", self._on_sample)
        self._monitor = threading.Thread(target=self._monitor_bus, daemon=True)

    def start(self) -> None:
        if self.pipeline.set_state(self.Gst.State.PLAYING) == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("preview pipeline failed to start")
        self._monitor.start()

    def stop(self) -> None:
        self.pipeline.set_state(self.Gst.State.NULL)

    def _on_sample(self, sink: Any) -> Any:
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        ok, mapping = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.ERROR
        try:
            self.latest.publish(bytes(mapping.data))
        finally:
            buffer.unmap(mapping)
        return self.Gst.FlowReturn.OK

    def _monitor_bus(self) -> None:
        message = self.pipeline.get_bus().timed_pop_filtered(
            self.Gst.CLOCK_TIME_NONE,
            self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS,
        )
        if message is not None:
            detail = "end_of_stream"
            if message.type == self.Gst.MessageType.ERROR:
                error, _debug = message.parse_error()
                detail = type(error).__name__
            print("native preview stopped: %s" % detail, flush=True)
            os._exit(1)


def build_app(latest: LatestJpeg, stale_seconds: float) -> Any:
    from aiohttp import web

    async def frame(_request: web.Request) -> web.Response:
        value = latest.snapshot()
        if not value["data"] or value["age_ms"] is None or value["age_ms"] > stale_seconds * 1000.0:
            return web.json_response({"ok": False, "error": "preview_unavailable"}, status=503)
        return web.Response(
            body=value["data"],
            content_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, max-age=0",
                "X-Frame-Sequence": str(value["sequence"]),
                "X-Frame-Age-Ms": "%.1f" % value["age_ms"],
            },
        )

    async def health(_request: web.Request) -> web.Response:
        value = latest.snapshot()
        ready = bool(value["data"] and value["age_ms"] is not None and value["age_ms"] <= stale_seconds * 1000.0)
        return web.Response(
            text=json.dumps(
                {
                    "ok": ready,
                    "sequence": value["sequence"],
                    "age_ms": None if value["age_ms"] is None else round(value["age_ms"], 1),
                },
                sort_keys=True,
            ),
            status=200 if ready else 503,
            content_type="application/json",
        )

    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_get("/api/frame.jpg", frame)
    app.router.add_get("/health", health)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--preview-fps", type=int, default=5)
    parser.add_argument("--jpeg-quant", type=int, default=8)
    parser.add_argument("--stale-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    from aiohttp import web

    args = parse_args()
    if args.stale_seconds <= 0.0 or args.stale_seconds > 30.0:
        raise ValueError("stale seconds are out of range")
    description = build_pipeline_description(
        args.device,
        args.width,
        args.height,
        args.source_fps,
        args.preview_fps,
        args.jpeg_quant,
    )
    latest = LatestJpeg()
    preview = GstPreview(description, latest)
    preview.start()
    try:
        if not latest.wait_ready(5.0):
            raise RuntimeError("preview did not produce a frame during startup")
        notify_systemd_ready()
        web.run_app(build_app(latest, args.stale_seconds), host=args.host, port=args.port)
    finally:
        preview.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
