#!/usr/bin/env python3
"""Mirror an HTTP JPEG endpoint into an atomic rotating snapshot set."""

from __future__ import print_function

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"


def normalize_jpeg(data, minimum_bytes=4096):
    if not isinstance(data, bytes) or len(data) < minimum_bytes:
        raise ValueError("frame is smaller than minimum_bytes")
    if not data.startswith(JPEG_START):
        raise ValueError("frame does not start with JPEG SOI")
    end = data.rfind(JPEG_END)
    if end < 2 or len(data) - (end + 2) > 4096:
        raise ValueError("frame does not contain a complete JPEG EOI")
    return data[: end + 2]


def atomic_write(path, data, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(str(temporary), mode)
    os.replace(str(temporary), str(path))


def write_json(path, payload):
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"
    atomic_write(path, raw)


class SnapshotBridge(object):
    def __init__(
        self,
        endpoint,
        output_pattern,
        status_file,
        slots=4,
        timeout=3.0,
        minimum_bytes=4096,
        opener=urlopen,
    ):
        if slots < 2:
            raise ValueError("slots must be at least 2")
        if "%d" not in output_pattern:
            raise ValueError("output_pattern must contain %d")
        self.endpoint = endpoint
        self.output_pattern = output_pattern
        self.status_file = Path(status_file)
        self.slots = int(slots)
        self.timeout = float(timeout)
        self.minimum_bytes = int(minimum_bytes)
        self.opener = opener
        self.sequence = 0
        self.published = 0
        self.failures = 0
        self.started_at = time.time()

    def fetch(self):
        request = Request(
            self.endpoint,
            headers={"Accept": "image/jpeg", "User-Agent": "RK3568-Camera-Bridge/0.1"},
        )
        started = time.monotonic()
        with self.opener(request, timeout=self.timeout) as response:
            content_type = str(response.headers.get("Content-Type") or "").lower()
            data = response.read()
        if content_type and "image/jpeg" not in content_type:
            raise ValueError("frame endpoint returned %s" % content_type)
        return normalize_jpeg(data, self.minimum_bytes), (time.monotonic() - started) * 1000.0

    def publish(self, image_bytes, fetch_ms, now=None):
        now = time.time() if now is None else float(now)
        slot = self.sequence % self.slots
        output = Path(self.output_pattern % slot)
        atomic_write(output, image_bytes)
        self.sequence += 1
        self.published += 1
        payload = {
            "ok": True,
            "updated_at": now,
            "started_at": self.started_at,
            "published_frames": self.published,
            "failures": self.failures,
            "slot": slot,
            "frame_bytes": len(image_bytes),
            "fetch_ms": round(float(fetch_ms), 2),
            "output": str(output),
        }
        write_json(self.status_file, payload)
        return payload

    def publish_error(self, error, now=None):
        self.failures += 1
        payload = {
            "ok": False,
            "updated_at": time.time() if now is None else float(now),
            "started_at": self.started_at,
            "published_frames": self.published,
            "failures": self.failures,
            "error": "%s: %s" % (type(error).__name__, str(error)),
        }
        write_json(self.status_file, payload)
        return payload

    def run_once(self):
        try:
            image, fetch_ms = self.fetch()
            return self.publish(image, fetch_ms)
        except (HTTPError, URLError, OSError, ValueError) as exc:
            self.publish_error(exc)
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="RK3568 HTTP JPEG to rotating snapshot bridge")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8090/api/frame.jpg?quality=95",
    )
    parser.add_argument("--output-pattern", default="/tmp/rk3568_camera_ocr_%d.jpg")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("/run/rk3568-camera/snapshot-bridge.json"),
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--minimum-bytes", type=int, default=4096)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0 or args.fps > 30:
        raise SystemExit("--fps must be in the range (0, 30]")
    bridge = SnapshotBridge(
        args.endpoint,
        args.output_pattern,
        args.status_file,
        slots=args.slots,
        timeout=args.timeout,
        minimum_bytes=args.minimum_bytes,
    )
    if args.once:
        print(json.dumps(bridge.run_once(), sort_keys=True))
        return 0

    stopping = [False]

    def stop(_signum, _frame):
        stopping[0] = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    interval = 1.0 / args.fps
    next_run = time.monotonic()
    while not stopping[0]:
        try:
            bridge.run_once()
        except (HTTPError, URLError, OSError, ValueError) as exc:
            print("snapshot bridge error: %s" % exc, file=sys.stderr, flush=True)
        next_run += interval
        delay = next_run - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_run = time.monotonic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
