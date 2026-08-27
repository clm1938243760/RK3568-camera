import importlib.util
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rk3568_camera_ocr_overlay",
    ROOT / "camera_ocr_overlay.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


JPEG = b"\xff\xd8" + (b"frame" * 1024) + b"\xff\xd9"


class FrameSourceHandler(BaseHTTPRequestHandler):
    payload = JPEG

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, fmt, *args):
        return


class FrameProxyTests(unittest.TestCase):
    def setUp(self):
        FrameSourceHandler.payload = JPEG
        self.source = ThreadingHTTPServer(("127.0.0.1", 0), FrameSourceHandler)
        self.source_thread = threading.Thread(target=self.source.serve_forever)
        self.source_thread.daemon = True
        self.source_thread.start()

        endpoint = "http://127.0.0.1:%d/frame.jpg" % self.source.server_port
        self.proxy = MODULE.Server(
            ("127.0.0.1", 0),
            None,
            None,
            None,
            frame_endpoint=endpoint,
        )
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever)
        self.proxy_thread.daemon = True
        self.proxy_thread.start()

    def tearDown(self):
        self.proxy.shutdown()
        self.proxy.server_close()
        self.proxy_thread.join(timeout=2)
        self.source.shutdown()
        self.source.server_close()
        self.source_thread.join(timeout=2)

    def proxy_url(self):
        return "http://127.0.0.1:%d/api/frame.jpg" % self.proxy.server_port

    def test_proxy_returns_complete_jpeg(self):
        with urlopen(self.proxy_url(), timeout=2) as response:
            body = response.read()

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers.get_content_type(), "image/jpeg")
        self.assertEqual(body, JPEG)
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_proxy_rejects_incomplete_jpeg(self):
        FrameSourceHandler.payload = b"not-a-jpeg"

        with self.assertRaises(HTTPError) as caught:
            urlopen(self.proxy_url(), timeout=2)

        self.assertEqual(caught.exception.code, 502)


if __name__ == "__main__":
    unittest.main()
