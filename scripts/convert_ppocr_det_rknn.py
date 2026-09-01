#!/usr/bin/env python3
"""Build a non-quantized RK3568 PP-OCR detector with RKNN-Toolkit2."""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("output_rknn", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rknn = RKNN(verbose=args.verbose)
    try:
        result = rknn.config(
            mean_values=[[123.675, 116.28, 103.53]],
            std_values=[[58.395, 57.12, 57.375]],
            target_platform="rk3568",
        )
        if result != 0:
            raise RuntimeError("rknn.config failed: %s" % result)
        result = rknn.load_onnx(model=str(args.onnx_model))
        if result != 0:
            raise RuntimeError("rknn.load_onnx failed: %s" % result)
        result = rknn.build(do_quantization=False)
        if result != 0:
            raise RuntimeError("rknn.build failed: %s" % result)
        args.output_rknn.parent.mkdir(parents=True, exist_ok=True)
        result = rknn.export_rknn(str(args.output_rknn))
        if result != 0:
            raise RuntimeError("rknn.export_rknn failed: %s" % result)
    finally:
        rknn.release()
    print("RK3568 PP-OCR detector exported: %s" % args.output_rknn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
