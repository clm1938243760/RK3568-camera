#!/usr/bin/env python3
"""Create a fixed-shape PP-OCR detector candidate from the official ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def set_shape(value_info: object, shape: list[int]) -> None:
    dimensions = value_info.type.tensor_type.shape.dim
    if len(dimensions) != len(shape):
        raise ValueError("unexpected PP-OCR tensor rank")
    for dimension, value in zip(dimensions, shape):
        dimension.ClearField("dim_param")
        dimension.dim_value = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=480)
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("candidate height and width must be positive multiples of 32")

    model = onnx.load(str(args.source))
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("unexpected PP-OCR detector input/output count")
    set_shape(model.graph.input[0], [1, 3, args.height, args.width])
    set_shape(model.graph.output[0], [1, 1, args.height, args.width])
    # Paddle2ONNX records every intermediate tensor with the original fixed
    # 480x480 shape. Let ONNX/RKNN infer those shapes from the new input.
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print("PP-OCR detector candidate: 1x3x%dx%d" % (args.height, args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
