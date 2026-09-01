#!/usr/bin/env python3
"""Convert the locked DocAligner ONNX model for RK3568 without quantization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="rk3568")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument(
        "--float-input",
        action="store_true",
        help="keep floating point input instead of embedding BGR uint8 / 255",
    )
    return parser.parse_args()


def check_onnx_contract(path: Path) -> Dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    inputs = {
        item.name: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in model.graph.input
    }
    outputs = {
        item.name: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in model.graph.output
    }
    expected_inputs = {"img": [1, 3, 256, 256]}
    expected_outputs = {"points": [1, 8], "has_obj": [1, 1]}
    if inputs != expected_inputs or outputs != expected_outputs:
        raise RuntimeError(
            "unexpected DocAligner contract: inputs=%r outputs=%r" % (inputs, outputs)
        )
    return {"inputs": inputs, "outputs": outputs}


def main() -> int:
    args = parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if args.optimization_level not in (0, 1, 2, 3):
        raise ValueError("optimization level must be between 0 and 3")

    contract = check_onnx_contract(args.onnx)
    from rknn.api import RKNN

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    try:
        config = {
            "target_platform": args.target,
            "optimization_level": args.optimization_level,
        }
        if not args.float_input:
            config["mean_values"] = [[0.0, 0.0, 0.0]]
            config["std_values"] = [[255.0, 255.0, 255.0]]
        ret = rknn.config(**config)
        if ret != 0:
            raise RuntimeError("rknn.config failed: %s" % ret)
        ret = rknn.load_onnx(model=str(args.onnx))
        if ret != 0:
            raise RuntimeError("rknn.load_onnx failed: %s" % ret)
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError("rknn.build failed: %s" % ret)
        ret = rknn.export_rknn(str(args.output))
        if ret != 0:
            raise RuntimeError("rknn.export_rknn failed: %s" % ret)
    finally:
        rknn.release()

    result = {
        "schema_version": 1,
        "target_platform": args.target,
        "quantized": False,
        "input_contract": "float32_bgr_0_to_1" if args.float_input else "uint8_bgr",
        "optimization_level": args.optimization_level,
        "source": {
            "path": str(args.onnx),
            "sha256": sha256_file(args.onnx),
            "contract": contract,
        },
        "output": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
