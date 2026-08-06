#!/usr/bin/env python3
"""
Force Hexagon NPU (QNN HTP) work — prove the silicon is not decorative.

Why zero NPU processes before
-----------------------------
1. onnxruntime-qnn 2.x is a *plugin* EP — must register_execution_provider_library
2. HTP/NPU only runs *quantized* (QDQ) graphs — FP32 DeBERTa stayed on CPU
3. Conflicting onnxruntime / directml / qnn packages hid the EP

This script:
  - registers QNN plugin
  - builds a tiny MatMul ONNX → QDQ quantizes it
  - runs on QNNExecutionProvider + QnnHtp.dll (Hexagon HTP)
  - writes prime/state/npu_qnn_smoke.json

Usage:
  python npu_qnn_smoke.py
  python npu_qnn_smoke.py --bench   # longer HTP loop (extra latency sample)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT.parent / "state" / "npu"
STATE.mkdir(parents=True, exist_ok=True)
OUT = ROOT.parent / "state" / "npu_qnn_smoke.json"


def register_qnn() -> dict:
    import onnxruntime as ort
    import onnxruntime_qnn as qnn

    lib = qnn.get_library_path()
    name = "QNNExecutionProvider"
    try:
        ort.register_execution_provider_library(name, lib)
    except Exception as e:
        # API does not guarantee "already" text — verify devices after failure
        try:
            devs_try = list(ort.get_ep_devices())
            if not any(d.ep_name == name for d in devs_try):
                return {
                    "ok": False,
                    "error": f"register failed: {e}",
                    "lib": lib,
                    "ort": ort.__version__,
                    "qnn": getattr(qnn, "__version__", None),
                }
        except Exception as e2:
            return {
                "ok": False,
                "error": f"register failed: {e}; devices: {e2}",
                "lib": lib,
                "ort": ort.__version__,
                "qnn": getattr(qnn, "__version__", None),
            }
    devs = []
    for d in ort.get_ep_devices():
        devs.append(
            {
                "ep_name": d.ep_name,
                "repr": str(d),
            }
        )
    htp = qnn.get_qnn_htp_path()
    return {
        "ok": True,
        "ort": ort.__version__,
        "qnn": getattr(qnn, "__version__", None),
        "lib": lib,
        "htp_dll": htp,
        "htp_exists": Path(htp).is_file() if htp else False,
        "devices": devs,
        "n_qnn_devices": sum(1 for d in devs if d["ep_name"] == name),
        "providers_list": ort.get_available_providers(),
    }


def build_and_quantize() -> dict:
    """Tiny Gemm/MatMul float32 → QDQ for HTP."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantType, quantize_static
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    fp_path = STATE / "tiny_gemm.onnx"
    qdq_path = STATE / "tiny_gemm.qdq.onnx"

    # y = x @ W + b   x:[1,16] W:[16,8] b:[8]
    W = np.random.randn(16, 8).astype(np.float32) * 0.1
    b = np.random.randn(8).astype(np.float32) * 0.01
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
    w_init = numpy_helper.from_array(W, "W")
    b_init = numpy_helper.from_array(b, "B")
    node = helper.make_node("Gemm", ["X", "W", "B"], ["Y"], alpha=1.0, beta=1.0)
    graph = helper.make_graph([node], "tiny_gemm", [X], [Y], [w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(fp_path))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.data = [
                {"X": np.random.randn(1, 16).astype(np.float32)} for _ in range(16)
            ]
            self.i = 0

        def get_next(self):
            if self.i >= len(self.data):
                return None
            d = self.data[self.i]
            self.i += 1
            return d

        def rewind(self):
            self.i = 0

    # Prefer QNN-aware quant if available
    try:
        from onnxruntime.quantization.execution_providers.qnn import (
            get_qnn_qdq_config,
            qnn_preprocess_model,
        )

        pre = STATE / "tiny_gemm.pre.onnx"
        changed = qnn_preprocess_model(str(fp_path), str(pre))
        src = str(pre) if changed else str(fp_path)
        reader = Reader()
        cfg = get_qnn_qdq_config(
            src,
            reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )
        reader.rewind()
        quantize_static(src, str(qdq_path), reader, quant_format=None, extra_options=None)
        # get_qnn_qdq_config returns config object for quantize()
        from onnxruntime.quantization import quantize

        reader.rewind()
        quantize(src, str(qdq_path), cfg)
        method = "qnn_qdq_config"
    except Exception as e1:
        # fallback classic static quant
        try:
            from onnxruntime.quantization import QuantFormat

            reader = Reader()
            quantize_static(
                str(fp_path),
                str(qdq_path),
                reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
            )
            method = f"quantize_static_fallback after {e1}"
        except Exception as e2:
            return {
                "ok": False,
                "error": f"quantize failed: {e1} | {e2}",
                "fp": str(fp_path),
            }

    return {
        "ok": True,
        "fp_model": str(fp_path),
        "qdq_model": str(qdq_path),
        "qdq_mb": round(qdq_path.stat().st_size / 1e6, 3),
        "method": method,
    }


def run_on_htp(qdq_path: str, *, disable_cpu_fallback: bool = False) -> dict:
    import numpy as np
    import onnxruntime as ort
    import onnxruntime_qnn as qnn

    # ensure registered
    reg = register_qnn()
    if not reg.get("ok"):
        return reg

    name = "QNNExecutionProvider"
    htp = qnn.get_qnn_htp_path()
    selected = [
        d for d in ort.get_ep_devices() if d.ep_name == name
    ]
    if not selected:
        return {"ok": False, "error": "no QNN EP devices after register", "reg": reg}

    so = ort.SessionOptions()
    if disable_cpu_fallback:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    # backend_path OR backend_type — not both (QNN EP hard error)
    ep_options = {
        "backend_path": htp,
        "htp_performance_mode": "burst",
        "enable_htp_fp16_precision": "1",
    }

    t0 = time.time()
    err = None
    sess = None
    api = ""
    # Plugin API (2.x) — only QNN devices
    try:
        so.add_provider_for_devices(selected, ep_options)
        sess = ort.InferenceSession(qdq_path, sess_options=so)
        api = "plugin_add_provider_for_devices"
    except Exception as e:
        err = str(e)
        # Classic providers= API (no SessionOptions pre-config)
        try:
            so2 = ort.SessionOptions()
            if disable_cpu_fallback:
                so2.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            sess = ort.InferenceSession(
                qdq_path,
                sess_options=so2,
                providers=["QNNExecutionProvider"],
                provider_options=[ep_options],
            )
            api = "classic_providers_qnn_only"
            err = None
        except Exception as e2:
            # last resort: allow CPU fallback to report error path
            try:
                so3 = ort.SessionOptions()
                sess = ort.InferenceSession(
                    qdq_path,
                    sess_options=so3,
                    providers=["QNNExecutionProvider", "CPUExecutionProvider"],
                    provider_options=[ep_options, {}],
                )
                api = "classic_providers_qnn_cpu_fallback"
                err = f"strict QNN failed: {e2}"
            except Exception as e3:
                return {
                    "ok": False,
                    "error": f"plugin: {err} | classic: {e2} | fallback: {e3}",
                    "reg": reg,
                    "ep_options": ep_options,
                }

    assert sess is not None
    providers_used = sess.get_providers()
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in (inp.shape or [1, 16])]
    x = np.random.randn(*shape).astype(np.float32)
    # warm
    for _ in range(3):
        sess.run(None, {inp.name: x})
    t1 = time.time()
    n = 50
    for _ in range(n):
        out = sess.run(None, {inp.name: x})
    elapsed = time.time() - t1
    y = out[0]
    on_qnn = any("QNN" in p for p in providers_used)
    return {
        "ok": True,
        "api": api,
        "providers_used": providers_used,
        "on_qnn_ep": on_qnn,
        "htp_dll": htp,
        "input_shape": shape,
        "output_shape": list(y.shape),
        "runs": n,
        "total_s": round(elapsed, 4),
        "ms_per_run": round(elapsed * 1000 / n, 3),
        "session_create_s": round(t0 and (time.time() - t0 - elapsed), 3),
        "reg_devices": reg.get("n_qnn_devices"),
        "note": (
            "QNN EP in providers list = graph scheduled on QNN stack (HTP=NPU). "
            "Task Manager may still show system NPU as 'Shared' not a named process."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Hexagon QNN HTP smoke")
    ap.add_argument(
        "--bench",
        action="store_true",
        help="extra latency sample (250 runs) after smoke",
    )
    args = ap.parse_args()
    t0 = time.time()
    report: dict = {"ok": False, "seconds": 0, "bench": bool(args.bench)}
    print("1) register QNN plugin…", flush=True)
    reg = register_qnn()
    report["register"] = reg
    print(json.dumps(reg, indent=2), flush=True)
    if not reg.get("ok") or reg.get("n_qnn_devices", 0) < 1:
        report["error"] = "QNN devices not available"
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("FAIL: no QNN devices", flush=True)
        return 1

    print("2) build+quantize tiny gemm…", flush=True)
    q = build_and_quantize()
    report["quantize"] = q
    print(json.dumps(q, indent=2), flush=True)
    if not q.get("ok"):
        report["error"] = q.get("error")
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    print("3) run on HTP (Hexagon NPU)…", flush=True)
    run = run_on_htp(q["qdq_model"], disable_cpu_fallback=False)
    report["run"] = run
    print(json.dumps(run, indent=2), flush=True)
    if args.bench and run.get("ok"):
        # Longer sample for ms/run stability (same session path)
        print("3b) --bench: extra 250 runs…", flush=True)
        try:
            import numpy as np
            import onnxruntime as ort
            from npu_qnn import register

            register()
            so = ort.SessionOptions()
            sess = ort.InferenceSession(
                q["qdq_model"],
                sess_options=so,
                providers=["QNNExecutionProvider", "CPUExecutionProvider"],
            )
            inp = sess.get_inputs()[0]
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in (inp.shape or [1, 16])]
            x = np.random.randn(*shape).astype(np.float32)
            for _ in range(5):
                sess.run(None, {inp.name: x})
            n = 250
            t1 = time.time()
            for _ in range(n):
                sess.run(None, {inp.name: x})
            elapsed = time.time() - t1
            report["bench_result"] = {
                "runs": n,
                "total_s": round(elapsed, 4),
                "ms_per_run": round(elapsed * 1000 / n, 3),
                "providers": sess.get_providers(),
            }
            print(json.dumps(report["bench_result"], indent=2), flush=True)
        except Exception as e:
            report["bench_result"] = {"ok": False, "error": str(e)}

    report["ok"] = bool(run.get("ok") and run.get("on_qnn_ep"))
    report["seconds"] = round(time.time() - t0, 1)
    report["verdict"] = (
        "NPU_PATH_LIVE"
        if report["ok"]
        else "NPU_PATH_FAIL"
    )
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("verdict", report["verdict"], "wrote", OUT, flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
