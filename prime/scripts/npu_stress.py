#!/usr/bin/env python3
"""
Sustained Hexagon NPU (QNN HTP) + optional Adreno GPU (DML) load.

Task Manager only moves if work lasts *seconds* at meaningful size.
A 50-iter microbench is invisible. This keeps HTP busy for --seconds.

Usage (watch Performance → NPU / GPU while this runs):
  python npu_stress.py --seconds 45
  python npu_stress.py --seconds 60 --also-gpu

Hard rules:
  - QNN plugin must be registered
  - HTP needs QDQ model
  - session.disable_cpu_ep_fallback=1 so we fail loud if NPU not used
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STATE = ROOT.parent / "state" / "npu"
STATE.mkdir(parents=True, exist_ok=True)
REPORT = ROOT.parent / "state" / "npu_stress_report.json"

# Large enough to show utilization (stacked gemms)
DIM = 512
DEPTH = 8  # chain of MatMuls


def build_heavy_qdq() -> Path:
    """Build FP32 chain of MatMuls → QDQ quantize for HTP."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    fp = STATE / "heavy_chain.onnx"
    qdq = STATE / "heavy_chain.qdq.onnx"
    # Quantize into temp then atomic replace — avoid partial QDQ opened by stress_htp
    qdq_tmp = STATE / "heavy_chain.qdq.onnx.tmp"

    nodes = []
    inits = []
    # X @ W0 @ W1 @ ... @ W{DEPTH-1}
    prev = "X"
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
    for i in range(DEPTH):
        w = np.random.randn(DIM, DIM).astype(np.float32) * (0.02)
        # slightly orthogonal-ish scale for stability
        w = w / (np.linalg.norm(w, axis=0, keepdims=True) + 1e-6)
        name_w = f"W{i}"
        inits.append(numpy_helper.from_array(w, name_w))
        out = f"Y{i}" if i < DEPTH - 1 else "Y"
        nodes.append(
            helper.make_node("MatMul", [prev, name_w], [out], name=f"mm{i}")
        )
        prev = out
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
    graph = helper.make_graph(nodes, "heavy_chain", [X], [Y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(fp))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.data = [
                {"X": np.random.randn(1, DIM).astype(np.float32)} for _ in range(32)
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

    reader = Reader()
    if qdq_tmp.is_file():
        qdq_tmp.unlink(missing_ok=True)
    try:
        from onnxruntime.quantization.execution_providers.qnn import (
            get_qnn_qdq_config,
            qnn_preprocess_model,
        )
        from onnxruntime.quantization import quantize

        pre = STATE / "heavy_chain.pre.onnx"
        changed = qnn_preprocess_model(str(fp), str(pre))
        src = str(pre) if changed else str(fp)
        reader.rewind()
        cfg = get_qnn_qdq_config(
            src, reader, activation_type=QuantType.QUInt8, weight_type=QuantType.QUInt8
        )
        reader.rewind()
        quantize(src, str(qdq_tmp), cfg)
    except Exception as e_qnn:
        reader.rewind()
        try:
            quantize_static(
                str(fp),
                str(qdq_tmp),
                reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
            )
        except Exception as e_static:
            raise RuntimeError(
                f"QDQ quantize failed: qnn_path={e_qnn}; static={e_static}"
            ) from e_static
        # record that we fell back so operators see why
        print(f"warn: qnn_qdq_config failed ({e_qnn}); used quantize_static", flush=True)
    if not qdq_tmp.is_file() or qdq_tmp.stat().st_size < 100:
        raise RuntimeError(f"QDQ quantize failed or empty: {qdq_tmp}")
    qdq_tmp.replace(qdq)
    return qdq


def open_htp_session(qdq: Path, *, profile: bool = True):
    import onnxruntime as ort
    import onnxruntime_qnn as qnn
    from npu_qnn import register

    reg = register()
    if not reg.get("ok"):
        raise RuntimeError(f"QNN register failed: {reg}")

    htp = qnn.get_qnn_htp_path()
    selected = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    if not selected:
        raise RuntimeError("No QNN devices")

    prof_csv = STATE / "htp_profile.csv"
    ep_options = {
        "backend_path": htp,
        "htp_performance_mode": "burst",
        "enable_htp_fp16_precision": "1",
    }
    if profile:
        ep_options["profiling_level"] = "detailed"
        ep_options["profiling_file_path"] = str(prof_csv)

    so = ort.SessionOptions()
    # FAIL if we silently fall to CPU — that was the invisible "fake NPU" path
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(selected, ep_options)
    sess = ort.InferenceSession(str(qdq), sess_options=so)
    prov = sess.get_providers()
    if "QNNExecutionProvider" not in prov:
        raise RuntimeError(f"Session not on QNN: {prov}")
    if prov == ["CPUExecutionProvider"]:
        raise RuntimeError("CPU only — NPU not engaged")
    return sess, prov, htp, prof_csv


def summarize_htp_profile(prof_csv: Path) -> dict:
    """Parse QNN HTP profile CSV — this is the real NPU proof (Task Manager has no NPU counters here).

    Stream the file line-by-line (profiles can be large); do not load whole CSV into RAM.
    """
    if not prof_csv.is_file():
        return {"ok": False, "error": "no profile csv"}
    hvx: list[str] = []
    accel: list[str] = []
    nodes: list[str] = []
    cycles: list[int] = []
    n_lines = 0
    with prof_csv.open(encoding="utf-8", errors="replace") as f:
        for ln in f:
            n_lines += 1
            if ("HVX" in ln or "HMX" in ln) and len(hvx) < 8:
                hvx.append(ln.rstrip())
            if "QNN accelerator (execute)" in ln and len(accel) < 6:
                accel.append(ln.rstrip())
            if ",NODE," in ln and "mm" in ln:
                if len(nodes) < 12:
                    nodes.append(ln.rstrip())
                parts = ln.split(",")
                try:
                    if len(parts) > 2 and parts[2] == "CYCLES":
                        cycles.append(int(parts[1]))
                except Exception:
                    pass
    return {
        "ok": True,
        "path": str(prof_csv),
        "bytes": prof_csv.stat().st_size,
        "n_lines_scanned": n_lines,
        "hvx_lines": hvx,
        "accel_execute_lines": accel,
        "matmul_node_lines": nodes,
        "matmul_cycles_sum": sum(cycles) if cycles else None,
        "proof": (
            "HVX/HMX + accelerator execute cycles in profile = Hexagon NPU ran the graph. "
            "This Windows build exposes NO Performance→NPU counters (typeperf empty); "
            "Task Manager cannot graph third-party QNN HTP."
        ),
    }


def _qdq_looks_complete(path: Path, *, min_bytes: int = 10_000) -> bool:
    """Reject partial/truncated QDQ left by interrupted quantize runs."""
    if not path.is_file():
        return False
    try:
        sz = path.stat().st_size
        if sz < min_bytes:
            return False
        # ONNX protobuf files start with a small field tag; empty/truncated often 0 bytes
        head = path.read_bytes()[:4]
        return len(head) == 4
    except OSError:
        return False


def stress_htp(seconds: float) -> dict:
    qdq = STATE / "heavy_chain.qdq.onnx"
    # Never reopen a partial QDQ from a crashed quantize (misreported as HTP reject)
    if not _qdq_looks_complete(qdq):
        if qdq.is_file():
            print(f"rebuilding incomplete QDQ ({qdq.stat().st_size} bytes)…", flush=True)
            qdq.unlink(missing_ok=True)
        print("building heavy QDQ chain…", flush=True)
        qdq = build_heavy_qdq()
        print("wrote", qdq, "mb", round(qdq.stat().st_size / 1e6, 2), flush=True)

    print("opening HTP session (no CPU fallback + HTP profile)…", flush=True)
    sess, prov, htp, prof_csv = open_htp_session(qdq, profile=True)
    print("providers", prov, flush=True)
    print("htp", htp, flush=True)
    print(
        f"\n>>> Task Manager on this OS often has NO NPU counter (we checked typeperf).\n"
        f">>> Real proof: QNN HTP profile CSV after run.\n"
        f">>> Hammering Hexagon HTP for {seconds:.0f}s at burst …\n",
        flush=True,
    )

    x = np.random.randn(1, DIM).astype(np.float32)
    # warm
    for _ in range(10):
        sess.run(None, {"X": x})

    n = 0
    t0 = time.time()
    last_print = t0
    samples = []
    while time.time() - t0 < seconds:
        # batch of runs between samples
        for _ in range(20):
            sess.run(None, {"X": np.random.randn(1, DIM).astype(np.float32)})
            n += 1
        now = time.time()
        if now - last_print >= 1.0:
            rate = n / (now - t0)
            samples.append({"t": round(now - t0, 1), "runs": n, "runs_per_s": round(rate, 1)})
            print(f"  t={now-t0:5.1f}s  runs={n}  rate={rate:.1f}/s  providers={prov}", flush=True)
            last_print = now

    elapsed = time.time() - t0
    # one more run to flush profile
    sess.run(None, {"X": x})
    del sess
    time.sleep(0.3)
    prof = summarize_htp_profile(prof_csv)
    print("\n=== HTP PROFILE PROOF ===", flush=True)
    print(json.dumps(prof, indent=2)[:2000], flush=True)
    return {
        "ok": True,
        "engine": "QNN_HTP",
        "providers": prov,
        "htp_dll": htp,
        "dim": DIM,
        "depth": DEPTH,
        "seconds": round(elapsed, 2),
        "runs": n,
        "runs_per_s": round(n / elapsed, 1),
        "samples": samples,
        "qdq": str(qdq),
        "qdq_mb": round(qdq.stat().st_size / 1e6, 2),
        "cpu_fallback_disabled": True,
        "htp_profile": prof,
    }


def stress_gpu_dml(seconds: float, stop_flag: list) -> dict:
    """Sustained Adreno load via DirectML if package present."""
    try:
        import onnxruntime as ort
    except Exception as e:
        return {"ok": False, "error": f"ort: {e}"}

    # DML only if this ORT build has it
    if "DmlExecutionProvider" not in ort.get_available_providers():
        # try loading directml package providers
        try:
            import onnxruntime.capi._pybind_state as C  # noqa: F401
        except Exception:
            pass
        if "DmlExecutionProvider" not in ort.get_available_providers():
            return {
                "ok": False,
                "error": "DmlExecutionProvider not in this ORT build",
                "providers": ort.get_available_providers(),
                "hint": "pip install onnxruntime-directml in a *separate* venv for GPU-only stress; base ORT kept for QNN plugin",
            }

    # simple float model for DML
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    fp = STATE / "gpu_chain.onnx"
    if not fp.is_file():
        nodes, inits = [], []
        prev = "X"
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, DIM])
        for i in range(DEPTH):
            w = np.random.randn(DIM, DIM).astype(np.float32) * 0.02
            inits.append(numpy_helper.from_array(w, f"W{i}"))
            out = f"Y{i}" if i < DEPTH - 1 else "Y"
            nodes.append(helper.make_node("MatMul", [prev, f"W{i}"], [out]))
            prev = out
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, DIM])
        g = helper.make_graph(nodes, "gpu_chain", [X], [Y], inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        onnx.save(m, str(fp))

    try:
        sess = ort.InferenceSession(
            str(fp),
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"device_id": "0"}, {}],
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

    prov = sess.get_providers()
    print("GPU session providers", prov, flush=True)
    n = 0
    t0 = time.time()
    while time.time() - t0 < seconds and not stop_flag[0]:
        for _ in range(20):
            sess.run(None, {"X": np.random.randn(1, DIM).astype(np.float32)})
            n += 1
    elapsed = max(time.time() - t0, 1e-6)
    return {
        "ok": "DmlExecutionProvider" in prov,
        "providers": prov,
        "runs": n,
        "runs_per_s": round(n / elapsed, 1),
        "seconds": round(elapsed, 2),
    }


def sample_gpu_counters() -> list:
    try:
        import subprocess

        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                # Use commas between calculated properties (semicolon terminates statement)
                r"Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue | "
                r"Select-Object -ExpandProperty CounterSamples | "
                r"Where-Object { $_.CookedValue -gt 1 } | "
                r"Select-Object -First 12 @{N='path';E={$_.Path}}, @{N='val';E={[math]::Round($_.CookedValue,1)}} | "
                r"ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.stdout.strip():
            return json.loads(r.stdout)
    except Exception as e:
        return [{"error": str(e)}]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--also-gpu", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    if a.rebuild:
        for p in STATE.glob("heavy_chain*"):
            p.unlink(missing_ok=True)

    stop = [False]
    gpu_thread = None
    gpu_result: dict = {"skipped": True}
    if a.also_gpu:
        def _g():
            nonlocal gpu_result
            gpu_result = stress_gpu_dml(a.seconds, stop)

        gpu_thread = threading.Thread(target=_g, daemon=True)
        gpu_thread.start()

    try:
        htp = stress_htp(a.seconds)
    except Exception as e:
        stop[0] = True
        report = {
            "ok": False,
            "error": str(e),
            "hint": (
                "If CPU fallback was forced off and this failed, HTP rejected the graph. "
                "If it succeeded only with CPU, Task Manager will never show NPU."
            ),
        }
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("FAIL", e, flush=True)
        return 1
    finally:
        stop[0] = True
        if gpu_thread:
            gpu_thread.join(timeout=5)

    gpu_snap = sample_gpu_counters()
    report = {
        "ok": bool(htp.get("ok")),
        "htp": htp,
        "gpu": gpu_result if a.also_gpu else {"skipped": True},
        "gpu_counter_sample": gpu_snap,
        "task_manager": {
            "look_at": "Performance → NPU 0 (Hexagon) and GPU 0 (Adreno)",
            "why_invisible_before": (
                "Micro-benchmarks finish in ms; HTP utilization averages to ~0%. "
                "This run holds burst HTP for tens of seconds."
            ),
            "no_process_named_npu": True,
        },
        "verdict": "NPU_STRESS_OK" if htp.get("ok") else "NPU_STRESS_FAIL",
    }
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n=== RESULT ===", flush=True)
    print(json.dumps({k: report[k] for k in ("ok", "verdict")}, indent=2))
    print("htp runs/s", htp.get("runs_per_s"), "providers", htp.get("providers"))
    print("wrote", REPORT, flush=True)
    print(
        "\nIf Performance→NPU still flat: Task Manager has no NPU counters here. "
        "Hard proof = htp_profile HVX/HMX + accelerator cycles "
        "(not the session providers list alone).",
        flush=True,
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
