#!/usr/bin/env python3
"""
Qualcomm Hexagon NPU via ONNX Runtime QNN plugin EP (HTP backend).

Critical facts (why you saw zero NPU processes before)
------------------------------------------------------
1. onnxruntime-qnn 2.x is a *plugin* — must call register_execution_provider_library
   or QNN never appears in get_available_providers().
2. HTP (Hexagon NPU) only executes *quantized QDQ* graphs. FP32 DeBERTa/rerank
   stay on CPU forever without quantize.
3. NPU work shows as HTP subsystem activity, not a process named "npu".

Public API
----------
  register()           → devices + dll paths
  status()             → operator snapshot
  session_qdq(path)    → InferenceSession preferring QNN HTP
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_REGISTERED = False
_LAST: dict[str, Any] = {}

STATE = Path(__file__).resolve().parent.parent / "state"


def register() -> dict[str, Any]:
    global _REGISTERED, _LAST
    with _LOCK:
        try:
            import onnxruntime as ort
            import onnxruntime_qnn as qnn
        except Exception as e:
            _LAST = {"ok": False, "error": f"import: {e}"}
            return _LAST

        lib = qnn.get_library_path()
        htp = qnn.get_qnn_htp_path()
        name = "QNNExecutionProvider"
        if not _REGISTERED:
            try:
                ort.register_execution_provider_library(name, lib)
            except Exception as e:
                # API does not guarantee message text — check devices after failure
                devs_try = list(ort.get_ep_devices())
                if not any(d.ep_name == name for d in devs_try):
                    _LAST = {
                        "ok": False,
                        "error": str(e),
                        "lib": lib,
                        "ort": ort.__version__,
                    }
                    return _LAST
            _REGISTERED = True

        devs = [{"ep_name": d.ep_name, "repr": str(d)} for d in ort.get_ep_devices()]
        n_qnn = sum(1 for d in devs if d["ep_name"] == name)
        htp_ok = Path(htp).is_file() if htp else False
        _LAST = {
            "ok": n_qnn > 0,
            "registered": _REGISTERED,
            "ort": ort.__version__,
            "qnn": getattr(qnn, "__version__", None),
            # Logical identifiers only in status surfaces (no user-profile paths)
            "lib": "<onnxruntime_qnn>/onnxruntime_providers_qnn.dll" if lib else None,
            "htp_dll": "<onnxruntime_qnn>/QnnHtp.dll" if htp else None,
            "htp_exists": htp_ok,
            "htp_dll_resolved": htp,  # local-only; strip before public evidence
            "lib_resolved": lib,
            "n_qnn_devices": n_qnn,
            "devices": [
                {"ep_name": d["ep_name"]} for d in devs
            ],  # drop pointer reprs
            "providers_builtin": ort.get_available_providers(),
            "note": (
                "HTP=Hexagon NPU. Use QDQ quantized ONNX only. "
                "No process named npu — activity is HTP subsystem. "
                "qnn_ep_registered ≠ HTP cycle proof — use htp_profile."
            ),
        }
        return _LAST


def status() -> dict[str, Any]:
    if not _LAST:
        return register()
    return dict(_LAST)


def session_qdq(
    model_path: str | Path,
    *,
    burst: bool = True,
    allow_cpu_fallback: bool = True,
) -> dict[str, Any]:
    """Create ORT session for a QDQ model on QNN HTP."""
    import onnxruntime as ort
    import onnxruntime_qnn as qnn

    reg = register()
    if not reg.get("ok"):
        return {"ok": False, "error": reg.get("error") or "qnn register failed", "reg": reg}
    if not reg.get("htp_exists"):
        return {
            "ok": False,
            "error": "QnnHtp.dll missing — cannot claim Hexagon HTP backend",
            "reg": reg,
        }

    htp = qnn.get_qnn_htp_path()
    selected = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    if not selected:
        return {"ok": False, "error": "no QNN devices", "reg": reg}

    # Session needs real filesystem path; status surfaces use logical names only
    ep_options = {
        "backend_path": htp,
        "htp_performance_mode": "burst" if burst else "balanced",
        "enable_htp_fp16_precision": "1",
    }
    so = ort.SessionOptions()
    if not allow_cpu_fallback:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    try:
        so.add_provider_for_devices(selected, ep_options)
        sess = ort.InferenceSession(str(model_path), sess_options=so)
    except Exception as e:
        if not allow_cpu_fallback:
            return {"ok": False, "error": str(e), "reg": reg}
        try:
            sess = ort.InferenceSession(
                str(model_path),
                providers=["QNNExecutionProvider", "CPUExecutionProvider"],
                provider_options=[ep_options, {}],
            )
        except Exception as e2:
            return {"ok": False, "error": f"{e} | {e2}", "reg": reg}

    prov = sess.get_providers()
    # providers list = EP registered for session, NOT per-node HTP proof
    qnn_ep_registered = any("QNN" in p for p in prov)
    return {
        "ok": True,
        "session": sess,
        "providers": prov,
        "qnn_ep_registered": qnn_ep_registered,
        "on_qnn": qnn_ep_registered,  # legacy alias; HTP cycles = hard proof
        "htp_proof": "htp_profile_cycles_or_disable_cpu_fallback",
        "htp_dll": htp,
        "reg": reg,
    }


if __name__ == "__main__":
    print(json.dumps(register(), indent=2))
