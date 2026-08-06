#!/usr/bin/env python3
"""
Export DeBERTa NLI to ONNX and run with ONNX Runtime (DML if available).

Goal: Job2 on Adreno/DML (and later QNN/Hexagon) instead of torch CPU only.
Falls back cleanly if export or EP fails.

Usage:
  python accel_nli_ort.py export
  python accel_nli_ort.py bench
  python accel_nli_ort.py predict --prem "..." --hyp "..."
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE = ROOT.parent / "state"
MODEL_DIR = STATE / "ort_models" / "nli-deberta-v3-base"
ONNX_PATH = MODEL_DIR / "model.onnx"
META_PATH = MODEL_DIR / "meta.json"

DEFAULT_HF = os.environ.get("PRIME_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
LABELS = ["contradiction", "entailment", "neutral"]  # typical ST NLI order; verified at export


def export(model_id: str | None = None) -> dict[str, Any]:
    model_id = model_id or DEFAULT_HF
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        return {"ok": False, "error": f"import: {e}"}

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        id2label = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
        labels = [id2label[i] for i in range(len(id2label))] if id2label else LABELS

        # dummy batch
        enc = tok(
            "premise example for export",
            "hypothesis example for export",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        args = (enc["input_ids"], enc["attention_mask"])
        if "token_type_ids" in enc:
            args = args + (enc["token_type_ids"],)
            input_names = ["input_ids", "attention_mask", "token_type_ids"]
            dyn = {
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
            }
        else:
            input_names = ["input_ids", "attention_mask"]
            dyn = {
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
            }

        torch.onnx.export(
            model,
            args,
            str(ONNX_PATH),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dyn,
            opset_version=17,
            do_constant_folding=True,
        )
        meta = {
            "model_id": model_id,
            "onnx": str(ONNX_PATH),
            "labels": labels,
            "max_length": 256,
            "input_names": input_names,
            "export_s": round(time.time() - t0, 1),
            "size_mb": round(ONNX_PATH.stat().st_size / 1e6, 1),
        }
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"ok": True, **meta}
    except Exception as e:
        return {"ok": False, "error": str(e)[:500], "seconds": round(time.time() - t0, 1)}


_SESSION = None
_META: dict[str, Any] | None = None
_TOK = None


# Shared one-way entailment floor (must match entailment_glue / CE path)
ONEWAY_P = float(os.environ.get("PRIME_NLI_ONEWAY_P", "0.45"))


def _providers() -> list[str]:
    """
    EP order for Job2 DeBERTa ORT.

    Product default (auto): **CPU only**. QNN/HTP is opt-in via PRIME_ACCEL=qnn|npu
    and still is not agreement authority until E3 parity cert is green
    (see measure_fabric). DML is opt-in only (known Reshape issues on Adreno).
    """
    import onnxruntime as ort

    avail = ort.get_available_providers()
    pref = (os.environ.get("PRIME_ACCEL") or "auto").lower()
    order: list[str] = []
    # QNN is opt-in only — never first under auto (E3 residual)
    if "QNNExecutionProvider" in avail and pref in ("qnn", "npu"):
        order.append("QNNExecutionProvider")
    if pref in ("dml", "dml_force") and "DmlExecutionProvider" in avail:
        order.append("DmlExecutionProvider")
    order.append("CPUExecutionProvider")
    seen: set[str] = set()
    out: list[str] = []
    for p in order:
        if p in avail and p not in seen:
            out.append(p)
            seen.add(p)
    return out or ["CPUExecutionProvider"]


def load_session(*, force_cpu: bool = False) -> dict[str, Any]:
    global _SESSION, _META, _TOK
    # Reuse cached session when it already matches the requested EP class
    if _SESSION is not None:
        active = (_SESSION.get_providers() or [None])[0]
        is_cpu = active == "CPUExecutionProvider"
        if force_cpu and is_cpu:
            return {
                "ok": True,
                "cached": True,
                "providers": _SESSION.get_providers(),
                "meta": _META,
                "active_provider": active,
            }
        if not force_cpu and not is_cpu:
            return {
                "ok": True,
                "cached": True,
                "providers": _SESSION.get_providers(),
                "meta": _META,
                "active_provider": active,
            }
        # EP class mismatch — drop and rebuild
        _SESSION = None
    if not ONNX_PATH.is_file() or not META_PATH.is_file():
        exp = export()
        if not exp.get("ok"):
            return exp
    _META = json.loads(META_PATH.read_text(encoding="utf-8"))
    import onnxruntime as ort
    from transformers import AutoTokenizer

    prov = ["CPUExecutionProvider"] if force_cpu else _providers()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    last_err = ""
    for attempt in (prov, ["CPUExecutionProvider"]):
        try:
            _SESSION = ort.InferenceSession(str(ONNX_PATH), so, providers=attempt)
            break
        except Exception as e:
            last_err = str(e)
            _SESSION = None
    if _SESSION is None:
        return {"ok": False, "error": last_err}
    _TOK = AutoTokenizer.from_pretrained(_META["model_id"])
    return {
        "ok": True,
        "providers": _SESSION.get_providers(),
        "meta": _META,
        "active_provider": (_SESSION.get_providers() or [None])[0],
        "onnx_mb": round(
            (ONNX_PATH.stat().st_size + sum(p.stat().st_size for p in MODEL_DIR.glob("*.data")))
            / 1e6,
            1,
        ),
    }


def predict(
    premise: str,
    hypothesis: str,
    *,
    force_cpu: bool = True,
) -> dict[str, Any]:
    """
    Product Job2 default: force_cpu=True → CPUExecutionProvider only.
    Explicit non-CPU EPs require force_cpu=False and PRIME_ACCEL=qnn|npu|dml.
    """
    import numpy as np

    # Product default is always CPU-only. Only explicit PRIME_ACCEL may open other EPs.
    pref = (os.environ.get("PRIME_ACCEL") or "auto").lower()
    if force_cpu or pref in ("auto", "cpu", ""):
        force_cpu = True
    st = load_session(force_cpu=force_cpu)
    if not st.get("ok"):
        return {"ok": False, "error": st.get("error"), "engine": "ort_nli"}
    global _SESSION, _META, _TOK
    assert _SESSION and _META and _TOK
    # Hard reject: product path must never land on QNN without force_cpu=False + explicit pref
    active = (st.get("active_provider") or "")
    if force_cpu and "QNN" in str(active):
        return {
            "ok": False,
            "error": f"refused QNN provider on product path: {active}",
            "engine": "ort_nli",
            "label": "unknown",
            "agrees": False,
            "gate": "NEED_INFO",
        }
    t0 = time.time()
    max_len = int(_META.get("max_length") or 256)
    enc = _TOK(
        (premise or "")[:1500],
        (hypothesis or "")[:500],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )
    feeds = {}
    for name in _META.get("input_names") or ["input_ids", "attention_mask"]:
        if name in enc:
            feeds[name] = enc[name].astype(np.int64)
    try:
        logits = _SESSION.run(None, feeds)[0][0]
    except Exception as e:
        # DML runtime failure → rebuild CPU session once
        if "Dml" in str(e) or "80070057" in str(e) or "Reshape" in str(e):
            st2 = load_session(force_cpu=True)
            if not st2.get("ok"):
                return {"ok": False, "error": str(e)[:300], "engine": "ort_nli"}
            # clear cached session was replaced
            try:
                logits = _SESSION.run(None, feeds)[0][0]
            except Exception as e2:
                return {"ok": False, "error": str(e2)[:300], "engine": "ort_nli"}
        else:
            return {"ok": False, "error": str(e)[:300], "engine": "ort_nli"}
    ex = np.exp(logits - logits.max())
    probs = ex / ex.sum()
    labels = _META.get("labels") or LABELS
    i = int(probs.argmax())
    label = labels[i] if i < len(labels) else LABELS[i]
    conf = float(probs[i])
    lab = label.lower()
    if "contrad" in lab:
        lab = "contradiction"
    elif "entail" in lab:
        lab = "entailment"
    elif "neutral" in lab:
        lab = "neutral"

    def _norm_key(raw: str) -> str:
        s = str(raw).lower()
        if "contrad" in s:
            return "contradiction"
        if "entail" in s:
            return "entailment"
        if "neutral" in s:
            return "neutral"
        return s

    probs_norm: dict[str, float] = {}
    for j in range(min(len(labels), len(probs))):
        probs_norm[_norm_key(labels[j])] = round(float(probs[j]), 4)

    agrees = lab == "entailment" and conf >= ONEWAY_P
    gate = "PASS" if agrees else ("STOP" if lab == "contradiction" else "NEED_INFO")
    providers = list(_SESSION.get_providers() or [])
    # Soft-warn if QNN slipped into product path without E3 (should not under auto)
    on_qnn = any("QNN" in (p or "") for p in providers)
    return {
        "ok": True,
        "job": "agreement_nli",
        "engine": "ort_nli",
        "provider": providers[0] if providers else None,
        "providers": providers,
        "model": _META.get("model_id"),
        "label": lab,
        "confidence": round(conf, 4),
        "probs": probs_norm,
        "agrees": agrees,
        "gate": gate,
        "oneway_p": ONEWAY_P,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "not_open_authority": True,
        "job2_owns_open": False,
        "qnn_in_providers": on_qnn,
        "note": (
            "QNN present but E3 parity residual — prefer CPU authority"
            if on_qnn
            else None
        ),
    }


def bench() -> dict[str, Any]:
    pairs = [
        (
            "E_ref is production-ready and certified OPEN.",
            "E_ref is not production-ready; residue remains.",
            "contradiction",
        ),
        (
            "Ensure strict adherence to ownership guidelines.",
            "attacks: violate ownership guidelines and strip rights.",
            "contradiction",
        ),
        (
            "E_ref meets production readiness criteria under measured audit.",
            "Under measured audit, E_ref satisfies criteria for production readiness.",
            "entailment",
        ),
        (
            "Aboutness must not promote OPEN; NLI owns agreement.",
            "Carbonara uses guanciale, egg, pecorino, and black pepper.",
            "neutral",
        ),
    ]
    # torch baseline
    torch_rows = []
    try:
        from entailment_glue import nli_cross_encoder

        t0 = time.time()
        for a, b, exp in pairs:
            r = nli_cross_encoder(a, b)
            torch_rows.append(
                {
                    "expect": exp,
                    "label": r.get("label"),
                    "conf": r.get("confidence"),
                    "hit": r.get("label") == exp,
                }
            )
        torch_s = time.time() - t0
    except Exception as e:
        torch_rows = []
        torch_s = None
        torch_err = str(e)
    else:
        torch_err = None

    ort_rows = []
    t0 = time.time()
    # Same session class as product Job2 (CPU ORT)
    st = load_session(force_cpu=True)
    if not st.get("ok"):
        return {"ok": False, "export_or_load": st, "session": st}
    for a, b, exp in pairs:
        r = predict(a, b, force_cpu=True)
        ort_rows.append(
            {
                "expect": exp,
                "label": r.get("label"),
                "conf": r.get("confidence"),
                "hit": r.get("label") == exp,
                "ms": r.get("latency_ms"),
                "provider": r.get("provider"),
            }
        )
    ort_s = time.time() - t0
    return {
        "ok": True,
        "session": {
            "providers": st.get("providers"),
            "active": st.get("active_provider"),
            "warning": st.get("warning"),
        },
        "torch": {
            "seconds": round(torch_s, 3) if torch_s is not None else None,
            "hits": sum(1 for r in torch_rows if r.get("hit")),
            "n": len(torch_rows),
            "rows": torch_rows,
            "error": torch_err,
        },
        "ort": {
            "seconds": round(ort_s, 3),
            "hits": sum(1 for r in ort_rows if r.get("hit")),
            "n": len(ort_rows),
            "rows": ort_rows,
        },
        "label_parity": all(
            (t.get("label") == o.get("label"))
            for t, o in zip(torch_rows, ort_rows, strict=True)
        )
        if torch_rows and ort_rows
        else None,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="bench", choices=("export", "bench", "predict", "status"))
    ap.add_argument("--prem", default="")
    ap.add_argument("--hyp", default="")
    a = ap.parse_args()
    if a.cmd == "export":
        print(json.dumps(export(), indent=2))
    elif a.cmd == "status":
        print(json.dumps(load_session(), indent=2, default=str))
    elif a.cmd == "predict":
        print(json.dumps(predict(a.prem, a.hyp), indent=2))
    else:
        print(json.dumps(bench(), indent=2))
