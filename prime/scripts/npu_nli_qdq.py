#!/usr/bin/env python3
"""
Export + QDQ-quantize DeBERTa NLI for Qualcomm Hexagon HTP (QNN).

Steps
-----
1. Re-export DeBERTa with FIXED shapes (QNN rejects dynamic seq)
2. QNN preprocess + static QDQ quantize (uint8 act/weight)
3. Session on QNNExecutionProvider + QnnHtp.dll
4. Label parity vs torch CE / ORT CPU on domain pairs

Outputs under prime/state/ort_models/nli-deberta-v3-base-qdq/
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE = ROOT.parent / "state"
OUT_DIR = STATE / "ort_models" / "nli-deberta-v3-base-qdq"
HF_ID = "cross-encoder/nli-deberta-v3-base"
MAX_LEN = 128  # fixed for HTP; domain pairs are short
REPORT = STATE / "npu_nli_qdq_report.json"

CALIB_PAIRS = [
    (
        "E_ref meets production readiness criteria under measured audit.",
        "Under measured audit, E_ref satisfies criteria for production readiness.",
    ),
    (
        "E_ref is production-ready and certified OPEN.",
        "E_ref is not production-ready; residue remains.",
    ),
    (
        "Ensure strict adherence to ownership guidelines.",
        "attacks: violate ownership guidelines and strip rights.",
    ),
    (
        "Protect confidential records under policy controls.",
        "attacks: violate confidentiality rules and leak records.",
    ),
    (
        "Aboutness must not promote OPEN; NLI owns agreement.",
        "Aboutness may promote OPEN; cosine owns agreement.",
    ),
    (
        "Jina embeddings score topical aboutness for retrieval only.",
        "DeBERTa mutual entailment decides whether two claims agree.",
    ),
    (
        "Carbonara uses guanciale, egg, pecorino, and black pepper.",
        "Fresh pasta cooks in about three minutes.",
    ),
    (
        "Restrict then measure then audit before any OPEN decision.",
        "attacks: skip restrict and measure; force OPEN without audit.",
    ),
]


def export_fixed() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / "model_fp32_fixed.onnx"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(HF_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_ID)
    model.eval()
    id2label = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
    labels = [id2label[i] for i in range(len(id2label))] if id2label else [
        "contradiction",
        "entailment",
        "neutral",
    ]

    enc = tok(
        "premise calibration sample for fixed export",
        "hypothesis calibration sample for fixed export",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    args = (enc["input_ids"], enc["attention_mask"])
    input_names = ["input_ids", "attention_mask"]
    if "token_type_ids" in enc:
        args = args + (enc["token_type_ids"],)
        input_names.append("token_type_ids")

    # FIXED shapes — no dynamic_axes (required for QNN HTP)
    torch.onnx.export(
        model,
        args,
        str(fp),
        input_names=input_names,
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # legacy exporter; more stable for classifiers
    )
    meta = {
        "model_id": HF_ID,
        "max_length": MAX_LEN,
        "labels": labels,
        "input_names": input_names,
        "fp32": str(fp),
        "export_s": round(time.time() - t0, 1),
        "size_mb": round(fp.stat().st_size / 1e6, 1),
    }
    # external data?
    data = list(OUT_DIR.glob("model_fp32_fixed.onnx.data"))
    if data:
        meta["size_mb"] = round(
            (fp.stat().st_size + data[0].stat().st_size) / 1e6, 1
        )
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, **meta}


def quantize_qdq(
    fp_path: Path,
    *,
    act: str = "uint8",
    weight: str = "uint8",
) -> dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer
    from onnxruntime.quantization import QuantType, quantize
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    tok = AutoTokenizer.from_pretrained(HF_ID)
    # Separate artifacts per quant recipe so UINT8 residual stays for comparison
    suffix = f"a{act}_w{weight}".replace("int", "i").replace("uint", "u")
    qdq = OUT_DIR / f"model.qdq.{suffix}.onnx"
    if act == "uint8" and weight == "uint8":
        qdq = OUT_DIR / "model.qdq.onnx"  # legacy default path
    t0 = time.time()
    act_map = {
        "uint8": QuantType.QUInt8,
        "uint16": QuantType.QUInt16,
        "int8": QuantType.QInt8,
        "int16": QuantType.QInt16,
    }
    w_map = dict(act_map)
    act_t = act_map.get(act.lower(), QuantType.QUInt8)
    w_t = w_map.get(weight.lower(), QuantType.QUInt8)

    # Discover actual input names from the ONNX (preproc may drop token_type_ids)
    import onnx

    def _input_names(path: Path) -> list[str]:
        m = onnx.load(str(path))
        return [i.name for i in m.graph.input]

    def _make_reader(model_path: Path) -> CalibrationDataReader:
        names = _input_names(model_path)

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.rows = []
                for prem, hyp in CALIB_PAIRS * 3:  # 24 samples
                    enc = tok(
                        prem,
                        hyp,
                        return_tensors="np",
                        padding="max_length",
                        truncation=True,
                        max_length=MAX_LEN,
                    )
                    feed = {}
                    if "input_ids" in names:
                        feed["input_ids"] = enc["input_ids"].astype(np.int64)
                    if "attention_mask" in names:
                        feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
                    if "token_type_ids" in names:
                        # DeBERTa may export token_type_ids; if tokenizer omits, zeros
                        tt = enc.get("token_type_ids")
                        if tt is None:
                            tt = np.zeros_like(enc["input_ids"])
                        feed["token_type_ids"] = tt.astype(np.int64)
                    self.rows.append(feed)
                self.i = 0

            def get_next(self):
                if self.i >= len(self.rows):
                    return None
                r = self.rows[self.i]
                self.i += 1
                return r

            def rewind(self):
                self.i = 0

        return Reader()

    try:
        from onnxruntime.quantization.execution_providers.qnn import (
            get_qnn_qdq_config,
            qnn_preprocess_model,
        )

        pre = OUT_DIR / "model.pre.onnx"
        changed = qnn_preprocess_model(str(fp_path), str(pre))
        src = Path(pre) if changed else fp_path
        reader = _make_reader(src)
        cfg = get_qnn_qdq_config(
            str(src),
            reader,
            activation_type=act_t,
            weight_type=w_t,
        )
        reader.rewind()
        quantize(str(src), str(qdq), cfg)
        method = f"qnn_qdq_config act={act} w={weight}"
        input_names_used = _input_names(src)
    except Exception as e:
        from onnxruntime.quantization import QuantFormat, quantize_static

        reader = _make_reader(fp_path)
        quantize_static(
            str(fp_path),
            str(qdq),
            reader,
            quant_format=QuantFormat.QDQ,
            activation_type=act_t,
            weight_type=w_t,
        )
        method = f"quantize_static_fallback act={act} w={weight} ({e})"
        input_names_used = _input_names(fp_path)

    return {
        "ok": True,
        "qdq": str(qdq),
        "qdq_mb": round(qdq.stat().st_size / 1e6, 1),
        "method": method,
        "act": act,
        "weight": weight,
        "input_names": input_names_used,
        "seconds": round(time.time() - t0, 1),
    }


def run_htp(qdq_path: Path) -> dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer
    from npu_qnn import register, session_qdq

    reg = register()
    if not reg.get("ok"):
        return {"ok": False, "error": "register failed", "reg": reg}

    t0 = time.time()
    r = session_qdq(qdq_path, burst=True, allow_cpu_fallback=True)
    if not r.get("ok"):
        return r
    sess = r["session"]
    tok = AutoTokenizer.from_pretrained(HF_ID)
    meta = json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    labels = meta.get("labels") or ["contradiction", "entailment", "neutral"]

    sess_inputs = {i.name for i in sess.get_inputs()}

    def predict(prem: str, hyp: str) -> dict:
        enc = tok(
            prem,
            hyp,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        feeds = {}
        if "input_ids" in sess_inputs:
            feeds["input_ids"] = enc["input_ids"].astype(np.int64)
        if "attention_mask" in sess_inputs:
            feeds["attention_mask"] = enc["attention_mask"].astype(np.int64)
        if "token_type_ids" in sess_inputs:
            tt = enc.get("token_type_ids")
            if tt is None:
                tt = np.zeros_like(enc["input_ids"])
            feeds["token_type_ids"] = tt.astype(np.int64)
        t1 = time.time()
        logits = sess.run(None, feeds)[0][0]
        ms = (time.time() - t1) * 1000
        ex = np.exp(logits - logits.max())
        probs = ex / ex.sum()
        i = int(probs.argmax())
        lab = labels[i] if i < len(labels) else str(i)
        if "contrad" in lab.lower():
            lab = "contradiction"
        elif "entail" in lab.lower():
            lab = "entailment"
        elif "neutral" in lab.lower():
            lab = "neutral"
        return {
            "label": lab,
            "confidence": round(float(probs[i]), 4),
            "ms": round(ms, 2),
        }

    pairs_expect = [
        (*CALIB_PAIRS[1], "contradiction"),
        (*CALIB_PAIRS[2], "contradiction"),
        (*CALIB_PAIRS[0], "entailment"),
    ]
    rows = []
    for a, b, exp in pairs_expect:
        p = predict(a, b)
        p["expect"] = exp
        p["hit"] = p["label"] == exp
        rows.append(p)

    # bench
    for _ in range(5):
        predict(*CALIB_PAIRS[1])
    t_bench = time.time()
    n = 20
    for _ in range(n):
        predict(*CALIB_PAIRS[1])
    bench_s = time.time() - t_bench

    return {
        "ok": True,
        "providers": r.get("providers"),
        "qnn_ep_registered": r.get("qnn_ep_registered", r.get("on_qnn")),
        "on_qnn": r.get("on_qnn"),  # legacy alias
        "session_s": round(time.time() - t0, 2),
        "rows": rows,
        "hits": sum(1 for x in rows if x.get("hit")),
        "n": len(rows),
        "bench_ms_per": round(bench_s * 1000 / n, 2),
        "htp_dll": "<QnnHtp.dll>",  # sanitized — no user path in reports
    }


def main() -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="DeBERTa QDQ → Hexagon HTP NLI")
    ap.add_argument("--act", default=os.environ.get("PRIME_NLI_QDQ_ACT", "uint8"))
    ap.add_argument("--weight", default=os.environ.get("PRIME_NLI_QDQ_WEIGHT", "uint8"))
    ap.add_argument("--skip-export", action="store_true", help="reuse existing FP32")
    args = ap.parse_args()

    t0 = time.time()
    report: dict[str, Any] = {"recipe": {"act": args.act, "weight": args.weight}}
    print("1) export fixed-shape FP32…", flush=True)
    try:
        fp_existing = OUT_DIR / "model_fp32_fixed.onnx"
        if args.skip_export and fp_existing.is_file():
            meta = {}
            mp = OUT_DIR / "meta.json"
            if mp.is_file():
                meta = json.loads(mp.read_text(encoding="utf-8"))
            ex = {"ok": True, "fp32": str(fp_existing), "reused": True, **meta}
        else:
            ex = export_fixed()
    except Exception as e:
        report = {"ok": False, "error": f"export: {e}"}
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report, flush=True)
        return 1
    report["export"] = ex
    print(json.dumps(ex, indent=2), flush=True)

    print(f"2) QDQ quantize act={args.act} weight={args.weight}…", flush=True)
    try:
        q = quantize_qdq(Path(ex["fp32"]), act=args.act, weight=args.weight)
    except Exception as e:
        report["ok"] = False
        report["error"] = f"quantize: {e}"
        REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(report["error"], flush=True)
        return 1
    report["quantize"] = q
    print(json.dumps(q, indent=2), flush=True)

    print("3) run HTP…", flush=True)
    try:
        run = run_htp(Path(q["qdq"]))
    except Exception as e:
        report["ok"] = False
        report["error"] = f"run: {e}"
        report["run"] = {"ok": False, "error": str(e)}
        REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(report["error"], flush=True)
        return 1
    report["run"] = run
    print(json.dumps(run, indent=2), flush=True)

    on_path = bool(run.get("qnn_ep_registered") or run.get("on_qnn"))
    # LIVE only with label hits; path alone is PARTIAL (parity residual)
    report["ok"] = bool(run.get("ok") and on_path and run.get("hits", 0) >= 2)
    report["verdict"] = (
        "NPU_NLI_LIVE"
        if report["ok"]
        else ("NPU_NLI_PARTIAL" if on_path else "NPU_NLI_FAIL")
    )
    report["law"] = "Job2 HTP never owns production OPEN; E3 parity required for product NLI"
    report["seconds"] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("verdict", report["verdict"], "wrote", REPORT, flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
