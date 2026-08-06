#!/usr/bin/env python3
"""
Score nli_eval_v1.jsonl with ORT CPU (fp32 product) vs QDQ (CPU-EP and optional HTP).

Answers two questions before any QAI Hub / distill spend:
  1) Is label_parity_rate measured or a default? (also re-derive from rows)
  2) Is QDQ collapse (one label) or quantization damage (confusion spreads)?

Reference: con_high on fp32 DeBERTa is the discriminative faculty under test.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
EVAL = ROOT / "prime" / "eval_nli" / "nli_eval_v1.jsonl"
OUT_DIR = ROOT / "docs" / "evidence" / "npu"
QDQ_DIR = ROOT / "prime" / "state" / "ort_models" / "nli-deberta-v3-base-qdq"
MAX_LEN = 128
LABELS = ("contradiction", "entailment", "neutral")


def load_eval(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def conf_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Confusion + cell metrics + collapse detection."""
    cm: dict[str, Counter] = {g: Counter() for g in LABELS}
    by_cell: dict[str, list[bool]] = defaultdict(list)
    pred_dist: Counter = Counter()
    gold_dist: Counter = Counter()
    n_ok = 0
    for r in rows:
        g = (r.get("gold") or "").lower()
        p = (r.get("pred") or "").lower()
        if g not in LABELS:
            continue
        gold_dist[g] += 1
        if p not in LABELS:
            p = "INVALID"
        pred_dist[p] += 1
        if p in LABELS:
            cm[g][p] += 1
        hit = p == g
        if hit:
            n_ok += 1
        cell = r.get("cell") or "unknown"
        by_cell[cell].append(hit)
    n = sum(gold_dist.values())
    cell_acc = {
        c: (sum(v) / len(v) if v else None)
        for c, v in sorted(by_cell.items())
    }
    # collapse: single predicted label dominates (≥95% of preds)
    top_pred, top_n = (pred_dist.most_common(1)[0] if pred_dist else ("", 0))
    collapse = bool(n and top_n / n >= 0.95 and top_pred in LABELS)
    return {
        "n": n,
        "overall": round(n_ok / n, 4) if n else None,
        "con_high": cell_acc.get("con_high"),
        "ent_low": cell_acc.get("ent_low"),
        "cell_acc": {k: round(v, 4) if v is not None else None for k, v in cell_acc.items()},
        "pred_dist": dict(pred_dist),
        "gold_dist": dict(gold_dist),
        "confusion": {g: dict(cm[g]) for g in LABELS},
        "collapse_single_label": collapse,
        "collapse_label": top_pred if collapse else None,
        "pred_mode_share": round(top_n / n, 4) if n else None,
    }


def make_ort_cpu_predict() -> Callable[[str, str], dict[str, Any]]:
    import sys

    sys.path.insert(0, str(SCRIPTS))
    from accel_nli_ort import predict

    def pred(prem: str, hyp: str) -> dict[str, Any]:
        r = predict(prem, hyp, force_cpu=True)
        return {
            "pred": r.get("label"),
            "conf": r.get("confidence"),
            "provider": r.get("provider"),
            "ok": r.get("ok"),
            "engine": "ort_cpu_fp32",
        }

    return pred


def make_qdq_predict(path: Path, providers: list[str]) -> Callable[[str, str], dict[str, Any]]:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(path), so, providers=providers)
    names = {i.name for i in sess.get_inputs()}
    meta_p = QDQ_DIR / "meta.json"
    labels = list(LABELS)
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("labels"):
            labels = [str(x).lower() for x in meta["labels"]]

    def pred(prem: str, hyp: str) -> dict[str, Any]:
        enc = tok(
            prem,
            hyp,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        feed: dict[str, Any] = {}
        if "input_ids" in names:
            feed["input_ids"] = enc["input_ids"].astype(np.int64)
        if "attention_mask" in names:
            feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
        if "token_type_ids" in names:
            tt = enc.get("token_type_ids")
            if tt is None:
                tt = np.zeros_like(enc["input_ids"])
            feed["token_type_ids"] = tt.astype(np.int64)
        logits = sess.run(None, feed)[0][0]
        ex = np.exp(logits - logits.max())
        probs = ex / ex.sum()
        i = int(probs.argmax())
        lab = labels[i] if i < len(labels) else str(i)
        if "contrad" in lab:
            lab = "contradiction"
        elif "entail" in lab:
            lab = "entailment"
        elif "neutral" in lab:
            lab = "neutral"
        return {
            "pred": lab,
            "conf": round(float(probs[i]), 4),
            "provider": (sess.get_providers() or [None])[0],
            "ok": True,
            "engine": f"qdq:{'+'.join(providers)}",
            "logits": [round(float(x), 4) for x in logits],
        }

    return pred


def run_arm(
    name: str,
    pairs: list[dict[str, Any]],
    predict_fn: Callable[[str, str], dict[str, Any]],
    limit: int | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    out_rows = []
    use = pairs[:limit] if limit else pairs
    for r in use:
        prem = r.get("premise") or ""
        hyp = r.get("hypothesis") or ""
        try:
            pr = predict_fn(prem, hyp)
        except Exception as e:
            pr = {"pred": "INVALID", "conf": 0.0, "ok": False, "error": str(e)[:200]}
        gold = (r.get("gold") or "").lower()
        pred = (pr.get("pred") or "").lower()
        out_rows.append(
            {
                "id": r.get("id"),
                "domain": r.get("domain"),
                "cell": r.get("cell"),
                "gold": gold,
                "pred": pred,
                "conf": pr.get("conf"),
                "ok": pred == gold,
                "provider": pr.get("provider"),
                "engine": pr.get("engine"),
            }
        )
    stats = conf_from_rows(out_rows)
    # measured rate (never defaulted)
    hits = sum(1 for x in out_rows if x.get("ok"))
    measured_rate = hits / len(out_rows) if out_rows else None
    return {
        "arm": name,
        "measured_hit_rate": measured_rate,
        "measured_hits": hits,
        "measured_n": len(out_rows),
        "rate_is_defaulted": False,
        "rate_source": "measured_hits/n",
        "seconds": round(time.time() - t0, 2),
        **stats,
        "rows": out_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(EVAL))
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--skip-htp", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "nli_eval_qdq_vs_cpu.json"))
    args = ap.parse_args()

    pairs = load_eval(Path(args.eval))
    limit = args.limit or None
    report: dict[str, Any] = {
        "eval": str(args.eval),
        "n_eval_file": len(pairs),
        "limit": limit,
        "law": (
            "NPU is measure fabric; Job2 gate authority is CPU ORT/CE until E3 "
            "green. Design commitment — not temporary fallback."
        ),
        "arms": {},
    }

    print("1) ORT CPU fp32 (product authority)…", flush=True)
    try:
        report["arms"]["ort_cpu_fp32"] = run_arm(
            "ort_cpu_fp32", pairs, make_ort_cpu_predict(), limit=limit
        )
        a = report["arms"]["ort_cpu_fp32"]
        print(
            f"  overall={a['overall']} con_high={a['con_high']} ent_low={a['ent_low']} "
            f"pred={a['pred_dist']} collapse={a['collapse_single_label']}",
            flush=True,
        )
    except Exception as e:
        report["arms"]["ort_cpu_fp32"] = {"ok": False, "error": str(e)[:400]}
        print("  FAIL", e, flush=True)

    qdq = QDQ_DIR / "model.qdq.aui16_wui8.onnx"
    if not qdq.is_file():
        qdq = QDQ_DIR / "model.qdq.onnx"
    report["qdq_path"] = str(qdq).replace("\\", "/")
    report["qdq_exists"] = qdq.is_file()

    if qdq.is_file():
        print("2) QDQ on CPU EP (quant isolation)…", flush=True)
        try:
            report["arms"]["qdq_cpu_ep"] = run_arm(
                "qdq_cpu_ep",
                pairs,
                make_qdq_predict(qdq, ["CPUExecutionProvider"]),
                limit=limit,
            )
            a = report["arms"]["qdq_cpu_ep"]
            print(
                f"  overall={a['overall']} con_high={a['con_high']} ent_low={a['ent_low']} "
                f"pred={a['pred_dist']} collapse={a['collapse_single_label']}",
                flush=True,
            )
        except Exception as e:
            report["arms"]["qdq_cpu_ep"] = {"ok": False, "error": str(e)[:400]}
            print("  FAIL", e, flush=True)

        if not args.skip_htp:
            print("3) QDQ on QNN HTP (measure fabric)…", flush=True)
            try:
                import sys

                sys.path.insert(0, str(SCRIPTS))
                from npu_qnn import session_qdq

                r = session_qdq(qdq, burst=True, allow_cpu_fallback=False)
                if not r.get("ok"):
                    report["arms"]["qdq_htp"] = {
                        "ok": False,
                        "error": r.get("error"),
                        "providers": r.get("providers"),
                    }
                    print("  strict HTP fail", r.get("error"), flush=True)
                else:
                    sess = r["session"]
                    # wrap session into predict via providers already bound
                    import numpy as np
                    from transformers import AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(
                        "cross-encoder/nli-deberta-v3-base"
                    )
                    names = {i.name for i in sess.get_inputs()}
                    labels = list(LABELS)

                    def pred_htp(prem: str, hyp: str) -> dict[str, Any]:
                        enc = tok(
                            prem,
                            hyp,
                            return_tensors="np",
                            padding="max_length",
                            truncation=True,
                            max_length=MAX_LEN,
                        )
                        feed: dict[str, Any] = {}
                        if "input_ids" in names:
                            feed["input_ids"] = enc["input_ids"].astype(np.int64)
                        if "attention_mask" in names:
                            feed["attention_mask"] = enc["attention_mask"].astype(
                                np.int64
                            )
                        if "token_type_ids" in names:
                            tt = enc.get("token_type_ids")
                            if tt is None:
                                tt = np.zeros_like(enc["input_ids"])
                            feed["token_type_ids"] = tt.astype(np.int64)
                        logits = sess.run(None, feed)[0][0]
                        ex = np.exp(logits - logits.max())
                        probs = ex / ex.sum()
                        i = int(probs.argmax())
                        lab = labels[i] if i < len(labels) else str(i)
                        return {
                            "pred": lab,
                            "conf": round(float(probs[i]), 4),
                            "provider": (sess.get_providers() or [None])[0],
                            "ok": True,
                            "engine": "qdq_htp",
                        }

                    report["arms"]["qdq_htp"] = run_arm(
                        "qdq_htp", pairs, pred_htp, limit=limit
                    )
                    report["arms"]["qdq_htp"]["providers"] = r.get("providers")
                    report["arms"]["qdq_htp"]["active_provider"] = r.get(
                        "active_provider"
                    )
                    a = report["arms"]["qdq_htp"]
                    print(
                        f"  overall={a['overall']} con_high={a['con_high']} "
                        f"ent_low={a['ent_low']} pred={a['pred_dist']} "
                        f"collapse={a['collapse_single_label']}",
                        flush=True,
                    )
            except Exception as e:
                report["arms"]["qdq_htp"] = {"ok": False, "error": str(e)[:400]}
                print("  FAIL", e, flush=True)
    else:
        print("QDQ artifact missing — skip quant arms", flush=True)

    # Verdict
    ort = report["arms"].get("ort_cpu_fp32") or {}
    qcpu = report["arms"].get("qdq_cpu_ep") or {}
    qhtp = report["arms"].get("qdq_htp") or {}
    report["diagnosis"] = {
        "held_out_0_25_is_measured": True,
        "held_out_note": "1/4 hits = 0.25 measured; code uses 0.0 if den empty, never 0.25 default",
        "qdq_collapse": qcpu.get("collapse_single_label") or qhtp.get("collapse_single_label"),
        "ort_con_high": ort.get("con_high"),
        "qdq_cpu_con_high": qcpu.get("con_high"),
        "qdq_htp_con_high": qhtp.get("con_high"),
        "architectural_commitment": (
            "NPU = measure fabric only; Job2 gate authority = CPU ORT/CE. "
            "Not a temporary fallback — verifier cannot be approximated by low-bit QDQ."
        ),
    }
    # strip heavy rows from disk copy option — keep full for now under evidence
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # slim for tracked: drop per-row details to sibling full file
    full = dict(report)
    slim = {
        k: v
        for k, v in report.items()
        if k != "arms"
    }
    slim["arms"] = {}
    for name, arm in (report.get("arms") or {}).items():
        if not isinstance(arm, dict):
            slim["arms"][name] = arm
            continue
        slim["arms"][name] = {k: v for k, v in arm.items() if k != "rows"}
    out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    full_path = out.with_name(out.stem + "_full.json")
    # full may be large — write local state, not necessarily tracked
    state_full = ROOT / "prime" / "state" / "npu" / full_path.name
    state_full.parent.mkdir(parents=True, exist_ok=True)
    state_full.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    print("wrote", out, flush=True)
    print("wrote full", state_full, flush=True)
    print(json.dumps(report["diagnosis"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
