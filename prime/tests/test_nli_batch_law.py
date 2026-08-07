#!/usr/bin/env python3
"""
NLI batching + NPU gate isolation unit tests (GO_MEASURE law).

No live ORT model required for isolation tests.
Optional integration tests skip if DeBERTa ORT export missing.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class TestProvidersAutoNeverQnn(unittest.TestCase):
    """PRIME_ACCEL=auto must not put QNN first (or at all)."""

    def test_auto_cpu_only_even_if_qnn_available(self) -> None:
        import accel_nli_ort as ort_mod

        fake_ort = MagicMock()
        fake_ort.get_available_providers.return_value = [
            "QNNExecutionProvider",
            "CPUExecutionProvider",
            "DmlExecutionProvider",
        ]
        with patch.dict(os.environ, {"PRIME_ACCEL": "auto"}, clear=False):
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                # re-bind import site inside function
                order = ort_mod._providers()
        self.assertEqual(order, ["CPUExecutionProvider"])
        self.assertNotIn("QNNExecutionProvider", order)

    def test_qnn_only_when_explicit_pref(self) -> None:
        import accel_nli_ort as ort_mod

        fake_ort = MagicMock()
        fake_ort.get_available_providers.return_value = [
            "QNNExecutionProvider",
            "CPUExecutionProvider",
        ]
        with patch.dict(os.environ, {"PRIME_ACCEL": "qnn"}, clear=False):
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                order = ort_mod._providers()
        self.assertEqual(order[0], "QNNExecutionProvider")
        self.assertIn("CPUExecutionProvider", order)


class TestEnsureProductSessionForceCpu(unittest.TestCase):
    def test_auto_env_forces_cpu_flag(self) -> None:
        import accel_nli_ort as ort_mod

        with patch.object(
            ort_mod,
            "load_session",
            return_value={
                "ok": True,
                "active_provider": "CPUExecutionProvider",
                "providers": ["CPUExecutionProvider"],
            },
        ):
            with patch.dict(os.environ, {"PRIME_ACCEL": "auto"}, clear=False):
                # force_cpu=False but auto env must still pin force_cpu
                st = ort_mod._ensure_product_session(force_cpu=False)
        self.assertTrue(st.get("ok"))
        self.assertTrue(st.get("force_cpu"))

    def test_refuse_qnn_active_on_product_path(self) -> None:
        import accel_nli_ort as ort_mod

        with patch.object(
            ort_mod,
            "load_session",
            return_value={
                "ok": True,
                "active_provider": "QNNExecutionProvider",
                "providers": ["QNNExecutionProvider", "CPUExecutionProvider"],
            },
        ):
            with patch.dict(os.environ, {"PRIME_ACCEL": "cpu"}, clear=False):
                st = ort_mod._ensure_product_session(force_cpu=True)
        self.assertFalse(st.get("ok"))
        self.assertIn("refused QNN", str(st.get("error") or ""))


class TestPredictBatchShapes(unittest.TestCase):
    def test_empty_batch(self) -> None:
        from accel_nli_ort import predict_batch

        self.assertEqual(predict_batch([]), [])

    def test_batch_failure_envelope_when_session_fails(self) -> None:
        import accel_nli_ort as ort_mod

        with patch.object(
            ort_mod,
            "_ensure_product_session",
            return_value={"ok": False, "error": "no_session"},
        ):
            rows = ort_mod.predict_batch([("a", "b"), ("c", "d")], force_cpu=True)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not r.get("ok") for r in rows))
        self.assertTrue(all(r.get("not_open_authority") for r in rows))
        self.assertTrue(all(r.get("job2_owns_open") is False for r in rows))


class TestMutualAndGlueBatch(unittest.TestCase):
    def test_mutual_from_ab_ba_agrees(self) -> None:
        from entailment_glue import _mutual_from_ab_ba

        r = _mutual_from_ab_ba(
            {
                "ok": True,
                "label": "entailment",
                "confidence": 0.91,
                "model": "t",
                "probs": {},
            },
            {
                "ok": True,
                "label": "entailment",
                "confidence": 0.92,
                "model": "t",
                "probs": {},
            },
            p_floor=0.8,
            batched=True,
        )
        self.assertTrue(r["agrees"])
        self.assertEqual(r["gate"], "PASS")
        self.assertTrue(r["batched"])
        self.assertTrue(r["not_open_authority"])

    def test_mutual_contra_stop(self) -> None:
        from entailment_glue import _mutual_from_ab_ba

        r = _mutual_from_ab_ba(
            {
                "ok": True,
                "label": "contradiction",
                "confidence": 0.99,
                "model": "t",
                "probs": {},
            },
            {
                "ok": True,
                "label": "neutral",
                "confidence": 0.5,
                "model": "t",
                "probs": {},
            },
            p_floor=0.8,
            batched=False,
        )
        self.assertFalse(r["agrees"])
        self.assertEqual(r["gate"], "STOP")

    def test_glue_agreement_batch_uses_predict_batch(self) -> None:
        import entailment_glue as eg

        fake_rows = [
            {
                "ok": True,
                "label": "entailment",
                "confidence": 0.9,
                "agrees": True,
                "gate": "PASS",
                "not_open_authority": True,
            },
            {
                "ok": True,
                "label": "neutral",
                "confidence": 0.4,
                "agrees": False,
                "gate": "NEED_INFO",
                "not_open_authority": True,
            },
        ]
        with patch.dict(os.environ, {"PRIME_NLI_ORT": "1"}, clear=False):
            with patch(
                "accel_nli_ort.predict_batch", return_value=fake_rows
            ) as pb:
                out = eg.glue_agreement_batch(
                    [("intent", "claim1"), ("intent", "claim2")],
                    prefer="auto",
                )
        pb.assert_called_once()
        kwargs = pb.call_args.kwargs
        self.assertTrue(kwargs.get("force_cpu"))
        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("n"), 2)
        self.assertEqual(out.get("n_agrees"), 1)
        self.assertFalse(out.get("job2_owns_open"))


class TestHtpPreferIsolation(unittest.TestCase):
    def test_prefer_htp_refuses_when_parity_red(self) -> None:
        import entailment_glue as eg

        with patch(
            "measure_fabric.nli_htp_parity_pass",
            return_value={"ok": False, "reason": "missing"},
        ):
            with patch.object(
                eg,
                "nli_ort",
                return_value={
                    "ok": True,
                    "label": "entailment",
                    "confidence": 0.9,
                    "agrees": True,
                    "gate": "PASS",
                    "provider": "CPUExecutionProvider",
                    "not_open_authority": True,
                    "job2_owns_open": False,
                },
            ) as nli:
                r = eg.glue_agreement("h", "d", prefer="htp")
        nli.assert_called()
        # product still CPU force_cpu path
        self.assertEqual(nli.call_args.kwargs.get("force_cpu"), True)
        self.assertIn("htp_refused", r)
        self.assertFalse(r.get("job2_owns_open"))


class TestMeasureFabricGateAuthority(unittest.TestCase):
    def test_route_job2_never_htp_gate(self) -> None:
        from measure_fabric import route_job2

        with patch(
            "measure_fabric.nli_htp_parity_pass",
            return_value={"ok": True, "reason": "pass"},
        ):
            r = route_job2()
        self.assertFalse(r.get("htp_is_gate_authority"))
        self.assertFalse(r.get("job2_owns_open"))
        self.assertEqual(r.get("gate_authority_order"), ["ort_cpu", "cross_encoder", "lfm"])
        # even if parity green, HTP may only appear on measure fabric order
        self.assertEqual(r["gate_authority_order"][0], "ort_cpu")


class TestSheafNotInJobModules(unittest.TestCase):
    """Sheaf Laplacian ALU must not live in Job1/2 ORT modules."""

    def test_no_eigsh_in_accel_or_glue(self) -> None:
        accel = (SCRIPTS / "accel_nli_ort.py").read_text(encoding="utf-8")
        glue = (SCRIPTS / "entailment_glue.py").read_text(encoding="utf-8")
        cg = (SCRIPTS / "compute_graph.py").read_text(encoding="utf-8")
        for blob in (accel, glue):
            self.assertNotIn("eigsh", blob)
            self.assertNotIn("Vietoris", blob)
        # process DAG may mention sheaf as phase name only — not sparse eigensolver
        self.assertNotIn("eigsh", cg)
        self.assertNotIn("scipy.sparse", cg)


class TestOptionalLiveBatchParity(unittest.TestCase):
    """Integration: skip if ORT model not exported."""

    def test_seq_vs_batch_label_parity_if_model(self) -> None:
        from accel_nli_ort import MODEL_DIR, ONNX_PATH, predict, predict_batch

        if not ONNX_PATH.is_file():
            self.skipTest(f"ORT model missing: {ONNX_PATH}")
        pairs = [
            (
                "Aboutness must not promote OPEN.",
                "NLI owns agreement; cosine never OPEN.",
            ),
            (
                "E_ref is production-ready.",
                "E_ref still has residue and is not production-ready.",
            ),
        ]
        seq = [predict(a, b, force_cpu=True) for a, b in pairs]
        bat = predict_batch(pairs, force_cpu=True)
        self.assertEqual(len(seq), len(bat))
        for s, b in zip(seq, bat, strict=True):
            self.assertTrue(s.get("ok") and b.get("ok"), msg=f"s={s} b={b}")
            self.assertEqual(s.get("label"), b.get("label"))
            self.assertTrue(b.get("force_cpu"))
            self.assertNotIn("QNN", str(b.get("provider") or ""))


if __name__ == "__main__":
    unittest.main()
