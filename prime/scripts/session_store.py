"""Prime session store — durable OPEN|STOP state across tool calls."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from compute_graph import ComputeGraph, advance, block, plan_graph


class SessionStore:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "session.json"
        self.s: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return self._empty()

    def _empty(self) -> dict[str, Any]:
        return {
            "session_id": None,
            "workspace": None,
            "intent": None,
            "phase": "idle",
            "restrict": None,
            "graph": None,
            "claims": [],
            "residue": [],
            "measures": [],
            "audits": [],
            "questions": [],
            "projections": {"human": None, "domains": {}},
            "alignments": [],
            "lm": {"base_url": "http://127.0.0.1:1234/v1", "default_model": None},
            "created_at": None,
            "updated_at": None,
            "meta": {
                "law": "restrict→measure→audit→OPEN|STOP",
                "force_open": False,
                "topology": "prime-session compute graph",
                "language": "bilateral projection (human ↔ domain); glue before OPEN",
            },
        }

    def save(self) -> None:
        self.s["updated_at"] = time.time()
        self.path.write_text(json.dumps(self.s, indent=2, default=str), encoding="utf-8")

    def start(self, workspace: str, intent: str = "", modes: list[str] | None = None) -> dict[str, Any]:
        sid = "ps_" + uuid.uuid4().hex[:12]
        g = plan_graph(intent or "(no intent yet)", modes=modes or ["code", "lm", "rplc"])
        self.s = self._empty()
        self.s.update(
            {
                "session_id": sid,
                "workspace": str(Path(workspace).resolve()),
                "intent": intent,
                "phase": "META_META",
                "graph": g.to_dict(),
                "created_at": time.time(),
            }
        )
        self.save()
        return {
            "ok": True,
            "session_id": sid,
            "workspace": self.s["workspace"],
            "phase": self.s["phase"],
            "graph_mermaid": g.to_mermaid(),
            "graph_id": g.graph_id,
            "message": "Session live. Plot graph (META_META→META→RESTRICT) before CODE. Residue never forced.",
        }

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "session_id": self.s.get("session_id"),
            "workspace": self.s.get("workspace"),
            "intent": self.s.get("intent"),
            "phase": self.s.get("phase"),
            "restrict": self.s.get("restrict"),
            "n_claims": len(self.s.get("claims") or []),
            "n_residue": len(self.s.get("residue") or []),
            "n_measures": len(self.s.get("measures") or []),
            "n_audits": len(self.s.get("audits") or []),
            "open_questions": [q for q in (self.s.get("questions") or []) if not q.get("answered")],
            "last_audit": (self.s.get("audits") or [None])[-1],
            "graph_current": (self.s.get("graph") or {}).get("current"),
            "graph_path": (self.s.get("graph") or {}).get("path"),
            "meta": self.s.get("meta"),
        }

    def set_restrict(self, goal: str, non_goals: list[str], success: list[str], constraints: list[str]) -> dict[str, Any]:
        self.s["restrict"] = {
            "goal": goal,
            "non_goals": non_goals,
            "success_checks": success,
            "constraints": constraints,
            "t": time.time(),
        }
        self.s["phase"] = "RESTRICT"
        self._graph_advance("RESTRICT", {"goal": goal})
        self.save()
        return {"ok": True, "restrict": self.s["restrict"], "phase": "RESTRICT"}

    def _graph_obj(self) -> ComputeGraph | None:
        gd = self.s.get("graph")
        if not gd:
            return None
        g = ComputeGraph(graph_id=gd["graph_id"], intent=gd.get("intent", ""))
        g.current = gd.get("current")
        g.path = list(gd.get("path") or [])
        g.meta_note = gd.get("meta_note", "")
        from compute_graph import Edge, Node

        for k, v in (gd.get("nodes") or {}).items():
            g.nodes[k] = Node(**v)
        for e in gd.get("edges") or []:
            g.edges.append(Edge(**e))
        return g

    def _graph_advance(self, to: str, payload: dict | None = None) -> dict[str, Any]:
        g = self._graph_obj()
        if not g:
            return {"ok": False, "error": "no graph"}
        r = advance(g, to, payload)
        if r.get("ok"):
            self.s["graph"] = g.to_dict()
            self.s["phase"] = to
            self.save()
        return r

    def graph_plan(self, intent: str, modes: list[str] | None = None) -> dict[str, Any]:
        g = plan_graph(intent, modes=modes)
        self.s["intent"] = intent or self.s.get("intent")
        self.s["graph"] = g.to_dict()
        self.s["phase"] = "GRAPH"
        self.save()
        return {"ok": True, "graph": g.to_dict()}

    def graph_advance(self, to: str, payload: dict | None = None) -> dict[str, Any]:
        return self._graph_advance(to, payload)

    def graph_block(self, node: str, reason: str) -> dict[str, Any]:
        g = self._graph_obj()
        if not g:
            return {"ok": False, "error": "no graph"}
        r = block(g, node, reason)
        self.s["graph"] = g.to_dict()
        self.save()
        return r

    def need_question(self, question: str, why: str, options: list[str] | None = None) -> dict[str, Any]:
        q = {
            "id": "q_" + uuid.uuid4().hex[:8],
            "question": question,
            "why": why,
            "options": options or [],
            "answered": False,
            "answer": None,
            "t": time.time(),
        }
        self.s.setdefault("questions", []).append(q)
        self.save()
        return {
            "ok": True,
            "halt_for_human": True,
            "question": q,
            "instruction": "ACCURACY GATE: ask the human this before proceeding. Do not invent the answer.",
        }

    def answer_question(self, qid: str, answer: str) -> dict[str, Any]:
        for q in self.s.get("questions") or []:
            if q["id"] == qid:
                q["answered"] = True
                q["answer"] = answer
                self.save()
                return {"ok": True, "question": q}
        return {"ok": False, "error": "question not found"}

    def add_measure(self, report: dict[str, Any]) -> None:
        # Tag purpose for task-purpose gating (AGREEMENT ≠ TOPICALITY)
        try:
            from purpose_gate import refuse_agreement_via_cosine, tag_measure

            if "purposes" not in report and report.get("mode"):
                report = {**report, **tag_measure(str(report.get("mode")), **report)}
            bad = refuse_agreement_via_cosine(report)
            if bad:
                report = {**report, "purpose_type_error": bad}
        except Exception:
            pass
        self.s.setdefault("measures", []).append({"t": time.time(), **report})
        self.save()

    def add_audit(self, verdict: str, reasons: list[str], controls: dict | None = None) -> dict[str, Any]:
        """
        AUDIT OPEN = production bar (survived measures under law).
        Not the same as deep-loop ACCEPTED (work item tracking).
        """
        if verdict not in ("OPEN", "STOP"):
            return {"ok": False, "error": "verdict must be OPEN or STOP"}
        # never force OPEN without measures
        if verdict == "OPEN" and not self.s.get("measures"):
            verdict = "STOP"
            reasons = list(reasons) + ["force-OPEN blocked: no measures recorded"]
        if verdict == "OPEN" and not self.s.get("restrict"):
            verdict = "STOP"
            reasons = list(reasons) + ["force-OPEN blocked: RESTRICT not set"]
        # purpose gate: production OPEN needs EXISTENCE + AGREEMENT (or honest STOP)
        purpose_gate = None
        if verdict == "OPEN":
            try:
                from purpose_gate import gate_claim

                purpose_gate = gate_claim("production", measures=self.s.get("measures") or [])
                if not purpose_gate.get("ok"):
                    verdict = "STOP"
                    reasons = list(reasons) + [
                        purpose_gate.get("error")
                        or "purpose mismatch: cannot AUDIT OPEN without required measure purposes"
                    ]
            except Exception as e:
                purpose_gate = {"ok": False, "error": str(e)}
        entry = {
            "t": time.time(),
            "verdict": verdict,
            "verdict_kind": "AUDIT_OPEN" if verdict == "OPEN" else "AUDIT_STOP",
            "reasons": reasons,
            "controls": controls or {},
            "purpose_gate": purpose_gate,
            "note": (
                "AUDIT OPEN ≠ deep-loop ACCEPTED. "
                "Only audit OPEN is production under law."
            ),
        }
        self.s.setdefault("audits", []).append(entry)
        self.s["phase"] = verdict
        if verdict == "STOP":
            self.s.setdefault("residue", []).append(
                {"kind": "audit_stop", "reasons": reasons, "t": time.time()}
            )
        self._graph_advance(verdict, {"reasons": reasons})
        self.save()
        return {"ok": True, "audit": entry, "residue_n": len(self.s["residue"])}

    def claim(self, status: str, text: str, evidence: str = "") -> dict[str, Any]:
        status = status.upper()
        if status not in ("OPEN", "RESIDUE", "CONTESTED", "ACCEPTED"):
            return {"ok": False, "error": "status must be OPEN|ACCEPTED|RESIDUE|CONTESTED"}
        # ACCEPTED = tracking only (deep-loop class); OPEN = audit-backed production
        if status == "ACCEPTED":
            c = {
                "status": "ACCEPTED",
                "status_kind": "WORK_ACCEPTED",
                "text": text,
                "evidence": evidence,
                "t": time.time(),
                "note": "Work item accepted for tracking — NOT production OPEN",
            }
            self.s.setdefault("claims", []).append(c)
            self.save()
            return {"ok": True, "claim": c}
        if status == "OPEN":
            last = (self.s.get("audits") or [{}])[-1]
            if last.get("verdict") != "OPEN":
                return {
                    "ok": False,
                    "error": "cannot claim OPEN (production) without last audit OPEN; use ACCEPTED for work tracking",
                    "last_audit": last,
                }
            try:
                from purpose_gate import gate_claim

                pg = gate_claim("production", measures=self.s.get("measures") or [])
                if not pg.get("ok"):
                    return {"ok": False, "error": pg.get("error"), "purpose_gate": pg}
            except Exception:
                pass
        c = {
            "status": status,
            "status_kind": "AUDIT_OPEN" if status == "OPEN" else status,
            "text": text,
            "evidence": evidence,
            "t": time.time(),
        }
        if status == "OPEN":
            c["note"] = "Production claim under audit OPEN — not deep-loop ACCEPTED"
        self.s.setdefault("claims", []).append(c)
        if status == "RESIDUE":
            self.s.setdefault("residue", []).append(c)
        self.save()
        return {"ok": True, "claim": c}

    def record_projection(self, side: str, domain: str, projection: dict[str, Any]) -> None:
        if side == "human":
            self.s.setdefault("projections", {})["human"] = projection
        else:
            self.s.setdefault("projections", {}).setdefault("domains", {})[domain] = projection
        self.save()

    def record_alignment(self, report: dict[str, Any]) -> None:
        self.s.setdefault("alignments", []).append({"t": time.time(), **report})
        self.save()

    def write_cert(self, out_path: str | None = None) -> dict[str, Any]:
        cert = {
            "cert_id": "pc_" + uuid.uuid4().hex[:12],
            "version": "prime-1.1",
            "session_id": self.s.get("session_id"),
            "workspace": self.s.get("workspace"),
            "intent": self.s.get("intent"),
            "restrict": self.s.get("restrict"),
            "phase": self.s.get("phase"),
            "graph": self.s.get("graph"),
            "projections": self.s.get("projections"),
            "alignments": self.s.get("alignments"),
            "measures": self.s.get("measures"),
            "audits": self.s.get("audits"),
            "claims": self.s.get("claims"),
            "residue": self.s.get("residue"),
            "questions": self.s.get("questions"),
            "verdict": (self.s.get("audits") or [{"verdict": "HALT"}])[-1].get("verdict", "HALT"),
            "law": "restrict→measure→audit→OPEN|STOP; residue never forced",
            "language_thesis": "language is a projection from both sides; glue on interface before OPEN",
            "created_at": time.time(),
        }
        ws = Path(self.s.get("workspace") or ".")
        out = Path(out_path) if out_path else ws / "prime_cert.json"
        if not out.is_absolute():
            out = ws / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cert, indent=2, default=str), encoding="utf-8")
        self._graph_advance("CERT", {"path": str(out)})
        self.save()
        return {"ok": True, "path": str(out), "verdict": cert["verdict"], "cert_id": cert["cert_id"]}
