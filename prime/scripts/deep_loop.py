#!/usr/bin/env python3
"""
Deep research → long-horizon verify loop.

Load a report (PDF/text), decompose into work items, tick until all OPEN or STOP-exhausted.
Designed for hours: each tick is bounded; scheduler / --daemon keeps going.

Law: restrict → measure → audit → OPEN|STOP. Residue never forced.
LFM + embed used as measures; Grok/tools do CODE; only verified items OPEN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_ROOT = Path(os.environ.get("PRIME_DEEP_DIR", str(ROOT.parent / "state" / "deep")))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    r = PdfReader(str(path))
    parts = []
    for i, page in enumerate(r.pages):
        t = page.extract_text() or ""
        parts.append(f"--- page {i+1} ---\n{t}")
    return "\n".join(parts)


def load_document(path: Path) -> tuple[str, dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pdf":
        text = extract_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    meta = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "chars": len(text),
        "suffix": path.suffix.lower(),
        "loaded_at": utc_now(),
    }
    return text, meta


def section_claims(text: str) -> list[dict[str, Any]]:
    """Heuristic work items from numbered section headers + fixed doctrine items."""
    headers = []
    for m in re.finditer(r"(?m)^\s*(\d+\.(?:\d+\.)?\s+[A-Za-z][^\n]{8,100})", text):
        h = re.sub(r"\s+", " ", m.group(1)).strip()
        # skip pure reference list noise
        if h.lower().startswith("http") or len(h) < 12:
            continue
        if h not in headers:
            headers.append(h)
    # keep top-level-ish unique
    items = []
    for i, h in enumerate(headers[:16]):
        items.append(
            {
                "id": f"S{i+1:02d}",
                "kind": "section_claim",
                "title": h,
                "status": "pending",  # pending|in_progress|ACCEPTED|STOP|residue
                # ACCEPTED = work item tracking — NOT Prime audit OPEN (production)
                "evidence": [],
                "notes": [],
                "attempts": 0,
            }
        )
    # always add operational synthesis items for this lineage
    extras = [
        (
            "OP01",
            "Map report thesis (GSA / manifold alignment / active state) onto PRIMEdEV Prime runner law",
        ),
        (
            "OP02",
            "Extract falsifiable claims (orders-of-magnitude compute reduction) and mark OPEN only if evidenced in doc",
        ),
        (
            "OP03",
            "Produce operational brief: what to build next in prime/rplc/topology without force-OPEN",
        ),
        (
            "OP04",
            "Cross-link: Liquid AI / LFM, Active Inference, neuromorphic — align with LFM-only orthogonal ops",
        ),
        (
            "OP05",
            "Residue ledger: claims in report that remain CONTESTED or out of scope for this workspace",
        ),
    ]
    for eid, title in extras:
        items.append(
            {
                "id": eid,
                "kind": "operational",
                "title": title,
                "status": "pending",
                "evidence": [],
                "notes": [],
                "attempts": 0,
            }
        )
    return items


@dataclass
class DeepJob:
    job_id: str
    source_path: str
    source_sha256: str
    goal: str
    status: str  # running|done|stopped|failed
    created_at: str
    updated_at: str
    text_path: str
    items: list[dict[str, Any]] = field(default_factory=list)
    ticks: int = 0
    max_ticks: int = 500
    max_hours: float = 12.0
    opened: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    residue: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    final_brief_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DeepJob":
        return DeepJob(**{k: d[k] for k in DeepJob.__dataclass_fields__ if k in d})


class DeepLoop:
    def __init__(self, job_dir: Path | None = None):
        self.job_dir = job_dir or STATE_ROOT
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.job_path = self.job_dir / "job.json"
        self.job: DeepJob | None = None

    def save(self) -> None:
        if not self.job:
            return
        self.job.updated_at = utc_now()
        self.job_path.write_text(json.dumps(self.job.to_dict(), indent=2), encoding="utf-8")
        # also symlink-ish current
        (self.job_dir / "CURRENT.md").write_text(self.status_md(), encoding="utf-8")

    def load(self) -> DeepJob | None:
        if not self.job_path.exists():
            return None
        raw = json.loads(self.job_path.read_text(encoding="utf-8"))
        # Migrate legacy status OPEN → ACCEPTED (work tracking ≠ audit OPEN)
        for it in raw.get("items") or []:
            if it.get("status") == "OPEN":
                it["status"] = "ACCEPTED"
                it["status_kind"] = "WORK_ACCEPTED"
                it.setdefault("notes", []).append(
                    "migrated OPEN→ACCEPTED: not production/audit OPEN"
                )
        self.job = DeepJob.from_dict(raw)
        # persist migration so ledgers stop saying OPEN for work items
        try:
            self.save()
        except Exception:
            pass
        return self.job

    def ingest(
        self,
        path: str,
        goal: str = "",
        max_hours: float = 12.0,
        max_ticks: int = 500,
    ) -> dict[str, Any]:
        p = Path(path)
        text, meta = load_document(p)
        text_path = self.job_dir / "source.txt"
        text_path.write_text(text, encoding="utf-8")
        items = section_claims(text)
        # Dimensional parse: chunk + embed index for retrieval (Grok/LFM handoff)
        index_path = self.job_dir / "dimensional_index.json"
        index_meta: dict[str, Any] = {}
        try:
            from dimensional_parse import build_index, save_index

            print("building dimensional index (embed chunks)...", flush=True)
            index = build_index(text, embed=True, max_chunks=64)
            save_index(index, index_path)
            index_meta = {
                "path": str(index_path),
                "n_chunks": index.get("n_chunks"),
                "embedded": index.get("embedded"),
                "dim": index.get("dim"),
            }
            print(f"index ready: {index_meta}", flush=True)
        except Exception as e:
            index_meta = {"error": str(e), "path": str(index_path)}
        jid = "deep_" + uuid.uuid4().hex[:10]
        self.job = DeepJob(
            job_id=jid,
            source_path=meta["path"],
            source_sha256=meta["sha256"],
            goal=goal
            or (
                "Ingest deep research report; verify claims operationally against PRIMEdEV; "
                "loop until confirmed OPEN or honest STOP/residue. Never force-OPEN."
            ),
            status="running",
            created_at=utc_now(),
            updated_at=utc_now(),
            text_path=str(text_path),
            items=items,
            max_hours=max_hours,
            max_ticks=max_ticks,
        )
        # stash index path on job via history seed
        self.job.history.append({"t": utc_now(), "event": "dimensional_index", **index_meta})
        self.save()
        (self.job_dir / "index_meta.json").write_text(
            json.dumps(index_meta, indent=2), encoding="utf-8"
        )
        # seed prime session
        try:
            os.environ["PRIME_STATE_DIR"] = str(self.job_dir / "session")
            import mcp_server

            m = mcp_server
            from session_store import SessionStore

            m.STORE = SessionStore(self.job_dir / "session")
            m.tool_meta_loop(self.job.goal)
            m.tool_restrict(
                goal=self.job.goal,
                non_goals="force-OPEN; accept report claims without evidence; multi-model thrash",
                success="all work items OPEN or explicit STOP/residue; final brief written",
                constraints="LFM measure only; dimensional retrieval packs; hours ok; residue never forced",
            )
        except Exception as e:
            pass
        return {
            "ok": True,
            "job_id": jid,
            "n_items": len(items),
            "source": meta,
            "dimensional_index": index_meta,
            "items": [{"id": i["id"], "title": i["title"]} for i in items],
        }

    def status_md(self) -> str:
        j = self.job
        if not j:
            return "No deep job loaded."
        pending = sum(1 for i in j.items if i["status"] == "pending")
        # ACCEPTED (legacy OPEN) = work tracking, not production
        accepted = sum(1 for i in j.items if i["status"] in ("ACCEPTED", "OPEN"))
        stopped = sum(1 for i in j.items if i["status"] == "STOP")
        res = sum(1 for i in j.items if i["status"] == "residue")
        lines = [
            f"# Deep job `{j.job_id}`",
            f"- status: **{j.status}**",
            f"- ticks: {j.ticks}/{j.max_ticks}",
            f"- source: `{j.source_path}`",
            f"- goal: {j.goal[:200]}",
            f"- progress: ACCEPTED={accepted} STOP={stopped} residue={res} pending={pending}",
            f"- **note:** ACCEPTED ≠ audit OPEN (production). See status_kind.",
            "",
            "## Items",
        ]
        for i in j.items:
            lines.append(f"- [{i['status']}] `{i['id']}` {i['title'][:100]}")
        if j.history:
            lines.append("\n## Last ticks")
            for h in j.history[-5:]:
                lines.append(f"- t={h.get('t')} item={h.get('item_id')} → {h.get('result')}")
        return "\n".join(lines)

    def _deadline_ok(self) -> bool:
        assert self.job
        created = datetime.fromisoformat(self.job.created_at)
        hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600.0
        if hours > self.job.max_hours:
            return False
        if self.job.ticks >= self.job.max_ticks:
            return False
        return True

    def _next_item(self) -> dict[str, Any] | None:
        assert self.job
        for i in self.job.items:
            if i["status"] in ("pending", "in_progress"):
                return i
        return None

    def _lfm_verify_item(self, item: dict[str, Any], excerpt: str) -> dict[str, Any]:
        from lfm_ops import lfm_role_pass

        # Prefer dimensional retrieval pack (vectors → top-k language stalks)
        pack = excerpt
        index_path = self.job_dir / "dimensional_index.json"
        retrieval_meta: dict[str, Any] = {}
        if index_path.is_file():
            try:
                from dimensional_parse import load_index, pack_for_grok, pack_for_lfm, retrieve

                index = load_index(index_path)
                query = f"{item['title']}\n{item.get('kind','')}\n{self.job.goal if self.job else ''}"
                hits = retrieve(index, query, k=4)
                pack = pack_for_lfm(query, hits)
                retrieval_meta = pack_for_grok(index, hits, query)
                (self.job_dir / "last_retrieval.json").write_text(
                    json.dumps(retrieval_meta, indent=2), encoding="utf-8"
                )
            except Exception as e:
                pack = excerpt[:3500]
                retrieval_meta = {"error": str(e), "fallback": "raw_excerpt"}

        prompt = (
            f"DEEP RESEARCH WORK ITEM\n"
            f"ID: {item['id']}\n"
            f"TITLE: {item['title']}\n"
            f"KIND: {item['kind']}\n\n"
            f"{pack}\n\n"
            f"Decide if this item can be OPEN as operationally verified for PRIMEdEV "
            f"(geometry/GSA/active-state lineage, compute reduction claims only if evidenced in pack). "
            f"If not fully verified, STOP or NEED_INFO. Residue never forced."
        )
        card = lfm_role_pass(prompt, embed=True, max_tokens=140)
        if isinstance(card, dict):
            card["retrieval"] = {
                "n_hits": len(retrieval_meta.get("retrieval") or []),
                "top_scores": [h.get("score") for h in (retrieval_meta.get("retrieval") or [])[:4]],
                "method": (retrieval_meta.get("retrieval") or [{}])[0].get("method")
                if retrieval_meta.get("retrieval")
                else None,
            }
        return card

    def tick(self) -> dict[str, Any]:
        if not self.job:
            self.load()
        if not self.job:
            return {"ok": False, "error": "no job — ingest first"}
        if self.job.status != "running":
            return {"ok": True, "done": True, "status": self.job.status, "md": self.status_md()}

        if not self._deadline_ok():
            self.job.status = "stopped"
            self.job.residue.append("deadline_or_max_ticks")
            self._write_final_brief()
            self.save()
            return {"ok": True, "done": True, "status": "stopped", "reason": "deadline"}

        item = self._next_item()
        if not item:
            self.job.status = "done"
            self._write_final_brief()
            self.save()
            return {"ok": True, "done": True, "status": "done", "md": self.status_md()}

        self.job.ticks += 1
        item["status"] = "in_progress"
        item["attempts"] += 1
        text = Path(self.job.text_path).read_text(encoding="utf-8", errors="replace")
        # excerpt around title keywords
        key = item["title"][:40]
        idx = text.find(key[:20]) if len(key) > 20 else -1
        if idx < 0:
            excerpt = text[:4000]
        else:
            excerpt = text[max(0, idx - 200) : idx + 3000]

        try:
            card = self._lfm_verify_item(item, excerpt)
        except Exception as e:
            card = {"ok": False, "error": str(e), "verdict": "NEED_INFO"}

        verdict = (card.get("verdict") or "NEED_INFO").upper()
        fatal = bool(card.get("fatal_flag"))

        def _role_text(role: str) -> str:
            """Layered ops use parsed/raw; legacy free-text uses content."""
            o = (card.get("outputs") or {}).get(role) or {}
            if not isinstance(o, dict):
                return str(o or "")
            if o.get("parsed") is not None:
                try:
                    return json.dumps(o["parsed"], ensure_ascii=False)
                except Exception:
                    return str(o["parsed"])
            return str(o.get("content") or o.get("raw") or "")

        emb = card.get("embeddings") or {}
        cos = emb.get("mean_cosine") if isinstance(emb, dict) else None
        if cos is None:
            cos = card.get("mean_cosine") or 0.0
        try:
            cos = float(cos or 0.0)
        except (TypeError, ValueError):
            cos = 0.0

        # operational mapping
        if fatal or verdict == "STOP":
            if item["attempts"] >= 3:
                item["status"] = "residue"
                item["notes"].append("max attempts — residue")
                self.job.residue.append(item["id"])
                result = "residue"
            else:
                item["status"] = "pending"
                item["notes"].append(f"STOP attempt {item['attempts']}")
                result = "retry"
        elif verdict in ("OPEN", "OPEN_CANDIDATE"):
            # OPEN only operational items lightly; section claims need evidence note
            if item["kind"] == "section_claim":
                scout = _role_text("SCOUT")
                scout_ok = len(scout) > 40 or (
                    isinstance((card.get("outputs") or {}).get("SCOUT"), dict)
                    and bool(((card.get("outputs") or {}).get("SCOUT") or {}).get("parsed"))
                )
                # Job 2: NLI agreement — NOT nomic cosine aboutness
                agree = card.get("agreement") if isinstance(card.get("agreement"), dict) else {}
                nli_label = str(agree.get("label") or "")
                nli_agrees = bool(agree.get("agrees"))
                if nli_label == "contradiction":
                    glue_ok = False
                    glue_reason = f"NLI contradiction conf={agree.get('confidence')}"
                elif nli_agrees and scout_ok:
                    glue_ok = True
                    glue_reason = f"NLI entailment conf={agree.get('confidence')} aboutness={cos:.3f}"
                elif not agree:
                    # NLI missing: refuse OPEN on aboutness alone
                    glue_ok = False
                    glue_reason = f"NLI missing; aboutness={cos:.3f} (not agreement)"
                else:
                    glue_ok = False
                    glue_reason = (
                        f"NLI={nli_label or 'missing'} agrees={nli_agrees} "
                        f"aboutness={cos:.3f} scout_ok={scout_ok}"
                    )
                if glue_ok:
                    # WORK tracking only — never call this production OPEN
                    item["status"] = "ACCEPTED"
                    item["status_kind"] = "WORK_ACCEPTED"
                    item["evidence"].append(
                        {
                            "lfm": True,
                            "aboutness_cos": cos,
                            "nli": agree if isinstance(agree, dict) else {},
                            "glue_reason": glue_reason,
                            "t": utc_now(),
                            "not_production_open": True,
                        }
                    )
                    self.job.opened.append(item["id"])  # legacy field name: accepted ids
                    item["notes"].append(
                        "ACCEPTED for tracking — NOT audit/production OPEN"
                    )
                    result = "ACCEPTED"
                elif item["attempts"] >= 3:
                    item["status"] = "STOP"
                    self.job.stopped.append(item["id"])
                    item["notes"].append(f"glue exhausted: {glue_reason}")
                    result = "STOP"
                else:
                    item["status"] = "pending"
                    item["notes"].append(f"weak glue — retry: {glue_reason}")
                    result = "retry"
            else:
                # operational: ACCEPTED with brief note from VERDICT (not production OPEN)
                item["status"] = "ACCEPTED"
                item["status_kind"] = "WORK_ACCEPTED"
                item["evidence"].append(
                    {
                        "lfm_verdict": verdict,
                        "aboutness_cos": cos,
                        "nli": agree if isinstance(agree, dict) else {},
                        "t": utc_now(),
                        "not_production_open": True,
                    }
                )
                item["notes"].append(_role_text("VERDICT")[:500])
                item["notes"].append(
                    "ACCEPTED for tracking — NOT audit/production OPEN; "
                    "OP02-class magnitude claims stay contested until EXISTENCE measures"
                )
                self.job.opened.append(item["id"])
                result = "ACCEPTED"
        else:
            # NEED_INFO
            if item["attempts"] >= 2:
                item["status"] = "STOP"
                self.job.stopped.append(item["id"])
                item["notes"].append("NEED_INFO exhausted")
                result = "STOP"
            else:
                item["status"] = "pending"
                result = "retry"

        self.job.history.append(
            {
                "t": utc_now(),
                "tick": self.job.ticks,
                "item_id": item["id"],
                "result": result,
                "verdict": verdict,
                "fatal": fatal,
            }
        )
        self.save()

        # also write last exchange style card
        (self.job_dir / "last_tick.md").write_text(
            f"# Tick {self.job.ticks} · {item['id']} → {result}\n\n"
            f"title: {item['title']}\n"
            f"lfm_verdict: {verdict} fatal={fatal}\n\n"
            f"{json.dumps({k: (v.get('content') if isinstance(v, dict) else v) for k,v in (card.get('outputs') or {}).items()}, indent=2)[:3000]}\n",
            encoding="utf-8",
        )

        done = self._next_item() is None
        if done:
            self.job.status = "done"
            self._write_final_brief()
            self.save()

        return {
            "ok": True,
            "done": done,
            "tick": self.job.ticks,
            "item_id": item["id"],
            "result": result,
            "status": self.job.status,
            "progress": self._progress(),
            "md": self.status_md(),
        }

    def _progress(self) -> dict[str, int]:
        assert self.job
        c: dict[str, int] = {}
        for i in self.job.items:
            c[i["status"]] = c.get(i["status"], 0) + 1
        return c

    def _write_final_brief(self) -> None:
        assert self.job
        path = self.job_dir / "FINAL_BRIEF.md"
        lines = [
            f"# Final brief — {self.job.job_id}",
            f"status: **{self.job.status}**",
            f"source: `{self.job.source_path}`",
            f"sha256: `{self.job.source_sha256}`",
            f"ticks: {self.job.ticks}",
            "",
            "## Goal",
            self.job.goal,
            "",
            "## ACCEPTED items (work tracking — NOT production OPEN)",
            "These survived deep-loop LFM/NLI gates for *tracking*. "
            "Production OPEN requires Prime audit + purpose-matched measures.",
            "",
        ]
        for i in self.job.items:
            if i["status"] in ("ACCEPTED", "OPEN"):
                lines.append(f"- `{i['id']}` {i['title']}")
                if i.get("notes"):
                    lines.append(f"  - note: {i['notes'][-1][:300]}")
        lines.append("\n## STOP items")
        for i in self.job.items:
            if i["status"] == "STOP":
                lines.append(f"- `{i['id']}` {i['title']}")
        lines.append("\n## Residue")
        for i in self.job.items:
            if i["status"] == "residue":
                lines.append(f"- `{i['id']}` {i['title']}")
        lines.extend(
            [
                "",
                "## Law",
                "Residue never forced. Report claims are not production OPEN without workspace measures.",
                "This brief is a deep-loop certificate face — re-verify before shipping code claims.",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        self.job.final_brief_path = str(path)

    def run_until_done(self, sleep_s: float = 0.5, report_every: int = 1) -> dict[str, Any]:
        """Foreground long loop (can be hours if max_hours set)."""
        outs = []
        while True:
            r = self.tick()
            outs.append(r)
            if report_every and self.job and self.job.ticks % report_every == 0:
                print(r.get("md", "")[:500], flush=True)
                print(f"--- tick result: {r.get('item_id')} → {r.get('result')} ---", flush=True)
            if r.get("done") or not r.get("ok"):
                break
            if sleep_s > 0:
                time.sleep(sleep_s)
        return {
            "ok": True,
            "final_status": self.job.status if self.job else "unknown",
            "ticks": self.job.ticks if self.job else 0,
            "brief": self.job.final_brief_path if self.job else None,
            "last": outs[-1] if outs else None,
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prime deep research long loop")
    ap.add_argument("command", choices=["ingest", "tick", "status", "run", "daemon"])
    ap.add_argument("--path", default="", help="PDF/text path for ingest")
    ap.add_argument("--goal", default="")
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--max-ticks", type=int, default=500)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--job-dir", default="", help="override deep job dir")
    args = ap.parse_args(argv)

    job_dir = Path(args.job_dir) if args.job_dir else STATE_ROOT
    loop = DeepLoop(job_dir)

    if args.command == "ingest":
        if not args.path:
            print("need --path", file=sys.stderr)
            return 2
        r = loop.ingest(args.path, goal=args.goal, max_hours=args.max_hours, max_ticks=args.max_ticks)
        print(json.dumps(r, indent=2))
        print(loop.status_md())
        return 0

    if args.command == "status":
        loop.load()
        print(loop.status_md())
        return 0

    if args.command == "tick":
        loop.load()
        r = loop.tick()
        print(json.dumps({k: v for k, v in r.items() if k != "md"}, indent=2, default=str))
        print(r.get("md", ""))
        return 0 if r.get("ok") else 1

    if args.command in ("run", "daemon"):
        if args.path:
            loop.ingest(args.path, goal=args.goal, max_hours=args.max_hours, max_ticks=args.max_ticks)
        else:
            loop.load()
        r = loop.run_until_done(sleep_s=args.sleep)
        print(json.dumps(r, indent=2, default=str))
        return 0 if r.get("ok") else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
