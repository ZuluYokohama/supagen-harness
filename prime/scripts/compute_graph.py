"""Compute graph for Prime sessions — plot the work before doing it.

Topology-first: nodes are process phases + instruments; edges are allowed
transitions under design law. Meta-meta = graph of how the graph is planned.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


# Canonical process topology (higher domain than flat agent checklists)
LAW_NODES = [
    "META_META",      # plot the compute graph itself
    "META",           # choose instruments / covers
    "RESTRICT",
    "RESEARCH",
    "PLAN",
    "GRAPH",          # materialize / revise compute graph
    "PROJECT",        # bilateral language projections (human ↔ domain)
    "ALIGN",          # glue measure on interface stalks
    "LM_SCOUT",       # local LM Studio probe (optional)
    "CODE",
    "MEASURE",
    "CONDITION",      # sheaf / energy gate
    "AUDIT",
    "OPEN",
    "STOP",
    "ROTATE",
    "GUARDIAN",
    "CLAIM",
    "CERT",
    "HALT",
]

# Allowed directed edges (incomplete → invalid path)
LAW_EDGES = [
    ("META_META", "META"),
    ("META", "RESTRICT"),
    ("RESTRICT", "RESEARCH"),
    ("RESTRICT", "PLAN"),
    ("RESEARCH", "PLAN"),
    ("PLAN", "GRAPH"),
    ("GRAPH", "PROJECT"),
    ("GRAPH", "LM_SCOUT"),
    ("GRAPH", "CODE"),
    ("GRAPH", "MEASURE"),
    ("PROJECT", "ALIGN"),
    ("ALIGN", "MEASURE"),
    ("ALIGN", "PLAN"),
    ("ALIGN", "CODE"),
    ("ALIGN", "STOP"),  # frustrated glue can park early
    ("LM_SCOUT", "PROJECT"),
    ("LM_SCOUT", "PLAN"),
    ("LM_SCOUT", "CODE"),
    ("LM_SCOUT", "MEASURE"),
    ("CODE", "MEASURE"),
    ("CODE", "CONDITION"),
    ("CODE", "PROJECT"),  # re-project after code changes domain language
    ("MEASURE", "CONDITION"),
    ("MEASURE", "AUDIT"),
    ("MEASURE", "ALIGN"),
    ("CONDITION", "AUDIT"),
    ("CONDITION", "GUARDIAN"),
    ("GUARDIAN", "AUDIT"),
    ("GUARDIAN", "ROTATE"),
    ("AUDIT", "OPEN"),
    ("AUDIT", "STOP"),
    ("OPEN", "CLAIM"),
    ("OPEN", "CERT"),
    ("STOP", "ROTATE"),
    ("STOP", "CLAIM"),
    ("ROTATE", "RESTRICT"),
    ("ROTATE", "PLAN"),
    ("ROTATE", "GRAPH"),
    ("ROTATE", "PROJECT"),
    ("CLAIM", "CERT"),
    ("CERT", "HALT"),
    ("OPEN", "HALT"),
    ("STOP", "HALT"),
    # re-entry for long sessions
    ("HALT", "META_META"),
    ("OPEN", "META"),
    ("STOP", "META"),
]


@dataclass
class Node:
    id: str
    kind: str
    label: str
    status: str = "pending"  # pending | active | done | blocked | skipped
    payload: dict[str, Any] = field(default_factory=dict)
    t_ms: float = 0.0


@dataclass
class Edge:
    src: str
    dst: str
    kind: str = "law"
    weight: float = 1.0


@dataclass
class ComputeGraph:
    graph_id: str
    intent: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    current: str | None = None
    meta_note: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "intent": self.intent,
            "current": self.current,
            "path": self.path,
            "meta_note": self.meta_note,
            "created_at": self.created_at,
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges],
            "mermaid": self.to_mermaid(),
            "adjacency": self.adjacency(),
        }

    def adjacency(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = {}
        for e in self.edges:
            adj.setdefault(e.src, []).append(e.dst)
        return adj

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for n in self.nodes.values():
            shape = {
                "pending": f'{n.id}["{n.label}"]',
                "active": f'{n.id}("{n.label} *")',
                "done": f'{n.id}["{n.label} ✓"]',
                "blocked": f'{n.id}{{"{n.label} !"}}',
                "skipped": f'{n.id}["{n.label} ·"]',
            }.get(n.status, f'{n.id}["{n.label}"]')
            lines.append(f"  {shape}")
        for e in self.edges:
            if e.src in self.nodes and e.dst in self.nodes:
                lines.append(f"  {e.src} --> {e.dst}")
        return "\n".join(lines)


def plan_graph(intent: str, modes: list[str] | None = None) -> ComputeGraph:
    """Build an intent-conditioned subgraph of the law topology."""
    modes = modes or []
    gid = "g_" + uuid.uuid4().hex[:10]
    g = ComputeGraph(graph_id=gid, intent=intent)
    g.meta_note = (
        "META_META: compute graph planned before execution. "
        "No OPEN without MEASURE→AUDIT. Residue never forced."
    )

    # Always include spine
    spine = [
        "META_META", "META", "RESTRICT", "PLAN", "GRAPH",
        "PROJECT", "ALIGN",
        "MEASURE", "AUDIT", "OPEN", "STOP", "ROTATE", "CERT", "HALT",
    ]
    if any(m in modes for m in ("lm", "lm_scout", "scout")):
        spine.insert(spine.index("PROJECT"), "LM_SCOUT")
    if any(m in modes for m in ("code", "edit", "implement")):
        spine.insert(spine.index("MEASURE"), "CODE")
        spine.insert(spine.index("MEASURE"), "CONDITION")
        spine.append("GUARDIAN")
    if "research" in modes or "falsif" in intent.lower():
        spine.insert(spine.index("PLAN"), "RESEARCH")
    if any(m in modes for m in ("rplc", "eref", "harness", "claim", "project", "language")):
        spine.insert(spine.index("CERT"), "CLAIM")

    # unique preserve order
    seen = set()
    ordered = []
    for k in spine:
        if k not in seen and k in LAW_NODES:
            seen.add(k)
            ordered.append(k)

    for k in ordered:
        g.nodes[k] = Node(id=k, kind="phase", label=k.replace("_", " "))

    # edges only if both endpoints present and law-allowed
    law_set = set(LAW_EDGES)
    for a, b in LAW_EDGES:
        if a in g.nodes and b in g.nodes and (a, b) in law_set:
            g.edges.append(Edge(src=a, dst=b))

    g.current = "META_META"
    g.nodes["META_META"].status = "active"
    g.path = ["META_META"]
    return g


def advance(g: ComputeGraph, to: str, payload: dict | None = None) -> dict[str, Any]:
    if to not in g.nodes:
        return {"ok": False, "error": f"unknown node {to}", "allowed": list(g.nodes)}
    cur = g.current
    if cur:
        allowed = {e.dst for e in g.edges if e.src == cur}
        # allow stay / reopen META_META for replan
        if to != cur and to not in allowed and to not in ("META_META", "META", "HALT"):
            return {
                "ok": False,
                "error": f"illegal transition {cur} → {to}",
                "allowed": sorted(allowed),
                "law": "plot legal path on graph before force",
            }
        g.nodes[cur].status = "done"
    g.nodes[to].status = "active"
    if payload:
        g.nodes[to].payload.update(payload)
    g.nodes[to].t_ms = time.time() * 1000
    g.current = to
    g.path.append(to)
    return {"ok": True, "current": to, "path": g.path, "graph": g.to_dict()}


def block(g: ComputeGraph, node: str, reason: str) -> dict[str, Any]:
    if node in g.nodes:
        g.nodes[node].status = "blocked"
        g.nodes[node].payload["block_reason"] = reason
    return {"ok": True, "blocked": node, "reason": reason, "graph": g.to_dict()}
