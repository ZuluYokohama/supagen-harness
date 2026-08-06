"""
Resource plane — utilize CPU / GPU / NPU / RAM *within reason*.

Hardware class on this field kit (observed):
  Snapdragon X Plus · 8 Oryon cores · ~16 GB RAM · Adreno X1-45 GPU · Hexagon NPU
  No CUDA. Torch is CPU build. Heavy ML should ride **LM Studio** (can use GPU/NPU
  drivers) rather than in-process CUDA kernels.

Policy:
  - Prefer parallel CPU for independent measures (smoke, projection, rplc)
  - Cap workers by free RAM and cores
  - Prefer small LM scouts when free RAM < 6 GB; Bonsai only if free ≥ 8 GB and user opts in
  - Never thrash: leave headroom for OS + Grok + LM Studio
"""
from __future__ import annotations

import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class ResourceSnapshot:
    cpu_name: str
    cores: int
    logical: int
    ram_total_gb: float
    ram_free_gb: float
    gpu_names: list[str]
    npu_names: list[str]
    cuda: bool
    profile: str  # snapdragon_x | cuda_workstation | generic
    max_workers: int
    lm_preferred: list[str]
    lm_avoid_unless_free: list[str]
    headroom_gb: float
    notes: list[str]


def _wmic_or_empty(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return (p.stdout or "") + (p.stderr or "")
    except Exception:
        return ""


def snapshot() -> ResourceSnapshot:
    notes: list[str] = []
    cores = os.cpu_count() or 4
    logical = cores

    ram_total = 0.0
    ram_free = 0.0
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        ram_total = vm.total / (1024**3)
        ram_free = vm.available / (1024**3)
        cores = psutil.cpu_count(logical=False) or cores
        logical = psutil.cpu_count(logical=True) or logical
    except Exception:
        notes.append("psutil missing — coarse defaults")
        ram_total = 16.0
        ram_free = 4.0

    cpu_name = platform.processor() or platform.machine()
    try:
        import wmi  # type: ignore

        c = wmi.WMI()
        cpu_name = c.Win32_Processor()[0].Name
    except Exception:
        # PowerShell already showed Snapdragon; keep platform string
        pass

    # GPU / NPU from env-friendly probes
    gpu_names: list[str] = []
    npu_names: list[str] = []
    uname = (cpu_name + " " + platform.platform()).lower()
    if "snapdragon" in uname or "oryon" in uname or "qualcomm" in uname or "arm" in platform.machine().lower():
        profile = "snapdragon_x"
        gpu_names = ["Qualcomm Adreno (expect X1 class)"]
        npu_names = ["Qualcomm Hexagon NPU"]
        notes.append("Snapdragon class: use LM Studio for GPU/NPU; in-process torch is CPU")
    else:
        profile = "generic"

    cuda = False
    try:
        import torch  # type: ignore

        cuda = bool(torch.cuda.is_available())
        if cuda:
            profile = "cuda_workstation"
            gpu_names.append(torch.cuda.get_device_name(0))
            notes.append("CUDA available for in-process tensors if needed")
    except Exception:
        notes.append("torch not cuda-capable or missing")

    # Headroom: keep OS+Grok+LM Studio breathing room
    if ram_total <= 16:
        headroom = 3.5
    elif ram_total <= 32:
        headroom = 4.5
    else:
        headroom = 6.0

    usable = max(ram_free - headroom * 0.35, 0.5)
    # workers: parallel measures are I/O + light CPU
    max_workers = max(1, min(logical - 1, 4, int(usable // 1.5) + 1))
    if ram_free < 3.0:
        max_workers = 1
        notes.append("low free RAM — serial measures only")

    # LM routing (IDs as seen on this kit)
    lm_preferred = ["liquid/lfm2.5-1.2b"]
    lm_avoid = ["prism-ml/bonsai-27b", "ibm/granite-4-h-tiny"]
    notes.append("Default fiber: LFM only + nomic embed; orthogonal roles replace multi-model")
    if ram_free >= 8.0 and ram_total >= 15:
        notes.append("Optional deep models only if explicitly requested / already loaded")

    # env overrides
    if os.environ.get("PRIME_MAX_WORKERS"):
        try:
            max_workers = max(1, min(8, int(os.environ["PRIME_MAX_WORKERS"])))
        except ValueError:
            pass

    return ResourceSnapshot(
        cpu_name=cpu_name,
        cores=int(cores),
        logical=int(logical),
        ram_total_gb=round(ram_total, 2),
        ram_free_gb=round(ram_free, 2),
        gpu_names=gpu_names,
        npu_names=npu_names,
        cuda=cuda,
        profile=profile,
        max_workers=max_workers,
        lm_preferred=lm_preferred,
        lm_avoid_unless_free=lm_avoid,
        headroom_gb=headroom,
        notes=notes,
    )


def pick_lm_model(available: list[str], prefer_deep: bool = False) -> str | None:
    snap = snapshot()
    if not available:
        return None
    if prefer_deep and snap.ram_free_gb >= 8.0:
        for m in ("prism-ml/bonsai-27b",):
            if m in available:
                return m
    for m in snap.lm_preferred:
        if m in available:
            return m
    # if bonsai already loaded and free-ish, allow
    if "prism-ml/bonsai-27b" in available and snap.ram_free_gb >= 5.0:
        return "prism-ml/bonsai-27b"
    return available[0]


def parallel_map(
    jobs: list[tuple[str, Callable[[], Any]]],
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run named callables concurrently within resource budget."""
    snap = snapshot()
    workers = max_workers or snap.max_workers
    workers = max(1, min(workers, len(jobs), snap.max_workers))
    out: dict[str, Any] = {
        "ok": True,
        "workers": workers,
        "resource": asdict(snap),
        "results": {},
        "errors": {},
    }
    if workers == 1:
        for name, fn in jobs:
            try:
                out["results"][name] = fn()
            except Exception as e:
                out["errors"][name] = str(e)
                out["ok"] = False
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn): name for name, fn in jobs}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                out["results"][name] = fut.result()
            except Exception as e:
                out["errors"][name] = str(e)
                out["ok"] = False
    return out


def plan_utilization() -> dict[str, Any]:
    """Human-readable plan for this machine."""
    s = snapshot()
    return {
        "ok": True,
        "snapshot": asdict(s),
        "plan": {
            "CPU": f"{s.max_workers} parallel measure workers on {s.logical} logical cores (leave 1+ for UI)",
            "RAM": f"total {s.ram_total_gb} GB · free {s.ram_free_gb} GB · keep ~{s.headroom_gb} GB headroom class",
            "GPU": (
                "Adreno / iGPU via LM Studio GPU offload when enabled in LMS settings — "
                "not in-process CUDA"
                if s.profile == "snapdragon_x"
                else ("CUDA tensors optional" if s.cuda else "no discrete CUDA")
            ),
            "NPU": (
                "Hexagon NPU: prefer OS/LM Studio / QNN paths if LMS exposes them; "
                "Prime does not pin NPU kernels directly"
                if s.npu_names
                else "no NPU detected"
            ),
            "LM_scouts": f"prefer {s.lm_preferred}; deep {s.lm_avoid_unless_free} only if free RAM high or model already loaded",
            "within_reason": [
                "Do not cold-load 27B + large embeds + heavy browser + Grok on 16 GB",
                "Serial fallback when free RAM < 3 GB",
                "Projection + rplc + smoke are CPU-cheap — parallelize those",
                "Long correctness > saturating all accelerators",
            ],
        },
        "notes": s.notes,
    }
