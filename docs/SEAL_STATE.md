# Seal state after the 2026-08-11 review pass

## What changed

The sealed protocol records under `research/results/` carry two kinds of hash:

1. **Result-artifact digests** — the `.sha256` sidecars and the
   `incumbent_pretruth_inventory` entries, covering shards, manifests and
   scored CSVs.
2. **Source digests** — `source_sha256`, covering 14 `.py` files that produced
   those results.

A code review on 2026-08-11 applied 29 findings to the research tree. Several
landed in files covered by (2): `interval_gate.py`, `ordered_transport.py`,
`repeated_group_gate.py`, `spatial_score_gate.py`, `structural_field.py`,
`structural_field_gate.py`, and the v1 structural-field tests.

**The `source_sha256` blocks in the committed protocol records therefore no
longer match the working tree, by design.** They describe the pre-review source
state — the code that actually produced the recorded results — and are retained
as history rather than rewritten to match edited files. Rewriting them would
assert that the current code produced those results, which it did not.

The result-artifact digests in (1) are **unaffected and still verify.** Nothing
in the review pass touched a result artifact.

## What still verifies

| Check | Status |
|---|---|
| `.sha256` sidecars against their artifacts (29) | all verify |
| `incumbent_pretruth_inventory` (53 entries) | all verify |
| `source_sha256` (14 entries) | 5 verify, **9 historical** |
| In-code pins (`CORE_SHA256`, `V1_SOURCE_SHA256`, …) | re-pinned to post-review bytes |

The in-code pins were re-pinned rather than demoted because they are runtime
drift guards: they exist to catch *unintended* edits between runs, so leaving
them stale would disable the guard for every future change.

Byte-exactness is preserved on checkout by `.gitattributes` (`research/** -text`
and `geosteern/*.py -text`). Without it, `core.autocrlf` rewrites line endings
on a fresh Windows clone and every digest above fails.

## Why the sealed artifacts themselves were not edited

The review also flagged machine-local absolute paths frozen inside three
protocol records (`roots`, `data.data_dir`, `spatial_manifests[*].path`, and the
fold-artifact inventory). Those were **not** rewritten.

Editing any of those JSON files changes its bytes, which breaks its `.sha256`
sidecar, which breaks the `protocol.byte_sha256` that
`anchored_structural_field_v1_STOP.json` binds, which breaks the v2 gate's
parent-lineage check. The paths are provenance — a record of where the run
happened — and are not used to resolve anything at read time.

The fix was applied to the **code that emits them** instead, so future artifacts
carry protocol-relative paths. This matches the review's own guidance on
`equal_ordered_joint_spatial_protocol.json`: keep the frozen paths, do not touch
the hash artifacts, and store relative paths in artifacts generated from now on.

## Consequence

The existing results cannot be re-derived from the current tree and verified
end to end in one pass. Restoring a fully verifiable chain requires re-running
`RUN` / `AGGREGATE` / `SCORE` under the edited sources and re-sealing.

Those stages are currently **closed**: `anchored_structural_field_v1_STOP.json`
records that the v1 benchmark hit its 3600 s watchdog (exit 124) before its
atomic commit, so no truth scoring has been performed. The v2 gate passes its
tests but has never been run.

Until that happens, treat the recorded results as evidence about the
pre-review code, and this document as the record of the gap.

## Pre-review source digests

Recoverable from git history — the state immediately before the review pass:

```
git show 7616edd:research/results/anchored_structural_field_protocol.json
```

That commit is the last one in which `source_sha256` matched the working tree.
