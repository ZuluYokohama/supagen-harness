# V&V Run Results — Full Matrix

**Verdict:** `GO_MEASURE`  
**Seconds:** 28.7  
**Pass/Fail:** 18 pass / 0 fail (0 critical fail)

GO_MEASURE = instruments+law green for measured advertise of dual metric. Not production OPEN authority. Hexagon residual WARN allowed.

**Law:** aboutness must not promote OPEN; NLI owns agreement; residue never forced

**Architecture:** hybrid LMS chat + off-LMS jina/DeBERTa/rerank

## Cells

| ID | Status | Critical |
|----|--------|----------|
| `D0_architecture` | **PASS** | yes |
| `D1_aboutness_jina` | **PASS** | yes |
| `D2_neural_rerank` | **PASS** | yes |
| `D3_deberta_nli` | **PASS** | yes |
| `D4_fiber_modes` | **PASS** | yes |
| `D5_cert_face_law` | **PASS** | yes |
| `D6_identity_floors` | **PASS** | yes |
| `D7_supagen_contract` | **PASS** | no |
| `D8_accel_npu` | **PASS** | no |
| `D9_adv_lexical` | **PASS** | no |
| `D10_truth_plane_scout` | **PASS** | yes |
| `D11_field_certify` | **PASS** | yes |
| `D12_ort_nli` | **PASS** | no |
| `D13_preserve_pick` | **PASS** | yes |
| `D14_cos_never_open_live` | **PASS** | yes |
| `D15_jina_small_bakeoff` | **PASS** | yes |
| `D16_kb_family_1024` | **PASS** | yes |
| `D17_push_suite` | **PASS** | no |

## Detail

### D0_architecture — PASS

```json
{
  "fiber_mode_default": "scout",
  "frankenstein": {
    "loaded": false,
    "required": false
  },
  "law": "aboutness must not promote OPEN; NLI owns agreement; residue never forced; production OPEN needs domain audit",
  "job1_lms": false,
  "job2_owns_open": true,
  "accel": {
    "preference": "auto",
    "ort": {
      "version": "1.24.4",
      "providers_builtin": [
        "DmlExecutionProvider",
        "CPUExecutionProvider"
      ],
      "dml": true
    },
    "torch": {
      "version": "2.12.0+cpu",
      "cuda": false,
      "threads": 8,
      "threads_set": 4
    },
    "hexagon_npu": {
      "ok": true,
      "n_qnn_devices": 3,
      "qnn_ver": "2.4.0",
      "registered": true,
      "note": "run npu-htp-2026-08-06; Job2 NLI parity residual"
    }
  }
}
```

### D1_aboutness_jina — PASS

```json
{
  "family": "jina",
  "floor_mean": 0.0852,
  "range": 0.6499,
  "negation_cos": 0.6888,
  "adversarial_cos": 0.6392,
  "pass_rule": "family=jina, floor<0.35, range>0.40",
  "cos_never_open": true
}
```

### D2_neural_rerank — PASS

```json
{
  "model": "jinaai/jina-reranker-v3",
  "prefer_benign_rate": 1.0,
  "mean_gap": 0.2964,
  "source": "C:\\PRIMEdEV-1\\prime\\state\\tier_b_challenger.json"
}
```

### D3_deberta_nli — PASS

```json
{
  "model": "cross-encoder/nli-deberta-v3-base",
  "negation_contra": 1.0,
  "adv_contra": 1.0,
  "adv_block_open": 1.0,
  "para_agree": 0.5
}
```

### D4_fiber_modes — PASS

```json
{
  "default_mode": "scout",
  "scout_pick": {
    "key": "liquid/lfm2.5-1.2b",
    "reason": "already_loaded_max_ctx",
    "loaded_ctx": 8192
  },
  "preserve_pick": {
    "key": "frankenstein-2.0-i1",
    "reason": "preserve_catalog",
    "loaded": false
  },
  "frankenstein_required_scout": false,
  "frankenstein_required_preserve": true,
  "heavy_keys": [
    "gemma-4-12b",
    "frankenstein",
    "bonsai",
    "granite-4-h-tiny",
    "queen-opus",
    "ibm/granite",
    "prism-ml/bonsai"
  ]
}
```

### D5_cert_face_law — PASS

```json
{
  "high_cos_no_nli": "NEED_INFO",
  "contradiction": "STOP",
  "entail_ok": "OPEN_CANDIDATE",
  "law": "cos never promotes; NLI contradiction STOP; OPEN only CANDIDATE"
}
```

### D6_identity_floors — PASS

```json
{
  "lfm": {
    "present": true,
    "path": "C:\\PRIMEdEV-1\\prime\\state\\holonomy_v3_lfm12b_identity_floor.json",
    "identity_p": 0.2857,
    "gate": "FAIL",
    "gate_failed": true,
    "depths": null,
    "model": "liquid/lfm2.5-1.2b"
  },
  "frankenstein": {
    "present": true,
    "path": "C:\\PRIMEdEV-1\\prime\\state\\holonomy_v3_frankenstein_identity_chain.json",
    "identity_p": 0.875,
    "gate": "PASS",
    "gate_failed": null,
    "depths": {
      "1": 0.875,
      "2": 0.875,
      "3": 0.875,
      "4": 0.875
    },
    "model": "frankenstein-2.0-i1"
  },
  "gemma": {
    "present": true,
    "path": "C:\\PRIMEdEV-1\\prime\\state\\holonomy_v3_gemma12b_floor.json",
    "identity_p": 0.375,
    "gate": "FAIL",
    "gate_failed": true,
    "depths": null,
    "model": "gemma-4-12b-agentic-fable5-composer2.5-v2-3.5x-tau2@q4_k_m"
  },
  "notes": [
    "LFM p=0.2857 FAIL as expected (scout only, not holonomy subject)",
    "frankenstein min-depth p=0.875 PASS (preserve fiber)",
    "gemma p=0.375 FAIL (capacity\u2260preserve)"
  ]
}
```

### D7_supagen_contract — PASS

```json
{
  "rc": 0,
  "stdout_tail": "contract ok=True pass=7 fail=0 live=False\n  [PASS] family_jina_default  jina\n  [PASS] prefix_query  \n  [PASS] ctx_lfm_rich  \n  [PASS] ctx_no_ui_default_as_max  \n  [PASS] fence_json  \n  [PASS] token_cap  387\n  [PASS] pick_chat_model  {\"key\": \"liquid/lfm2.5-1.2b\", \"reason\": \"already_loaded_max_ctx\", \"loaded_ctx\": 8192}\n",
  "stderr_tail": ""
}
```

### D8_accel_npu — PASS (path) / residual (Job2 parity)

```json
{
  "preference": "auto",
  "ort": {
    "version": "1.24.4",
    "providers_builtin": [
      "DmlExecutionProvider",
      "CPUExecutionProvider"
    ],
    "dml": true
  },
  "torch": {
    "version": "2.12.0+cpu",
    "cuda": false,
    "threads": 4,
    "threads_set": 4
  },
  "hexagon_npu": {
    "ok": true,
    "n_qnn_devices": 3,
    "qnn_ver": "2.4.0",
    "registered": true,
    "run_id": "npu-htp-2026-08-06"
  },
  "residual": "Job2 DeBERTa QDQ on HTP: label parity FAIL — CPU ORT/CE remains OPEN authority",
  "next": "E3: UINT16 act / better calib / distill stance head before product HTP NLI"
}
```

### D9_adv_lexical — PASS

```json
{
  "pearson_cos_jaccard": 0.8458,
  "reading": "adv cos failures track lexical overlap \u2014 not arbitrary"
}
```

### D10_truth_plane_scout — PASS

```json
{
  "ok": true,
  "fiber_mode": "scout",
  "frankenstein": {
    "loaded": false,
    "required": false
  },
  "jina": {
    "ok": true,
    "status": "already_running",
    "base": "http://127.0.0.1:8765",
    "dim": 1024,
    "started": false
  },
  "nli": null,
  "rerank": null,
  "seconds": 5.33
}
```

### D11_field_certify — PASS

```json
{
  "rc": 0,
  "tail": "          LIVE  refs=1\n  - tool_telem               LIVE  refs=1\n  - tool_voltages            LIVE  refs=1\n  - wits_surface             LIVE  refs=6\n  - decoder_rt               LIVE  refs=6\n  - acq_db                   LIVE  refs=2\n  - acq_emz_package          LIVE  refs=1\nexit=0\n\n=== local_scout (optional online) ===\nSkip in offline smoke. When LM Studio server is up:\n  C:\\Python314\\python.exe C:\\PRIMEdEV-1\\harness\\local_mode\\lfm_scout_v1\\smoke_local.py\n\nHARNESS OFFLINE SMOKE OK\nHARNESS=C:\\PRIMEdEV-1\\harness\nROOT=C:\\PRIMEdEV-1\nTree: golden Q + external certify gate live. Scout waits on LMS.\n",
  "law": "external certifier; DRAFT\u2192STOP; multiplane OPEN when covered"
}
```

### D12_ort_nli — PASS

```json
{
  "session": {
    "providers": [
      "CPUExecutionProvider"
    ],
    "active": "CPUExecutionProvider",
    "warning": null
  },
  "ort_hits": 3,
  "torch_hits": 3,
  "label_parity": true,
  "ort_s": 0.773,
  "torch_s": 4.125,
  "rows": [
    {
      "expect": "contradiction",
      "label": "contradiction",
      "conf": 0.9999,
      "hit": true,
      "ms": 198.0,
      "provider": "CPUExecutionProvider"
    },
    {
      "expect": "contradiction",
      "label": "contradiction",
      "conf": 0.9999,
      "hit": true,
      "ms": 205.9,
      "provider": "CPUExecutionProvider"
    },
    {
      "expect": "entailment",
      "label": "entailment",
      "conf": 0.9985,
      "hit": true,
      "ms": 193.5,
      "provider": "CPUExecutionProvider"
    },
    {
      "expect": "neutral",
      "label": "contradiction",
      "conf": 0.9773,
      "hit": false,
      "ms": 174.9,
      "provider": "CPUExecutionProvider"
    }
  ]
}
```

### D13_preserve_pick — PASS

```json
{
  "pick": {
    "key": "frankenstein-2.0-i1",
    "reason": "preserve_catalog",
    "loaded": false
  },
  "note": "load only when PRIME_FIBER_MODE=preserve explicitly"
}
```

### D14_cos_never_open_live — PASS

```json
{
  "cosine": 0.5332,
  "family": "jina",
  "nli_label": "contradiction",
  "nli_engine": "ort_nli",
  "face": "STOP"
}
```

### D15_jina_small_bakeoff — PASS

```json
{
  "floor_mean": 0.0852,
  "range": 0.6499,
  "negation": 0.6888,
  "adv_mean": 0.6392,
  "worst_adv": 0.7752,
  "prefix": "Document: Aboutness must not promote OPE",
  "dim": 1024,
  "model": "jina-embeddings-v5-text-small-retrieval"
}
```

### D16_kb_family_1024 — PASS

```json
{
  "path": "C:\\PRIMEdEV-1\\prime\\state\\kb\\manifold_index.json",
  "embed_family": "jina",
  "dim": 1024,
  "live_dim": 1024,
  "n_chunks": 53,
  "embedded": 53
}
```

### D17_push_suite — PASS

```json
{
  "go_no_go": "GO_MEASURE",
  "n_pass": 5,
  "n_fail": 1,
  "cells": [
    {
      "id": "P1_jina_v5_small",
      "status": "PASS"
    },
    {
      "id": "P6_hexagon_qnn",
      "status": "WARN"
    },
    {
      "id": "P5_ort_hot",
      "status": "PASS"
    },
    {
      "id": "P4_negative_force_open",
      "status": "PASS"
    },
    {
      "id": "P2_preserve_smoke",
      "status": "PASS"
    },
    {
      "id": "P3_truth_plane_enter",
      "status": "PASS"
    }
  ]
}
```

## Sign-off rule

- Critical FAIL → **NO-GO** advertise
- WARN (NPU, package path) → residual documented, not force-OPEN
- Production OPEN still requires domain audit + external certifier

Artifact JSON: `prime/state/vv_full_matrix.json`
