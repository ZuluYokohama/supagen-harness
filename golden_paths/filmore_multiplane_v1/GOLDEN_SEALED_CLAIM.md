# Golden Path — Filmore Multi-Plane Sealed Claim (v1)

**Status:** OPEN instance of the Lineage OPEN set (executable Q).  
**Well:** Mount Filmore 3974-24-25-36-1NH (TD26324)  
**Tool:** MicroPulse SN 1989  
**Compiled claims:** 2026-08-03 · Release chain sealed 2026-08-03T10:39:27Z  
**Value function \(v\):** multi-plane operational truth (NPT narrative integrity; tool vs surface vs decode) — not fleet ML score.

This is **P3** on the decision tree: one continuous move that *instances* the operator without boiling the ocean.

---

## 1. Term series on this run

| Stage | Series | Instance here |
|-------|--------|----------------|
| **X** | Substrate / condition:states | MicroPulse dump CSVs (system Mag-PI, survey, shock, continuous…) + Erdos `.emz` (WITS, decoder sessions, DBs) |
| **f(x)** | Dynamics constrain order | Depth trajectory, park/RIH phases, session timeline |
| **M** | Latent order | Multi-plane timeline at ~415 ft RIH burst window |
| **v** | Value warp | Prefer claims that bind planes over single-strip stories |
| **M_v** | Operational geometry | “What can we certify about Mag-PI vs decoder vs surface?” |
| **H_k / relations** | Robust structure / deviation | LIVE/DEAD matrix; 50 Mag-PI; 0 decoder sessions in burst; survey between ticks |
| **Q** | Opened meaning | Eight HIGH claims in `claims.json` (especially `MAGPI_DOWNHOLE_ONLY_NO_RT_DECODE`) |
| **Agency / continuity** | Don’t force residue | Single-plane “tool fine” or “surface saw it” would be discontinuous — **STOP** those stories |

---

## 2. Plane matrix (relationships, not strips)

During Mag-PI burst window  
`2026-08-01 11:12:22` → `11:21:55` · depth ~415–420 ft · n=50 · cadence ~11s · RPM=0 · SPP~0:

| Plane | Source | State | Notes |
|-------|--------|-------|-------|
| **Tool memory (Mag-PI)** | Dump `SYSTEM` / Mag-PI events | **LIVE** | 50 events; 0 during 22h static park; all on RIH label |
| **Surface WITS** | ACQ `.emz` WITS | **LIVE** | Depth hold neighborhood; RPM=0; SPP~0 |
| **Mud-pulse RT decoder** | Decoder sessions in `.emz` | **DEAD** in burst | 0 sessions starting in burst; last S0028 before; first S0029 ~28 min after burst end (11:50) |
| **Survey** | Tool survey + WITS depth | **LIVE / clean** | Survey 11:13:15 @ 415.6 ft between Mag-PI ticks |
| **BHA health (shock / bus)** | Dump shock / voltages | **Quiet in park** | Supports “not BHA failure during top-drive park” narrative |

**Deviation from expected process (epistemic cut):**  
Naive process: “if the tool is eventing, RT decode should see something” or “if surface is quiet, tool is quiet.”  
Observed relation: **tool memory LIVE + decoder DEAD + surface HOLD** → multi-plane obstruction.  
That obstruction **is** the product, not a missing chart.

---

## 3. OPEN claims (sealed)

Source of truth (content + hash):

- Path (sandbox extract):  
  `123abc/_sandbox_extract/SandBox/Post_Run_SSI_Reports/ACQ_RT/TD26324/FORENSIC_DOSSIER/claims.json`
- **SHA-256:** `baea65a9f52a820d98eec0ebb3d75b072a77f1edc71a48e966d076493c924405`  
  (matches SSI release certificate artifact table)

| ID | Confidence | Claim (short) | Planes required |
|----|------------|---------------|-----------------|
| PRE_DRILL_WAIT | HIGH | 52h pre-drill ACQ wait shallow | WITS |
| TOPDRIVE_POOH_DOWNTIME | HIGH | POOH ~2089 then ~22h static ~230 ft; top-drive narrative | WITS depth |
| BHA_WITNESS | HIGH | Bus ~27V, shock quiet in park — BHA not failure mode | tool voltages/shock + park phase |
| MAGPI_NOT_PARK | HIGH | 0 Mag-PI in 22h static; 50 on RIH | tool Mag-PI + phase labels |
| MAGPI_BURST_415 | HIGH | All 50 in 9.5 min @ 415–420 ft, RPM=0 SPP~0 ~11s | Mag-PI + WITS |
| SURVEY_BETWEEN_TICKS | HIGH | Survey 11:13:15 @ 415.6 ft between ticks; QC clean | survey + Mag-PI clock |
| **MAGPI_DOWNHOLE_ONLY_NO_RT_DECODE** | **HIGH** | **Zero decoder sessions in burst; S0029 re-acquire ~28 min later — tool memory only** | **Mag-PI + decoder + (context) WITS** |
| DECODER_REACQUIRE_SNR_LIVE | HIGH | S0029 SNR_p50~6.5; stream not dead on restart | decoder SNR |

**Golden claim (the one that sells the operator):**  
`MAGPI_DOWNHOLE_ONLY_NO_RT_DECODE` — multi-plane sealed; single-plane Flatland cannot OPEN this.

---

## 4. Continuity guards (what must STOP)

| Forced story | Why STOP |
|--------------|----------|
| “Mag-PI was a surface/decode glitch” | Decoder DEAD; tool memory LIVE |
| “Nothing happened at 415 — surface quiet” | Mag-PI 50 + survey between ticks |
| “BHA failed during top-drive park” | Shock quiet / bus stable (BHA_WITNESS) without multi-plane contradicting bind for that claim’s scope |
| Claim needing both dump + EMZ with only one side dropped | Incomplete cover — residue, not OPEN |

---

## 5. Inputs (user will have)

Same classes as SandBox product contract:

1. **Tool dump folder** — MicroPulse-class CSVs for the run  
2. **ACQ export** — `.emz` (or unpacked: `well.db`, `phm.db`, WITS, Decoder Logs)

Customer send ZIP deliberately **excludes** raw dumps and `.emz` (release notes). Golden path for *recompute* needs the full SandBox; golden path for *inspect Q* needs `claims.json` + this document + release cert.

---

## 6. Validity spine tags used

From `VALIDITY_LEDGER.md` load-bearing OPEN set:

- G1 selection · G2 value · 1.1–1.3 condition/activity · 3.2/6.2 continuity · 5.2 term series · 9.2 relations · dump+DB face · explore≠certify  

**Not used:** ATFT numerics, quale=consciousness, Universal Tensor Space as physics.

---

## 7. Verify

```bash
python golden_paths/filmore_multiplane_v1/verify_golden.py
```

Expect: `GOLDEN VERIFY OK` when sandbox extract (or `GOLDEN_SANDBOX_ROOT`) is present.

---

## 8. Residue (honest)

- Full EDR×depth continuous tensor product beyond this window — not required for this Q  
- `phm.db` mining — not used for these eight claims  
- Independent re-derive of every fusion CSV from raw in this script — deferred; we bind to sealed `claims.json` hash  
- Public redaction policy for well names if this pack leaves private workshop — decide before external share  

---

*One operator. One golden Q. Continuous map from Geometry of Being → multi-plane files → sealed claim.*
