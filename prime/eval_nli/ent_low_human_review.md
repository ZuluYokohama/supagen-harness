# ent_low gold review (12 pairs)

**Reviewer:** Grok session (model review — not a human labeler).  
**Standard:** strict NLI without private project axioms.  
**Date:** 2026-08-05

| id | gold | my_label | agree_with_gold? | note |
|----|------|----------|------------------|------|
| LAW-EL-01 | entailment | entailment | **yes** | constant output ⇒ no input info |
| LAW-EL-02 | entailment | soft_entailment | soft | “largely vocabulary” is interpretive |
| LAW-EL-03 | entailment | entailment | **yes** | forged field passes ⇒ no tamper protection |
| PEP-EL-01 | entailment | **neutral** | **no** | holonomy→global imbalance needs domain math |
| PEP-EL-02 | entailment | **neutral** | **no** | θ²/N scaling ≠ “reproduces classical model” |
| PEP-EL-03 | entailment | entailment | **yes** | same angle across copies ⇒ reproducible |
| FLD-EL-01 | entailment | soft_entailment | soft | needs multiplane certification law |
| FLD-EL-02 | entailment | **neutral** | **no** | downloads ≠ “no additional hardware” |
| MET-EL-01 | entailment | soft_entailment | soft | “constant fraction” stronger than P states |
| MET-EL-02 | entailment | **neutral** | **no** | reason≠label doesn’t entail commit order |
| MET-EL-03 | entailment | entailment | **yes** | 115 runs on done job ⇒ wasted compute |
| GEN-EL-01 | entailment | entailment | **yes** | closed to all traffic ⇒ nobody drove |

## Scorecard

- **Hard disagree with gold: 4 / 12** (PEP-EL-01, PEP-EL-02, FLD-EL-02, MET-EL-02)
- Soft / project-theory: 3 / 12
- Hard agree: 5 / 12

**Conclusion:** ≥4 disagreements → **eval needs repair before it can alone condemn the model.**  
Sub-chance `ent_low` is consistent with (a) model collapse to neutral **and/or** (b) golds that are domain-theory entailments mislabeled as pure NLI.

Raw stream (post-fix): entailment 13% overall still suggests **some** model under-calling entailment — but the discriminating cell is contaminated.

## Provenance

All rows in `nli_eval_v1.jsonl` were **model-generated** (bench session).  
Field: `label_source: model` until human-relabeled.
