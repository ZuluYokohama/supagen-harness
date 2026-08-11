"""Build the reader-facing, reproducible interval-gate notebook."""
from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "stratigraphic_interval_gate.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    notebook["cells"] = [
        markdown(
            """
            # Stratigraphic Contact Gate — interval-valued gamma evidence

            ## tl;dr

            On a strict 272-well holdout grouped by exact typewell profile and scored on all
            1,324,387 missing-TVT rows, the inherited inference-safe baseline scores **14.834 ft
            pooled RMSE**. Equal-cell typewell evidence reaches **14.437 ft**; calibrated ordered
            stay/cross/reverse transport reaches **14.501 ft**. Their development-only joint
            correction reaches **14.142 ft**, a **0.692 ft** improvement with an exact-typewell-group
            bootstrap interval of approximately **+0.350 to +1.025 ft**.

            This is a valid **MEASURE_ONLY** result, not OPEN. The fusion was proposed after the
            component results on this holdout were inspected, so it must survive a newly frozen
            outer-fold evaluation before supporting a generalized or IP performance claim.
            """
        ),
        markdown(
            """
            ## Context & Methods

            The geological object is a well trajectory through an ordered stratigraphic contact
            complex. Gamma ray is treated as interval/package evidence, not an independent point
            label. The experiment compares:

            1. an inherited, own-well inference-safe policy baseline;
            2. **equal-cell typewell transport**, where every occupied predicted TVT cell receives
               equal weight and carries empirical GR quartiles;
            3. **ordered reversible transport**, a frozen Viterbi solver with stay/cross/reverse
               transitions and 13-sample ordered windows across ±90 MD;
            4. simple and two-coefficient combinations of the two complementary corrections.

            ### Key Assumptions

            - Allowed validation inputs are horizontal `MD,X,Y,Z,GR,TVT_input` and typewell
              `TVT,GR` only.
            - Suffix `TVT`, formation surfaces, typewell `Geology`, PNGs, and ID lookup are forbidden.
            - Three exact train/test-overlap IDs are excluded before any split.
            - Exact duplicate typewell profiles stay on one side of the outer split.
            - Primary score is pooled station-level RMSE; per-well statistics are diagnostic.
            - Ordered diagnostic correction magnitudes in the raw artifact precede the fitted
              0.18335 correction shrink.
            """
        ),
        code(
            """
            from pathlib import Path
            import hashlib
            import json
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            ROOT_HINT = Path(__ROOT_HINT__)
            root_candidates = [Path.cwd(), *Path.cwd().parents, ROOT_HINT]
            ROOT = next(
                (path for path in root_candidates if (path / "research" / "results").exists()),
                None,
            )
            if ROOT is None:
                raise FileNotFoundError(
                    "Cannot locate research/results; set the notebook working directory to the worktree."
                )
            RESULT_DIR = ROOT / "research" / "results"
            RESULT_PATH = RESULT_DIR / "interval_gate_strict_confirm_v2.json"
            PER_WELL_PATH = RESULT_DIR / "interval_gate_strict_confirm_v2_per_well.csv"
            FOLD_PATH = RESULT_DIR / "interval_gate_strict_confirm_v2_fold_manifest.csv"
            PROVENANCE_PATH = RESULT_DIR / "interval_gate_strict_confirm_v2_provenance.json"
            FIGURE_PATH = RESULT_DIR / "interval_gate_strict_confirm_v2_summary.png"

            result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
            per_well = pd.read_csv(PER_WELL_PATH)
            folds = pd.read_csv(FOLD_PATH)
            provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
            print(f"Loaded {len(folds)} eligible wells; {len(per_well)} holdout wells.")
            """.replace("__ROOT_HINT__", repr(str(ROOT)))
        ),
        markdown("## Data"),
        code(
            """
            excluded = set(result["excluded_test_overlap_ids"])
            dev_groups = set(folds.loc[folds.split == "dev", "typewell_profile_hash"])
            hold_groups = set(folds.loc[folds.split == "holdout", "typewell_profile_hash"])
            hold_rows = int(folds.loc[folds.split == "holdout", "suffix_rows"].sum())

            assert excluded == {"000d7d20", "00bbac68", "00e12e8b"}
            assert len(folds) == 770
            assert folds.typewell_profile_hash.nunique() == 749
            assert not (dev_groups & hold_groups)
            assert hold_rows == 1_324_387 == result["base_model"]["holdout"]["n_rows"]
            assert result["evaluation_stride"] == 1
            assert provenance["status"] == "MEASURE_ONLY_NOT_OPEN"

            def sha256_file(path):
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                return digest.hexdigest()

            tracked_paths = {
                "research/interval_gate.py": ROOT / "research" / "interval_gate.py",
                "research/ordered_transport.py": ROOT / "research" / "ordered_transport.py",
                "research/test_interval_gate.py": ROOT / "research" / "test_interval_gate.py",
                "research/test_ordered_transport.py": ROOT / "research" / "test_ordered_transport.py",
                RESULT_PATH.name: RESULT_PATH,
                FOLD_PATH.name: FOLD_PATH,
                "competition_pdf": (
                    Path(provenance["data_dir"]) / "AI_wellbore_geology_prediction_task_en.pdf"
                ),
            }
            # The competition PDF lives in the private data directory, not in
            # the repo, so a reader without it must still get the in-repo
            # hashes checked rather than a FileNotFoundError before the first.
            verified_hashes = {
                name: (sha256_file(path) if path.exists() else "unavailable")
                for name, path in tracked_paths.items()
            }
            checked = {n: h for n, h in verified_hashes.items() if h != "unavailable"}
            skipped = sorted(set(verified_hashes) - set(checked))
            assert all(provenance["sha256"][n] == h for n, h in checked.items())
            if skipped:
                print(f"hash check skipped for unavailable inputs: {skipped}")

            data_summary = pd.DataFrame({
                "measure": [
                    "eligible wells", "unique typewell groups", "development wells",
                    "holdout wells", "holdout suffix rows",
                    "mean per-well holdout GR valid fraction", "provenance hashes verified"
                ],
                "value": [
                    len(folds), folds.typewell_profile_hash.nunique(),
                    int((folds.split == "dev").sum()), int((folds.split == "holdout").sum()),
                    hold_rows,
                    folds.loc[folds.split == "holdout", "gr_valid_fraction"].mean(),
                    verified_hashes == provenance["sha256"],
                ],
            })
            data_summary
            """
        ),
        markdown("## Results"),
        code(
            """
            baseline_rmse = result["base_model"]["holdout"]["pooled_row_rmse"]
            display_names = {
                "typewell_equal_cell": "Equal-cell typewell",
                "prefix_equal_cell": "Prefix equal-cell",
                "fused_interval_ridge": "Landscape ridge",
                "ordered_transport_raw": "Ordered raw",
                "ordered_transport_calibrated": "Ordered calibrated",
                "equal_ordered_average": "Equal + ordered average",
                "equal_ordered_joint": "Equal + ordered joint",
            }
            rows = [{
                "method": "Baseline", "pooled_rmse_ft": baseline_rmse,
                "gain_ft": 0.0, "median_well_rmse_ft": result["base_model"]["holdout"]["median_well_rmse"],
                "p90_well_rmse_ft": result["base_model"]["holdout"]["p90_well_rmse"],
                "win_rate": np.nan,
            }]
            for key, label in display_names.items():
                candidate = result["candidates"][key]
                metric = candidate["holdout"]
                gain = candidate["holdout_gain"]
                rows.append({
                    "method": label,
                    "pooled_rmse_ft": metric["pooled_row_rmse"],
                    "gain_ft": baseline_rmse - metric["pooled_row_rmse"],
                    "median_well_rmse_ft": metric["median_well_rmse"],
                    "p90_well_rmse_ft": metric["p90_well_rmse"],
                    "win_rate": gain["win_rate"],
                })
            comparison = pd.DataFrame(rows).sort_values("pooled_rmse_ft")
            comparison.round(4)
            """
        ),
        code(
            """
            # Anatomy of the headline fusion, fitted on development OOF corrections only.
            joint_coefficients = result["candidates"]["equal_ordered_joint"][
                "coefficients_from_dev_oof"
            ]
            typewell_shrink = result["candidates"]["typewell_equal_cell"][
                "calibrated_shift_shrink"
            ]
            ordered_shrink = result["candidates"]["ordered_transport_calibrated"][
                "correction_shrink_from_dev_oof"
            ]
            fusion_anatomy = pd.DataFrame({
                "component": ["Equal-cell typewell", "Ordered transport"],
                "raw_correction_shrink": [typewell_shrink, ordered_shrink],
                "joint_coefficient": [
                    joint_coefficients["typewell_equal_cell"],
                    joint_coefficients["ordered_transport"],
                ],
            })
            fusion_anatomy["effective_raw_multiplier"] = (
                fusion_anatomy.raw_correction_shrink * fusion_anatomy.joint_coefficient
            )

            base_well = pd.Series(result["base_model"]["holdout"]["per_well"])
            typewell_well = pd.Series(
                result["candidates"]["typewell_equal_cell"]["holdout"]["per_well"]
            )
            ordered_well = pd.Series(
                result["candidates"]["ordered_transport_calibrated"]["holdout"]["per_well"]
            )
            component_gains = pd.DataFrame({
                "typewell_gain": base_well - typewell_well,
                "ordered_gain": base_well - ordered_well,
            })
            component_correlation = pd.DataFrame({
                "measure": ["Pearson", "Spearman"],
                "per_well_gain_correlation": [
                    component_gains.corr(method="pearson").iloc[0, 1],
                    component_gains.corr(method="spearman").iloc[0, 1],
                ],
            })
            display(fusion_anatomy.round(4), component_correlation.round(4))
            """
        ),
        code(
            """
            # Exact-typewell-group bootstrap: resample reference-log groups, not individual wells.
            metric_by_well = pd.DataFrame({
                "well": sorted(result["base_model"]["holdout"]["per_well"]),
            })
            metric_by_well["base"] = metric_by_well.well.map(
                result["base_model"]["holdout"]["per_well"]
            )
            for key in ("typewell_equal_cell", "ordered_transport_calibrated",
                        "equal_ordered_average", "equal_ordered_joint"):
                metric_by_well[key] = metric_by_well.well.map(
                    result["candidates"][key]["holdout"]["per_well"]
                )
            metric_by_well = metric_by_well.merge(
                folds.loc[folds.split == "holdout",
                          ["well", "suffix_rows", "typewell_profile_hash"]],
                on="well", validate="one_to_one",
            )

            def group_bootstrap(candidate, draws=4000, seed=20260810):
                frame = metric_by_well.copy()
                frame["base_sse"] = frame.base.pow(2) * frame.suffix_rows
                frame["candidate_sse"] = frame[candidate].pow(2) * frame.suffix_rows
                grouped = frame.groupby("typewell_profile_hash").agg(
                    base_sse=("base_sse", "sum"), candidate_sse=("candidate_sse", "sum"),
                    rows=("suffix_rows", "sum"),
                )
                rng = np.random.default_rng(seed)
                samples = np.empty(draws)
                for i in range(draws):
                    take = rng.integers(0, len(grouped), len(grouped))
                    chosen = grouped.iloc[take]
                    samples[i] = (
                        np.sqrt(chosen.base_sse.sum() / chosen.rows.sum())
                        - np.sqrt(chosen.candidate_sse.sum() / chosen.rows.sum())
                    )
                observed = (
                    np.sqrt(grouped.base_sse.sum() / grouped.rows.sum())
                    - np.sqrt(grouped.candidate_sse.sum() / grouped.rows.sum())
                )
                return observed, np.quantile(samples, [0.025, 0.975])

            ci_rows = []
            for key, label in [
                ("typewell_equal_cell", "Equal-cell typewell"),
                ("ordered_transport_calibrated", "Ordered calibrated"),
                ("equal_ordered_average", "Equal + ordered average"),
                ("equal_ordered_joint", "Equal + ordered joint"),
            ]:
                observed, interval = group_bootstrap(key)
                ci_rows.append({"method": label, "gain_ft": observed,
                                "ci_low_ft": interval[0], "ci_high_ft": interval[1]})
            confidence = pd.DataFrame(ci_rows)
            confidence.round(4)
            """
        ),
        markdown(
            """
            ### Visual contract

            - **Question:** Which frozen interval corrections improve pooled competition RMSE, and
              which are harmful?
            - **Takeaway:** Equal-cell and calibrated ordered evidence are complementary; their
              joint correction is strongest, while unconstrained ordered transport is harmful.
            - **Form:** signed horizontal gain bars with a zero reference, plus group-bootstrap
              intervals for the four surviving candidates.
            - **Data sufficiency:** eight methods and 265 independent typewell groups.
            - **Palette:** one blue root for positive evidence, orange for negative evidence,
              charcoal references; signs, labels, and intervals provide non-color distinction.
            - **Surface:** static Matplotlib figure embedded in this notebook and exported beside
              the result artifact.
            """
        ),
        code(
            """
            blue, blue_dark = "#4C78A8", "#234A76"
            orange, charcoal, grid = "#E07A2D", "#263238", "#D9DEE3"
            chart = comparison.loc[comparison.method != "Baseline"].sort_values("gain_ft")
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

            colors = [blue if value >= 0 else orange for value in chart.gain_ft]
            axes[0].barh(chart.method, chart.gain_ft, color=colors, edgecolor=charcoal, linewidth=0.6)
            axes[0].axvline(0, color=charcoal, linewidth=1)
            for y, value in enumerate(chart.gain_ft):
                if value >= 0:
                    axes[0].text(value + 0.035, y, f"{value:+.3f}", va="center", ha="left", fontsize=9)
                else:
                    axes[0].text(value + 0.06, y, f"{value:+.3f}", va="center", ha="left",
                                 fontsize=9, color="white")
            axes[0].set_title("Pooled RMSE change versus baseline")
            axes[0].set_xlabel("Improvement (ft); positive is better")
            axes[0].grid(axis="x", color=grid, linewidth=0.7)
            axes[0].set_axisbelow(True)

            ci_plot = confidence.sort_values("gain_ft")
            xerr = np.vstack((ci_plot.gain_ft - ci_plot.ci_low_ft,
                              ci_plot.ci_high_ft - ci_plot.gain_ft))
            axes[1].errorbar(ci_plot.gain_ft, ci_plot.method, xerr=xerr, fmt="o",
                             color=blue_dark, ecolor=blue, capsize=4, linewidth=2)
            axes[1].axvline(0, color=charcoal, linewidth=1)
            for y, value in enumerate(ci_plot.gain_ft):
                if y == len(ci_plot) - 1:
                    axes[1].text(value, y - 0.20, f"{value:+.3f}", ha="center", va="top",
                                 fontsize=9)
                else:
                    axes[1].text(value, y + 0.18, f"{value:+.3f}", ha="center", fontsize=9)
            axes[1].set_title("Exact-typewell-group uncertainty")
            axes[1].set_xlabel("Pooled RMSE improvement (ft), 95% bootstrap interval")
            axes[1].grid(axis="x", color=grid, linewidth=0.7)
            axes[1].set_axisbelow(True)

            fig.suptitle("Stratigraphic interval gate — 272 held-out wells / 1,324,387 rows",
                         fontsize=14, color=charcoal)
            fig.savefig(FIGURE_PATH, dpi=170, bbox_inches="tight", facecolor="white")
            plt.show()
            print(f"Saved {FIGURE_PATH}")
            """
        ),
        code(
            """
            # Concentration audit for the exploratory joint fusion.
            audit = metric_by_well.copy()
            audit["base_sse"] = audit.base.pow(2) * audit.suffix_rows
            audit["joint_sse"] = audit.equal_ordered_joint.pow(2) * audit.suffix_rows
            audit["sse_gain"] = audit.base_sse - audit.joint_sse
            positive = audit.loc[audit.sse_gain > 0].sort_values("sse_gain", ascending=False)
            top10_share = positive.head(10).sse_gain.sum() / audit.sse_gain.sum()

            removal = []
            for n in (0, 1, 5, 10, 20):
                removed = set(positive.head(n).well)
                kept = audit.loc[~audit.well.isin(removed)]
                gain = (np.sqrt(kept.base_sse.sum() / kept.suffix_rows.sum())
                        - np.sqrt(kept.joint_sse.sum() / kept.suffix_rows.sum()))
                removal.append({"top_positive_wells_removed": n, "pooled_gain_ft": gain})
            print(f"Top-10 positive SSE share: {top10_share:.1%}")
            pd.DataFrame(removal).round(4)
            """
        ),
        markdown(
            """
            ## Takeaways

            1. **The user’s interval interpretation survives measurement.** Equal-cell typewell
               evidence improves the strict full-row holdout by 0.397 ft pooled RMSE.
            2. **Ordering adds a distinct surface.** Ordered transport has near-zero per-well gain
               correlation with equal-cell transport. After strong development-only restriction,
               it improves 64.3% of wells.
            3. **Unrestricted topology is rejected.** Raw ordered transport worsens pooled RMSE by
               1.552 ft; its inferred reversals cannot yet be read literally as geological crossings.
            4. **Complementarity is the strongest measured entry point.** The joint correction
               improves pooled RMSE by 0.692 ft, median-well RMSE by 1.077 ft, and p90 well RMSE by
               1.652 ft. Removing its ten strongest wells still leaves about 0.279 ft improvement.
            5. **Residue remains.** Prefix-only transport is null, raw ordered transport is harmful,
               and the fusion reused a previously inspected holdout.

            **Next gate:** freeze the equal-cell and ordered corrections exactly as encoded, then run
            repeated outer folds grouped by exact typewell hash, followed by a spatial-block stress
            test. No algorithm revision may occur between those frozen evaluations.
            """
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
