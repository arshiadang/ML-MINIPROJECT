"""Review saved evaluation evidence without fitting or changing the model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from .data_loader import read_boundary, proposed_region_lookup

REGION_COLORS = {"coastal": "#168b8b", "hilly": "#ad763d", "plains": "#617dba"}


def region_map(root):
    root = Path(root)
    cells = pd.read_csv(root / "data/processed/cells.csv")
    boundary = read_boundary(root / "data/boundaries/maharashtra.geojson")
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#faf9f5")
    ax.set_facecolor("#f0f4f6")
    polygons = list(boundary.geoms) if boundary.geom_type == "MultiPolygon" else [boundary]
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.fill(x, y, facecolor="#fffdf7", edgecolor="#6a777c", linewidth=.8)
        for hole in polygon.interiors:
            x, y = hole.xy
            ax.fill(x, y, facecolor="#f0f4f6")
    for region, color in REGION_COLORS.items():
        subset = cells[cells.region_type == region]
        ax.scatter(subset.lon, subset.lat, s=100, c=color, edgecolors="white", linewidths=1, label=f"{region.title()} ({len(subset)})", zorder=3)
    for _, cell in cells.iterrows():
        ax.annotate(cell.cell_id, (cell.lon, cell.lat), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7, color="#263b42")
    ax.axvline(74, color="#168b8b", ls="--", lw=.8, alpha=.6)
    ax.plot([75, 75], [15.7, 20], color="#ad763d", ls="--", lw=.8, alpha=.6)
    ax.plot([74, 75], [20, 20], color="#ad763d", ls="--", lw=.8, alpha=.6)
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", title="Maharashtra: geographic region proxy and all 26 grid centres")
    ax.set_aspect(1 / np.cos(np.radians(19)))
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=.12)
    fig.text(.5, .015, "Boundary: geoBoundaries / DataMeet, CC BY 2.5 India. Dashed lines are project cutoffs, not terrain boundaries.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, 1])
    return fig


def review_saved_results(root):
    root = Path(root)
    out = root / "reports/results"
    report = pd.read_csv(out / "classification_report.csv", index_col=0)
    cv = pd.read_csv(out / "grid_search.csv")
    fixed = cv[(cv.param_knn__metric == "euclidean") & (cv.param_knn__weights == "uniform")].sort_values("param_knn__n_neighbors")
    best = fixed.loc[fixed.mean_test_macro_f1.idxmax()]
    runner = fixed[fixed.param_knn__n_neighbors != best.param_knn__n_neighbors].sort_values("mean_test_macro_f1", ascending=False).iloc[0]
    fold_columns = [f"split{i}_test_macro_f1" for i in range(8)]
    delta = best[fold_columns].astype(float).to_numpy() - runner[fold_columns].astype(float).to_numpy()
    folds = pd.DataFrame({"held_out_year": range(2015, 2023), "k1_macro_f1": best[fold_columns].astype(float).to_numpy(), "runner_macro_f1": runner[fold_columns].astype(float).to_numpy(), "difference": delta})
    folds.to_csv(out / "k_fold_review.csv", index=False)
    noise = pd.read_csv(out / "noise_by_seed.csv")
    selected_k = json.loads((out / "run_summary.json").read_text())["selected"]["knn__n_neighbors"]
    noisy = noise[(noise.k == selected_k) & (noise.comparison == "Recomputed rules")]
    support = noisy.groupby("sigma").agg(seeds=("seed", "size"), seeds_with_severe=("severe_support", lambda x: int((x > 0).sum())), min_severe_cases=("severe_support", "min"), max_severe_cases=("severe_support", "max"), mean_defined_severe_recall=("severe_recall", "mean"))
    support.to_csv(out / "noise_severe_support.csv")
    cells = pd.read_csv(root / "data/processed/cells.csv")
    boundary = read_boundary(root / "data/boundaries/maharashtra.geojson")
    lookup = pd.read_csv(root / "data/region_lookup.csv")
    joined = cells.merge(lookup, on="cell_id", suffixes=("_processed", "_lookup"), validate="one_to_one")
    rule = proposed_region_lookup(cells).set_index("cell_id")
    summary = {
        "heatwave_f1": float(report.loc["Heatwave", "f1-score"]),
        "normal_f1": float(report.loc["Normal", "f1-score"]),
        "supported_macro_f1": float(report.loc[["Normal", "Heatwave"], "f1-score"].mean()),
        "fixed_three_class_macro_f1": float(report.loc["macro avg", "f1-score"]),
        "k_curve_best": int(best.param_knn__n_neighbors), "k_curve_best_score": float(best.mean_test_macro_f1),
        "k_curve_runner": int(runner.param_knn__n_neighbors), "k_curve_runner_score": float(runner.mean_test_macro_f1),
        "k_curve_gap": float(best.mean_test_macro_f1-runner.mean_test_macro_f1),
        "k1_fold_wins": int((delta > 1e-12).sum()), "fold_ties": int(np.isclose(delta, 0).sum()), "k1_fold_losses": int((delta < -1e-12).sum()),
        "best_metric_gap": float(best.mean_test_macro_f1-cv[(cv.param_knn__metric == "manhattan") & (cv.param_knn__n_neighbors == 1) & (cv.param_knn__weights == "uniform")].mean_test_macro_f1.iloc[0]),
        "region_counts": {k: int(v) for k,v in cells.region_type.value_counts().items()},
        "all_cells_inside": bool(all(boundary.covers(Point(r.lon,r.lat)) for r in cells.itertuples())),
        "lookup_matches_processed": bool(len(joined)==len(cells)==len(lookup) and (joined.region_type_processed==joined.region_type_lookup).all()),
        "lookup_matches_proxy": bool(lookup.set_index("cell_id").region_type.sort_index().equals(rule.region_type.sort_index())),
    }
    errors = pd.read_csv(out / "prediction_errors.csv")
    wrong = errors[errors.label != errors.prediction]
    summary.update(error_count=len(wrong), max_error_threshold_distance=float(wrong.threshold_distance.max()))
    (out / "sanity_review.json").write_text(json.dumps(summary, indent=2))
    fig = region_map(root)
    fig.savefig(out / "region_lookup_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return summary, folds, support


def review_sections(root):
    root = Path(root)
    s = json.loads((root / "reports/results/sanity_review.json").read_text())
    return {
        "metrics": f"""### Why 0.80, 0.90 and 0.60 are all different metrics

**0.80 is the F1 score for Heatwave alone**, treating Heatwave as the positive class (precision 0.8421, recall 0.7619). It is not two-class macro-F1. Normal has F1 {s['normal_f1']:.6f}.

**Supported two-class macro-F1 is {s['supported_macro_f1']:.6f} (about 0.90)**: (Normal F1 + Heatwave F1) / 2. **The predeclared three-class macro-F1 is {s['fixed_three_class_macro_f1']:.6f} (about 0.60)**: (Normal F1 + Heatwave F1 + 0) / 3. The zero term is the scoring convention for the absent Severe class; it is not a measured Severe F1. The same predictions produce all three values. The three-class score remains the primary protocol metric, and the two-class score supplies context rather than replacing it after evaluation.

The model makes {s['error_count']} held-out errors (5 missed Heatwaves and 3 false alarms). All lie within 0.25°C of a candidate rule threshold (largest recorded distance {s['max_error_threshold_distance']:.4f}°C). This supports a threshold-local error interpretation. The distance is a diagnostic proxy, so it does not prove the errors are harmless or establish forecast performance.
""",
        "k": f"""### K-curve sanity check

The Euclidean/uniform validation curve has its highest observed mean at K={s['k_curve_best']}: {s['k_curve_best_score']:.6f}. The next-highest K on that curve is K={s['k_curve_runner']}: {s['k_curve_runner_score']:.6f}, a gap of {s['k_curve_gap']:.6f}. The curve is not flat. K=1 wins {s['k1_fold_wins']} of the eight paired held-out-year comparisons with that runner-up, ties {s['fold_ties']}, and loses {s['k1_fold_losses']} (see `results/k_fold_review.csv`).

This is an empirical best mean, not evidence of a statistically distinct optimum. Fold standard deviations are about 0.10, and the folds' training sets overlap. The Euclidean lead over Manhattan at K=1 is only {s['best_metric_gap']:.6f}. Uniform and distance weights tie at K=1 because only one neighbour votes; deterministic grid ordering selects uniform. K=1 is defensible under the stated selection rule, while the metric/weighting distinction should not be overstated. Accuracy is nearly flat because Normal dominates; macro-F1 reveals the K sensitivity.
""",
        "noise": """### What zero Severe recall under noise actually means

The saved results already contain zero recall against noisy, recomputed Severe labels. For selected K=1, the Severe counts across seeds are **0–1 at σ=0.5°C**, **2–6 at σ=1°C**, and **47–68 at σ=2°C**. Every run with Severe support has recall **0.0000**: no Severe training neighbours exist, so KNN cannot reproduce the Severe labels created by noisy inputs. This is a demonstrated coverage limitation, not an undefined metric when the noisy target does contain Severe examples.

At σ=0.5°C, seed 23 has zero Severe cases and its recall is N/A. The reported mean 0.0000 averages the **four defined recalls**, not five zeros. All five recalls are defined for σ=1°C and σ=2°C. Against original labels, Severe recall remains N/A at every noise level because those labels have zero Severe support. `results/noise_severe_support.csv` records the support range and defined-seed count for each sigma. The same structural inability applies at every tested K; increasing K cannot introduce a class absent from training.
""",
        "map": f"""### Geographic review of the region proxy

![Maharashtra region lookup map](results/region_lookup_map.png)

Figure R. All {sum(s['region_counts'].values())} grid centres on the actual Maharashtra boundary, colored by the stored region lookup. Source: saved geoBoundaries polygon; dashed lines show the project's longitude/latitude cutoffs.

The map shows {s['region_counts']['coastal']} coastal labels along the western longitude column, {s['region_counts']['hilly']} hilly labels in the next column below 20°N, and {s['region_counts']['plains']} plains labels elsewhere. All centres pass the polygon check, the stored lookup agrees with the processed feature-cell metadata, and it exactly implements the documented proxy. Visual inspection makes the limitation explicit: these are rectangular coordinate bands, not a coastline-distance or terrain classification. In particular, the northern coastal-assigned point at 20.5°N, 73.5°E and the hilly band merit geographic review rather than automatic acceptance as physical terrain types.

This completes a positional and implementation sanity check, not geographic validation. The boundary map has no elevation or terrain layer and cannot establish whether the names are physically accurate. Project decision (Option A, 1 October 2026): retain the coordinate proxy as an explicitly accepted limitation for this mini project. Region assignment does not use measured coastal distance or elevation. Cell `20.5_73.5` remains labeled coastal and is flagged as a known questionable geographic assignment; its actual coastal distance has not been measured in this analysis. Acceptance of the proxy does not validate that physical label. The lookup and trained model remain unchanged, with no relabeling or retraining. A terrain-based reassignment is deferred to future work. The original approved revision-2 spec left the region assignment rule open and did not prescribe 50 km/600 m thresholds.
""",
    }
