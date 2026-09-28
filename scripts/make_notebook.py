from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md('''# Maharashtra heatwave severity with KNN
Machine Learning (216H13C501), Exp8 extension · KJS-CES-01

**Operational Heatwave Severity Rules (Project Approximation)**

The high KNN performance is expected because the target labels are generated from two of the model's input variables. The experiment therefore evaluates how effectively KNN approximates our operational decision boundaries, not independent forecasting capability.

Run the acquisition command in the README first. This notebook then prepares the feature handoff, trains the single pipeline, and produces every report figure and table. No synthetic observations are used.''')
code('''from pathlib import Path
import json, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from heatwave import INPUT_COLUMNS, LABEL_MAP
from heatwave.pipeline import FeatureBuilder
from heatwave.model import split_by_year, tune_model, save_artifact, load_artifact
from heatwave.evaluate import *
ROOT = Path.cwd() if (Path.cwd() / "heatwave").exists() else Path.cwd().parent
OUT = ROOT / "reports/results"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
def figure(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    display(fig)
    plt.close(fig)
subprocess.run([sys.executable, "-m", "heatwave.cli", "prepare", "--root", str(ROOT)], check=True)
inspection = json.loads((ROOT / "data/processed/inspection.json").read_text())
display(inspection)''')
md('''## Normal and labeling
Normals pool all finite Tmax observations from a ±7-day window in 1991–2014 per cell and fixed calendar day. February 29 is excluded. The normal differs from IMD's official normal. Baseline years precede all model years.

Base thresholds are 40°C for plains, 37°C for coastal and 30°C for hilly. Above the base, departures of 4.5°C and 6.5°C trigger Heatwave and Severe. Where normal ≥40°C, absolute maxima of 45°C and 47°C also trigger those classes. The higher severity wins. These rules are unverified project approximations, applied per cell per day without station-level persistence criteria.

The editable project lookup assigns longitude <74°E to coastal, 74–75°E below 20°N to hilly, and the remainder to plains. This deliberately coarse geographic proxy has no terrain validation. A revised lookup changes target labels and requires the full analysis to be rerun.''')
code('''features = pd.read_parquet(ROOT / "data/processed/features.parquet")
train, test = split_by_year(features)
counts = class_counts(train, test)
counts.to_csv(OUT / "class_counts.csv")
display(counts)
if counts.loc["Severe Heatwave", "test"] < 30:
    print("CAUTION: fewer than 30 Severe test observations; recall is unstable. This is a project reporting heuristic.")
np.testing.assert_allclose(FeatureBuilder().transform(features).departure, features.departure)
assert pd.to_datetime(train.date).dt.year.max() == 2022
assert pd.to_datetime(test.date).dt.year.min() == 2023
cells = pd.read_csv(ROOT / "data/processed/cells.csv")
display(cells.groupby("region_type").size().rename("grid_cells"))''')
md('''## Search and model handoff
The whole pipeline is refit for each leave-one-year-out fold, including scaling. Search: 13 odd K values, two metrics and two vote weightings, selected by macro-F1 with all three classes explicitly included. Training scores are fold averages. Test data never enters tuning.''')
code('''search = tune_model(train, n_jobs=4)
cv = pd.DataFrame(search.cv_results_)
cv.to_csv(ROOT / "data/processed/grid_search.csv", index=False)
cv.to_csv(OUT / "grid_search.csv", index=False)
selected = search.best_params_
model = search.best_estimator_
artifact = save_artifact(model, ROOT / "models/knn_heatwave.joblib", source="IMD", best_params=selected, training_years=list(range(2015, 2023)), baseline_years=[1991, 2014], end_year=inspection["end_year"])
print("Selected:", selected, "CV macro-F1:", search.best_score_)
comparison_columns = ["param_knn__n_neighbors", "param_knn__metric", "param_knn__weights", "mean_test_macro_f1", "std_test_macro_f1", "mean_test_accuracy", "rank_test_macro_f1"]
display(cv[comparison_columns].sort_values("rank_test_macro_f1"))
figure(plot_k_curves(cv), "k_curves")
if selected["knn__metric"] != "euclidean" or selected["knn__weights"] != "uniform":
    figure(plot_k_curves(cv, selected["knn__metric"], selected["knn__weights"]), "k_curves_selected")''')
md('''## Held-out evaluation
Macro-F1 and Severe recall accompany accuracy and per-class precision/recall/F1. The majority baseline uses the most frequent **training** class. The three test years are evaluated only after model selection.''')
code('''metrics, reports, fig, prediction = evaluate_test(model, train, test)
metrics.to_csv(OUT / "metrics.csv")
display(metrics)
for name, report in reports.items():
    report.to_csv(OUT / ("classification_report.csv" if name == "KNN" else "baseline_report.csv"))
    print(name)
    display(report)
figure(fig, "confusion_matrix")
by_year = pd.DataFrame([{ "year": int(year), **scores(group.label, model.predict(group[INPUT_COLUMNS]))} for year, group in test.groupby(pd.to_datetime(test.date).dt.year)])
by_year.to_csv(OUT / "metrics_by_year.csv", index=False)
display(by_year)
figure(decision_regions(train, k=selected["knn__n_neighbors"], metric=selected["knn__metric"], weights=selected["knn__weights"]), "decision_regions")
errors, bands, fig = boundary_errors(test, prediction)
errors.to_csv(OUT / "prediction_errors.csv", index=False)
bands.to_csv(OUT / "boundary_error_summary.csv")
display(bands)
figure(fig, "boundary_errors")''')
md('''The decision-region figure uses a **separate two-feature KNN fitted only on plains training observations**. It is not a slice of the deployed six-numeric-feature model. Its background may include combinations absent from the data. Dashed lines mark departure thresholds; the dotted line marks the base threshold.

The error-band distance is a diagnostic proxy to the nearest candidate threshold, not exact distance to the piecewise decision boundary. Candidate normal-temperature and absolute-temperature gates are included.''')
md('''## Noise robustness
Perturb only held-out maximum temperature, keep normal temperature fixed and recompute departure inside the pipeline. Use σ = 0.5, 1, 2°C and seeds 11, 23, 37, 51, 71. For each odd K, refit on training observations with selected metric and weighting fixed. Reuse the same perturbation across K. Test noise results are diagnostic and do not select the deployed model.

Compare predictions with both original labels and recomputed project rules. The rules-versus-original reference isolates how often the rule target itself changes. Tables preserve each seed, and means/standard deviations summarize them.''')
code('''noise = noise_robustness(model, train, test)
noise.to_csv(OUT / "noise_by_seed.csv", index=False)
noise_mean, fig = plot_noise(noise)
noise_mean.to_csv(OUT / "noise_mean.csv", index=False)
noise.groupby(["k", "sigma", "comparison"])[["macro_f1", "severe_recall"]].agg(["mean", "std"]).to_csv(OUT / "noise_summary.csv")
display(noise_mean)
figure(fig, "noise_robustness")''')
md('''## Fresh-process parity and report
The artifact contains the pipeline, label map, input columns and exact scikit-learn version. The app imports the same package and calls the pipeline directly. The neighbor table reports counts and weighted vote shares, which are not calibrated probabilities.''')
code('''fixed = test[INPUT_COLUMNS].iloc[::max(1, len(test)//20)].head(20)
fixed.to_json(OUT / "parity_inputs.json", orient="table", date_format="iso")
script = "from heatwave.model import load_artifact; import pandas as pd, sys, json; a=load_artifact(sys.argv[1]); x=pd.read_json(sys.argv[2],orient='table'); print(json.dumps(a['pipeline'].predict(x).tolist()))"
actual = json.loads(subprocess.check_output([sys.executable, "-c", script, str(ROOT / "models/knn_heatwave.joblib"), str(OUT / "parity_inputs.json")], text=True))
np.testing.assert_array_equal(model.predict(fixed), actual)
from heatwave.demo import predict_observation
for _, row in fixed.iterrows():
    result = predict_observation(artifact, cells, **row.drop("region_type").to_dict())
    assert result["prediction"] == LABEL_MAP[int(model.predict(row.to_frame().T)[0])]
summary = {"selected": selected, "cv_macro_f1": float(search.best_score_), "inspection": inspection, "parity_rows": len(fixed)}
(OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
from heatwave.reporting import write_report
write_report(ROOT)
print("Artifact, tables, figures and report are ready.")''')
nb.cells = cells
nbf.write(nb, ROOT / "notebooks/analysis.ipynb")
