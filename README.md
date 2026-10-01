# Maharashtra Heatwave Lab

KNN same-day severity classification for Machine Learning (216H13C501), Exp8 extension, KJS-CES-01.

**Real-data limitation:** the acquired 1° IMD data and approved rules produce 142 Heatwave training observations and 21 test observations, with **zero Severe Heatwave observations in either split**. The fitted KNN cannot predict Severe. The report retains the requested rules/split and explains this limitation. The app displays the project rule label alongside KNN, including when the rule says Severe.

Labels are the **Operational Heatwave Severity Rules (Project Approximation)**. They derive from model inputs. Results measure rule approximation, not independent forecasting.

## Run the existing demo

Use Python 3.11–3.13 (the implementation was verified with Python 3.12).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The trained artifact and processed data are included. On the current machine the `.venv` is already installed, so `.venv/bin/streamlit run app.py` is sufficient.

## Reproduce from raw data

From this project directory:

```bash
source .venv/bin/activate
python scripts/download_data.py
python -m heatwave.cli prepare
python scripts/run_analysis.py
python -m pytest -q
streamlit run app.py
```

The downloader retrieves annual IMD Tmax binaries for 1991–2025 (about 49 MB total) and a geoBoundaries Maharashtra polygon. Existing raw files are reused after a size check. SHA-256 hashes, source URLs and boundary licensing are saved. Network access is needed only for acquisition and package installation. Data acquisition uses three concurrent requests and retries transient errors.

`notebooks/analysis.ipynb` is the authoritative executable analysis: preparation, counts, full grid search, all report figures and tables, noise experiments, fresh-process parity checks, and the generated Markdown report. `scripts/run_analysis.py` executes and saves its outputs with the active Python environment. `scripts/make_notebook.py` regenerates the notebook source and clears outputs; it is for development, not a prerequisite to rerunning the delivered notebook.

For artifact-only training, use `python -m heatwave.cli train --jobs 4`. For a future approved end-year change, pass `--end-year` to both acquisition and preparation, and edit the notebook preparation call accordingly. Do not silently change the locked evaluation split.

## Project layout

- `heatwave/`: shared loader, climatology, labeling, pipeline, model, evaluation, app adapter and report generator.
- `data/raw/`: IMD annual files plus checksum manifest.
- `data/boundaries/`: Maharashtra polygon and source/license metadata.
- `data/region_lookup.csv`: editable per-cell project region assignment.
- `data/processed/features.parquet`: Person 1 feature handoff.
- `models/knn_heatwave.joblib`: one dictionary containing the fitted pipeline, label map, raw input columns and sklearn version.
- `notebooks/analysis.ipynb`: executed reproducible analysis.
- `reports/report.md`: generated written report, with linked notebook figures.
- `reports/results/`: metrics, full grid search, errors, noise seeds/means and figures.
- `reports/slides/heatwave_knn.pptx`: editable 10-slide presentation.
- `scripts/build_slides.mjs`: presentation authoring source using the bundled Artifact Tool runtime; its narrative reflects this completed dataset.
- `tests/`: unit and integration verification.

## Region lookup and data contract

The included region rule is a transparent, coarse **project proxy requiring team review**: longitude <74°E is coastal; 74≤longitude<75°E and latitude<20°N is hilly; all remaining Maharashtra grid centres are plains. It is not a terrain-derived classification or an IMD region lookup. The authoritative editable file has columns `cell_id,region_type`. Preparation preserves it rather than overwriting it. If you change it, rerun preparation, notebook and slide generation.

The raw input contract is `date,lat,lon,region_type,max_temp,normal_temp`. FeatureBuilder computes departure and the 365-day seasonal terms. The feature handoff also stores `cell_id,departure,label`. The app never accepts departure as an independent input.

The baseline normal pools all finite values in ±7 fixed-calendar days across 1991–2014 and excludes February 29. All 35 full annual files are loaded, including the required seasonal margins. The model window is March–June, 2015–2025. Train: 2015–2022. Test: 2023–2025. There is no random train/test split.

## Metrics and interpretation

The fixed-label macro-F1 includes all three requested classes even when a class has zero support. With no Severe class in the real data, its contribution is zero and the best possible fixed-three-class macro-F1 is 2/3. Severe recall is `NaN`/N/A when its ground-truth support is zero; do not interpret this as measured recall. Raw sklearn classification-report tables use zero placeholders for unsupported classes. The report explains this convention.

Noise experiments perturb only test Tmax, with the same five seeds for every K. Noisy rule labels are recomputed through the same labeling function. The selected metric/weighting remain fixed; noise results never select the model. The separate two-feature decision-region KNN is plot-only.

The artifact requires the importable `heatwave==0.1.0` source and scikit-learn 1.6.1. Only load trusted local joblib files. A version mismatch raises an error. The app displays weighted votes and neighbour counts; vote shares are not calibrated probabilities.

## Sources

- [IMD annual maximum temperature archive](https://www.imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html). The live form listed 2025 and that file was downloaded and parsed successfully.
- Srivastava, Rajeevan and Kshirsagar (2009), [doi:10.1002/asl.232](https://doi.org/10.1002/asl.232).
- [geoBoundaries India ADM1](https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/), underlying DataMeet India community / Election Commission of India. CC BY 2.5 India. The exact boundary revision and attribution are preserved in `data/boundaries/source.json`.
- `reports/design_spec.md`: supplied approved design, revision 2.

## Review updates and revised interface

The app now separates classification, model evidence, the region map and method/data notes. Example readings, an immediate departure calculation and explicit model/rule disagreement make the demo easier to inspect. Changed inputs hide the previous prediction until you classify again.

The report explains the different F1 scores: **0.80 Heatwave-only F1**, **0.899789 macro-F1 for Normal and Heatwave**, and **0.599860 for the predeclared three-class convention**. The K review compares saved year-fold results, and the noise review records which seeds actually have Severe support. A zero-support run stays N/A.

`python scripts/refresh_review.py` regenerates the map, review tables and written report from saved evidence without retraining. The same review is included as an executed section of `notebooks/analysis.ipynb` and its source generator. New artifacts are `region_lookup_map.png`, `k_fold_review.csv`, `noise_severe_support.csv`, and `sanity_review.json` under `reports/results/`.

The geographic review checks positions and proxy consistency on the actual state boundary. It does **not** validate terrain or distance from the coast. In particular, the northern coastal assignment and hilly column warrant team review. The existing model artifact and region lookup remain unchanged.
