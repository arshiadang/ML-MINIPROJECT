# Heatwave Severity Classification with KNN: Final Design Spec (rev 2)

**Course:** Machine Learning (216H13C501), Exp8 (KNN) extended into a mini project
**Use case:** KJS-CES-01, Climate Intelligence for Heatwave Monitoring, Prediction, and Early Warning
**Team:** 3 members
**Status:** Final. Approved for implementation.

## 1. Goal

Train a K-Nearest Neighbors classifier that labels a day's weather observation for a Maharashtra grid cell as **Normal / Heatwave / Severe Heatwave**. Analyse how K, the distance metric and vote weighting affect it, and expose it through a Streamlit demo app.

## 2. Decisions (locked)

| Decision | Choice |
|---|---|
| Technique | KNN only (Module 4, unit 4.5), via scikit-learn. No other models. |
| Target | Same-day severity classification |
| Region and period | Maharashtra only, 2015-2025, March-June |
| Regions | Our own plains / coastal / hilly lookup (not IMD's seven regions) |
| Label rules | **Operational Heatwave Severity Rules (project approximation)** of IMD's criteria. Not verified against IMD's published criteria. |
| Model artifact | One scikit-learn `Pipeline` (feature building, scaling, KNN) saved as a single object |
| Calendar | Fixed 365-day calendar for the seasonal features |
| Demo app | Streamlit |
| Use case | KJS-CES-01 |

## 3. Deliverables

1. A shared Python package `heatwave/` (data, labels, pipeline, evaluation).
2. An analysis notebook that produces every figure and table in the report.
3. A Streamlit app that loads the one saved pipeline and predicts severity from user input.
4. The written report and slides (Person 3 drafts, all review).

## 4. Framing the team must keep honest

- **The rules are ours, not IMD's.** Everywhere in the code, notebook, app and report they are called the **"Operational Heatwave Severity Rules (Project Approximation)"**. The words "official IMD classification" are never used for them.
- **The labels are computed from two of the model's inputs.** Labels come from rules on max temperature and departure from normal, and the model receives those same quantities. High KNN scores are therefore expected and are not hidden.
- Required wording in the report (adapt, don't drop): *"The high KNN performance is expected because the target labels are generated from two of the model's input variables. The experiment therefore evaluates how effectively KNN approximates our operational decision boundaries, not independent forecasting capability."*
- The analysis value is in where errors fall (near threshold boundaries), how K, metric and weighting change them, and robustness to noisy readings.

## 5. Architecture

One package `heatwave/`, used by both the notebook and the app. The app never re-implements preprocessing; it calls the saved pipeline.

| Module | Responsibility |
|---|---|
| `data_loader` | Parse IMD gridded max-temp files (GRD) into one tidy table: date, lat, lon, max_temp. Keep only cells inside Maharashtra. |
| `climatology` | Daily normal max temp per cell (6.2). |
| `labeling` | Apply the Operational Heatwave Severity Rules (6.3). |
| `pipeline` | `FeatureBuilder` transformer and the full `Pipeline` (5.1). |
| `model` | Hyperparameter search, fitting, save and load of the artifact. |
| `evaluate` | Metrics and plots. |

**Ownership:** Person 1: data_loader, climatology, labeling, region lookup. Person 2: pipeline, model, evaluate. Person 3: Streamlit app, report, slides.

### 5.1 The pipeline
Input columns (raw): `date, lat, lon, region_type, max_temp, normal_temp`.

1. `FeatureBuilder` (custom transformer): computes `departure = max_temp - normal_temp`, `doy365` from `date` (6.5), then `doy_sin` and `doy_cos`.
2. `ColumnTransformer`: `StandardScaler` on `max_temp, departure, doy_sin, doy_cos, lat, lon`; `OneHotEncoder` on `region_type` (unknown categories raise an error).
3. `KNeighborsClassifier`.

Because scaling sits inside the pipeline, cross-validation refits the scaler on each training fold only, and the app cannot preprocess differently from the notebook.

### 5.2 Handoff contracts
- **Feature table** (Person 1 to 2): `data/processed/features.parquet` with columns `date, cell_id, lat, lon, region_type, max_temp, normal_temp, departure, label`. `departure` is stored for labeling and EDA; the pipeline recomputes it from `normal_temp`, and a test checks the two agree.
- **Artifact** (Person 2 to 3): `models/knn_heatwave.joblib`, a dict with keys `pipeline, label_map, input_columns, sklearn_version`. Loading needs the `heatwave` package importable (the custom transformer is pickled by reference) and the same scikit-learn version, so both are pinned in `requirements.txt`.

## 6. Data, labels, features

### 6.1 Source and scope
- IMD gridded daily max temperature (the repository named in the use case: imdpune.gov.in/lrfindex.php).
- Files must cover **1991-2025** (1991-2014 for the normal, 2015-2025 for modelling). Modelling window is 1 March to 30 June.
- Exact resolution and file layout are confirmed in the first task (data inspection). If 2025 is not yet published, use the latest available year.

### 6.2 Climatological normal (exact definition)
- **For each grid cell and calendar day** (365-day calendar, 6.5),
- **use the years 1991-2014**,
- **take the average of max temperature over a +/-7-day window** around that calendar day, pooled across all those years (15 days x 24 years, minus missing values).

Feb 29 observations are excluded from the calculation. Because the window reaches 7 days beyond the modelling window, the loader must keep 1991-2014 data for at least 22 Feb to 7 Jul. These years lie entirely before the modelling window, so no train or test data enters the normal. It differs from IMD's official normal; the report says so.

### 6.3 Operational Heatwave Severity Rules (project approximation)
Implemented as follows and described in the report as our operationalisation, not verified IMD criteria:

- Base threshold on max temp: plains >= 40 C, coastal >= 37 C, hilly >= 30 C. Below it, the label is Normal.
- Departure from normal d: 4.5 <= d < 6.5 gives Heatwave; d >= 6.5 gives Severe.
- Absolute rule where the normal is >= 40 C: max temp >= 45 C gives Heatwave; >= 47 C gives Severe.
- The higher severity among the applicable rules wins.

Other approximations to state in the report:
- Rules are applied per grid cell per day (IMD declares at station level over consecutive days).
- `region_type` per cell is our own assignment from location, kept in `data/region_lookup.csv` (columns `cell_id, region_type`). Person 1 documents the assignment rule in the report.

### 6.4 Features
Inside the pipeline: `max_temp`, `departure`, `doy_sin`, `doy_cos`, `lat`, `lon` (all standardised) and `region_type` (one-hot, unscaled). Humidity is not in the IMD file and is not used.

### 6.5 Calendar convention
- `doy365` is the day of year in a **non-leap reference calendar**, so 1 March is day 60 in every year, leap or not. Implementation: map the date to the same month and day in a fixed non-leap year, then take its day of year.
- `doy_sin = sin(2*pi*doy365/365)`, `doy_cos = cos(2*pi*doy365/365)`.
- The modelling window (1 Mar to 30 Jun) contains no 29 Feb, so no sample needs special handling; 29 Feb only matters for excluding it from the normal (6.2).

## 7. Split and evaluation

- **Split by year:** train 2015-2022, test 2023-2025. No random splitting.
- **Tuning:** leave-one-year-out cross-validation across the training years, refitting the whole pipeline each fold.
- **Search grid:** K in odd values 1-25; metric Euclidean or Manhattan; vote weights uniform or distance. Selection score: macro-F1.
- **Class imbalance:** no resampling by default. Macro-F1 for selection and per-class reporting handle it.
- **Reported on the test set:** confusion matrix; per-class precision, recall, F1 (Severe recall highlighted); a majority-class baseline for context. Accuracy is never reported alone.

### 7.1 K curves (bias-variance)
- Metric and weighting are **fixed** at Euclidean and uniform; only K changes (K = 1, 3, ..., 25).
- Two panels, each showing train vs cross-validated: **Accuracy vs K** and **Macro-F1 vs K**. Macro-F1 is included because it is the selection score.
- Uniform weighting is used on purpose: with distance weighting a training point is its own zero-distance neighbour, so train scores become trivially perfect and hide the trade-off. If the selected configuration differs from the default, add the same two panels for it as a secondary figure.

### 7.2 Other figures
- Decision regions on max_temp vs departure for one region type, from a separate 2-feature KNN fitted only for the plot (stated in the caption).
- Grid-search comparison table.

### 7.3 Noise robustness
Setup: add Gaussian noise (sigma = 0.5, 1, 2 C) to the **test `max_temp` only**. `normal_temp` is unchanged, so the pipeline recomputes `departure` from the noisy value. Repeat over 5 fixed seeds and report the mean. Report, per K and sigma, both:

1. **Noisy prediction vs original label** (macro-F1, Severe recall): is the model stable under noise?
2. **Noisy prediction vs new rule-based label** (macro-F1, Severe recall), where the rule label is recomputed by the same `labeling` function on the noisy inputs: does the model still follow the rule?

Reference line: the rule-based label on the noisy inputs vs the original label. This shows how much of the drop in (1) is just the correct answer changing under noise.

### 7.4 Reproducibility
Fixed random seeds; all figures come from the notebook.

## 8. Demo app

- **Inputs:** max temp, **normal temp** (not departure), date (restricted to 1 Mar to 30 Jun, the range the model was trained on), lat/lon. Region type is looked up from the nearest cell.
- The app calculates and displays `departure = max temp - normal temp`; the user never types a departure.
- The app builds a one-row table with the six raw input columns and calls the saved pipeline directly.
- **Outputs:** predicted severity, the vote breakdown of the K neighbours (obtained by passing the input through the pipeline's fitted preprocessing steps, then the KNN's neighbour lookup), and the rule-based label alongside for comparison. Extra tabs show the evaluation figures.

## 9. Testing and verification

- `labeling`: unit tests on boundary values for each region type (departure 4.4 / 4.5 / 6.4 / 6.5; max temp 44.9 / 45 / 46.9 / 47 C).
- Calendar: 1 March in a leap year and a non-leap year give the same `doy365`, `doy_sin` and `doy_cos`; 29 Feb is excluded from the normal.
- Pipeline: `departure` computed inside the pipeline matches the stored column; scaler statistics come only from training rows; no test-year row appears in training or in the normal.
- Artifact: a freshly loaded artifact in a new process gives identical predictions to the notebook pipeline on a fixed set of inputs; the app path gives the same results.
- The whole flow runs from raw files to artifact with documented commands.

## 10. Report requirements

1. Name the rules the **Operational Heatwave Severity Rules (Project Approximation)** and list every approximation in 6.3.
2. Include the accuracy explanation in Section 4.
3. Include a **class-count table** (train and test, per class) right after labeling. Accuracy is not the headline metric: macro-F1, per-class precision/recall/F1 and Severe recall are, next to the majority baseline.
4. State that the normal differs from IMD's official normal (6.2).

## 11. Risks and open items

- **Severe class may be very small** in Maharashtra over a 3-year test window. Person 1 reports counts right after labeling. If the Severe test count is too small for stable metrics, the team chooses between a wider test window and reporting Severe with an explicit caveat.
- **Coarse grid** means few cells in Maharashtra. The number of cells is reported after data inspection.
- **2025 data availability** is unconfirmed.
- **Label rules are unverified** against IMD's published criteria. Accepted and disclosed (Sections 4 and 10).
- **Region assignment rule** for plains / coastal / hilly is not yet defined; Person 1 proposes it in the first tasks.
- Other models (Random Forest, K-Means hotspots, forecasting) are out of scope.

## 12. Global constraints

- Python with scikit-learn, pandas, Streamlit; versions pinned.
- No use of test-year data in fitting, scaling or normals.
- All figures in the report must be reproducible from the notebook.

## 13. Order of work

1. Person 1: data inspection (resolution, file layout, cell count, 2025 availability).
2. Person 1: normals, labels, class counts; hand off `features.parquet`.
3. Person 2: pipeline, baseline KNN, grid search, evaluation and noise tests.
4. Person 3: app skeleton against a stub artifact from the start, then the real one; report drafted alongside.
