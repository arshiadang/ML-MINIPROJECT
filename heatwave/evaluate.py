"""Reproducible metrics, error diagnostics, K curves and noise experiments."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline as sklearn_pipeline
from sklearn.preprocessing import StandardScaler
from . import INPUT_COLUMNS, LABEL_MAP
from .labeling import BASE_THRESHOLDS, label_observations
from .model import K_VALUES

COLORS = ["#237b77", "#e6a23c", "#c44c45"]


def scores(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    severe_count = int((truth == 2).sum())
    return {"macro_f1": f1_score(truth, prediction, labels=[0, 1, 2], average="macro", zero_division=0), "severe_recall": recall_score(truth, prediction, labels=[2], average=None, zero_division=0)[0] if severe_count else np.nan, "accuracy": accuracy_score(truth, prediction), "severe_support": severe_count}


def class_counts(train, test):
    return pd.DataFrame({"train": train.label.value_counts().reindex([0, 1, 2], fill_value=0), "test": test.label.value_counts().reindex([0, 1, 2], fill_value=0)}).rename(index=LABEL_MAP)


def evaluate_test(pipeline, train, test):
    prediction = pipeline.predict(test[INPUT_COLUMNS])
    majority = int(train.label.mode().iloc[0])
    baseline = np.full(len(test), majority)
    comparison = pd.DataFrame({"KNN": scores(test.label, prediction), "Majority baseline": scores(test.label, baseline)}).T
    reports = {}
    for name, pred in [("KNN", prediction), ("Majority baseline", baseline)]:
        reports[name] = pd.DataFrame(classification_report(test.label, pred, labels=[0, 1, 2], target_names=list(LABEL_MAP.values()), output_dict=True, zero_division=0)).T
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(confusion_matrix(test.label, prediction, labels=[0, 1, 2]), display_labels=list(LABEL_MAP.values())).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Held-out years: KNN confusion matrix")
    fig.tight_layout()
    return comparison, reports, fig, prediction


def plot_k_curves(cv_results, metric="euclidean", weights="uniform"):
    frame = pd.DataFrame(cv_results)
    selected = frame[(frame.param_knn__metric == metric) & (frame.param_knn__weights == weights)].sort_values("param_knn__n_neighbors")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, score, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro-F1"]):
        k = selected.param_knn__n_neighbors.astype(int)
        ax.plot(k, selected[f"mean_train_{score}"], "o-", label="Training folds")
        ax.plot(k, selected[f"mean_test_{score}"], "o-", label="Leave-one-year-out")
        ax.fill_between(k, selected[f"mean_test_{score}"] - selected[f"std_test_{score}"], selected[f"mean_test_{score}"] + selected[f"std_test_{score}"], alpha=.15)
        ax.set(xlabel="K", ylabel=title, ylim=(0, 1.02), xticks=K_VALUES)
        ax.legend()
    fig.suptitle(f"K curves: {metric}, {weights} (band: fold standard deviation)")
    fig.tight_layout()
    return fig


def decision_regions(train, region="plains", k=5, metric="euclidean", weights="uniform"):
    subset = train.loc[train.region_type == region]
    if len(subset) < k:
        raise ValueError("Not enough training samples in chosen region")
    features = ["max_temp", "departure"]
    model = sklearn_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)).fit(subset[features], subset.label)
    x = np.linspace(subset.max_temp.min()-1, subset.max_temp.max()+1, 220)
    y = np.linspace(subset.departure.min()-1, subset.departure.max()+1, 220)
    xx, yy = np.meshgrid(x, y)
    zz = model.predict(pd.DataFrame({"max_temp": xx.ravel(), "departure": yy.ravel()})).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.contourf(xx, yy, zz, levels=[-.5, .5, 1.5, 2.5], cmap=ListedColormap(COLORS), alpha=.35)
    sample = subset.sample(min(1200, len(subset)), random_state=42)
    for label, name in LABEL_MAP.items():
        points = sample[sample.label == label]
        ax.scatter(points.max_temp, points.departure, s=7, c=COLORS[label], label=name, alpha=.5)
    ax.axhline(4.5, color="grey", ls="--", lw=1)
    ax.axhline(6.5, color="grey", ls="--", lw=1)
    ax.axvline(BASE_THRESHOLDS[region], color="grey", ls=":", lw=1)
    ax.set(xlabel="Maximum temperature (°C)", ylabel="Departure (°C)", title=f"{region.title()}: separate 2-feature KNN, training data only")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def boundary_errors(test, prediction):
    data = test.copy()
    d = data.max_temp - data.normal_temp
    base = data.region_type.map(BASE_THRESHOLDS)
    distances = np.column_stack([abs(data.max_temp-base), abs(d-4.5), abs(d-6.5), np.where(data.normal_temp >= 40, abs(data.max_temp-45), np.inf), np.where(data.normal_temp >= 40, abs(data.max_temp-47), np.inf), abs(data.normal_temp-40)])
    data["threshold_distance"] = distances.min(axis=1)
    data["prediction"] = prediction
    data["error"] = data.label != prediction
    data["threshold_band"] = pd.cut(data.threshold_distance, [-.001, .25, .5, 1, 2, np.inf])
    summary = data.groupby("threshold_band", observed=False).error.agg(["size", "sum", "mean"]).rename(columns={"size": "observations", "sum": "errors", "mean": "error_rate"})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary.index.astype(str), summary.error_rate, color="#237b77")
    ax.set(xlabel="Distance to nearest candidate rule threshold (°C)", ylabel="Error rate", title="Errors near rule thresholds (diagnostic proxy)")
    fig.tight_layout()
    return data, summary, fig


def noise_robustness(pipeline, train, test, ks=K_VALUES, sigmas=(.5, 1, 2), seeds=(11, 23, 37, 51, 71)):
    rows = []
    # Same perturbation for every K permits a paired comparison.
    for k in ks:
        fitted = clone(pipeline).set_params(knn__n_neighbors=k).fit(train[INPUT_COLUMNS], train.label)
        for sigma in sigmas:
            for seed in seeds:
                noisy = test[INPUT_COLUMNS].copy()
                noisy["max_temp"] += np.random.default_rng(seed).normal(0, sigma, len(noisy))
                changed_truth = label_observations(noisy)
                prediction = fitted.predict(noisy)
                for target, truth, pred in [("Original labels", test.label, prediction), ("Recomputed rules", changed_truth, prediction), ("Rules vs original", test.label, changed_truth)]:
                    rows.append({"k": k, "sigma": sigma, "seed": seed, "comparison": target, **scores(truth, pred)})
    return pd.DataFrame(rows)


def plot_noise(noise):
    mean = noise.groupby(["k", "sigma", "comparison"])[["macro_f1", "severe_recall"]].mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey="row")
    for col, sigma in enumerate(sorted(mean.sigma.unique())):
        for row, metric in enumerate(["macro_f1", "severe_recall"]):
            ax = axes[row, col]
            for label, group in mean[mean.sigma == sigma].groupby("comparison", sort=False):
                ax.plot(group.k, group[metric], "--" if label == "Rules vs original" else "-", label=label)
            ax.set(title=f"σ = {sigma:g} °C", xlabel="K", ylabel=metric.replace("_", " "), ylim=(-.03, 1.02) if row else (0, 1.02))
            if row == 1 and mean.loc[(mean.sigma == sigma) & (mean.comparison == "Original labels"), metric].isna().all():
                ax.text(.5, .55, "Original-label recall: N/A\nNo Severe cases in original test labels", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="dimgray")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Test Tmax noise: five fixed seeds; metric and weighting fixed at selected values")
    fig.tight_layout()
    return mean, fig
