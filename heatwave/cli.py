import argparse
import json
from pathlib import Path
from .data_loader import load_years, proposed_region_lookup
from .climatology import compute_normals, build_features
from .model import split_by_year, tune_model, save_artifact
from .evaluate import class_counts
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    root = args.root
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        observations, cells = load_years(root / "data/raw", root / "data/boundaries/maharashtra.geojson", end=args.end_year)
        lookup_path = root / "data/region_lookup.csv"
        if not lookup_path.exists():
            proposed_region_lookup(cells).to_csv(lookup_path, index=False)
        lookup = pd.read_csv(lookup_path)
        cells.merge(lookup, on="cell_id", validate="one_to_one").to_csv(processed / "cells.csv", index=False)
        normals = compute_normals(observations)
        normals.to_parquet(processed / "normals.parquet", index=False)
        features, dropped = build_features(observations, normals, lookup, args.end_year)
        features.to_parquet(processed / "features.parquet", index=False)
        train, test = split_by_year(features)
        class_counts(train, test).to_csv(processed / "class_counts.csv")
        inspection = {"source": "IMD 1-degree annual Tmax GRD", "end_year": args.end_year, "grid_cells": len(cells), "baseline_years": [1991, 2014], "train_years": [2015, 2022], "test_years": sorted(pd.to_datetime(test.date).dt.year.unique().tolist()), "missing_model_rows_dropped": dropped, "baseline_missing_readings": int(observations.loc[observations.date.dt.year <= 2014, "max_temp"].isna().sum()), "rows": len(features), "byte_order": "little-endian", "region_rule_status": "Project geographic proxy; review data/region_lookup.csv"}
        (processed / "inspection.json").write_text(json.dumps(inspection, indent=2))
        print(json.dumps(inspection, indent=2))
        print(class_counts(train, test))
    else:
        features = pd.read_parquet(processed / "features.parquet")
        train, _ = split_by_year(features)
        search = tune_model(train, n_jobs=args.jobs)
        save_artifact(search.best_estimator_, root / "models/knn_heatwave.joblib", source="IMD", best_params=search.best_params_, training_years=list(range(2015, 2023)))
        pd.DataFrame(search.cv_results_).to_csv(processed / "grid_search.csv", index=False)
        print(search.best_params_)
        print(f"LOYO macro-F1: {search.best_score_:.4f}")


if __name__ == "__main__":
    main()
