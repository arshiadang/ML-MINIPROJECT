"""Pooled +/-7-day cell normals from 1991-2014 only."""
import numpy as np
import pandas as pd
from .calendar import doy365
from .labeling import label_observations


def compute_normals(observations):
    dates = pd.to_datetime(observations.date)
    baseline = observations.loc[dates.dt.year.between(1991, 2014) & ~((dates.dt.month == 2) & (dates.dt.day == 29))].copy()
    years = set(pd.to_datetime(baseline.date).dt.year)
    if years != set(range(1991, 2015)):
        raise ValueError("Normal requires all baseline years 1991-2014")
    if baseline.duplicated(["date", "cell_id"]).any():
        raise ValueError("Duplicate cell/day observations")
    baseline["doy365"] = doy365(baseline.date)
    grouped = baseline.groupby(["cell_id", "doy365"]).max_temp.agg(["sum", "count"])
    result = []
    for cell, group in grouped.groupby(level=0):
        daily = group.droplevel(0).reindex(range(1, 366), fill_value=0)
        sums, counts = daily["sum"].to_numpy(), daily["count"].to_numpy()
        # Sum and count separately: missing readings must not receive equal-day weight.
        pooled_sum = sum(np.roll(sums, shift) for shift in range(-7, 8))
        pooled_count = sum(np.roll(counts, shift) for shift in range(-7, 8))
        mean = np.divide(pooled_sum, pooled_count, out=np.full(365, np.nan), where=pooled_count > 0)
        result.append(pd.DataFrame({"cell_id": cell, "doy365": range(1, 366), "normal_temp": mean, "normal_count": pooled_count}))
    return pd.concat(result, ignore_index=True)


def build_features(observations, normals, lookup, end_year=2025):
    dates = pd.to_datetime(observations.date)
    data = observations.loc[dates.dt.year.between(2015, end_year) & dates.dt.month.between(3, 6)].copy()
    if data.duplicated(["date", "cell_id"]).any():
        raise ValueError("Duplicate model cell/day observations")
    data["doy365"] = doy365(data.date)
    data = data.merge(normals, on=["cell_id", "doy365"], validate="many_to_one")
    data = data.merge(lookup, on="cell_id", how="left", validate="many_to_one")
    if data.region_type.isna().any():
        raise ValueError("Region lookup does not cover all cells")
    missing = data[["max_temp", "normal_temp"]].isna().any(axis=1)
    dropped = int(missing.sum())
    data = data.loc[~missing].copy()
    data["departure"] = data.max_temp - data.normal_temp
    data["label"] = label_observations(data)
    columns = ["date", "cell_id", "lat", "lon", "region_type", "max_temp", "normal_temp", "departure", "label"]
    return data[columns].sort_values(["date", "cell_id"]).reset_index(drop=True), dropped
