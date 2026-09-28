"""Operational Heatwave Severity Rules (Project Approximation)."""
import numpy as np
import pandas as pd

BASE_THRESHOLDS = {"plains": 40.0, "coastal": 37.0, "hilly": 30.0}


def label_observations(frame):
    """Integer severity, with a base-temperature gate and highest severity wins."""
    if not frame.region_type.isin(BASE_THRESHOLDS).all():
        raise ValueError("region_type must be plains, coastal or hilly")
    t = frame.max_temp.to_numpy(dtype=float)
    n = frame.normal_temp.to_numpy(dtype=float)
    if not np.isfinite(np.column_stack([t, n])).all():
        raise ValueError("Temperatures must be finite")
    d = t - n
    eligible = t >= frame.region_type.map(BASE_THRESHOLDS).to_numpy()
    departure_rule = np.select([d >= 6.5, d >= 4.5], [2, 1], default=0)
    absolute_rule = np.select([(n >= 40) & (t >= 47), (n >= 40) & (t >= 45)], [2, 1], default=0)
    return pd.Series(np.where(eligible, np.maximum(departure_rule, absolute_rule), 0), index=frame.index, name="label")
