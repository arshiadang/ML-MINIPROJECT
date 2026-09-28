"""Prediction path shared by the app and integration tests."""
import numpy as np
import pandas as pd
from . import INPUT_COLUMNS, LABEL_MAP
from .labeling import label_observations
from .model import neighbor_breakdown


def nearest_cell(cells, lat, lon):
    if not np.isfinite([lat, lon]).all():
        raise ValueError("Coordinates must be finite")
    latitude = np.radians(cells.lat.to_numpy())
    longitude = np.radians(cells.lon.to_numpy())
    a = np.sin((latitude - np.radians(lat))/2)**2 + np.cos(latitude)*np.cos(np.radians(lat))*np.sin((longitude - np.radians(lon))/2)**2
    distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    idx = np.argmin(distance)
    return cells.iloc[idx], float(distance[idx])


def predict_observation(artifact, cells, date, lat, lon, max_temp, normal_temp):
    cell, distance = nearest_cell(cells, lat, lon)
    raw = pd.DataFrame([[date, lat, lon, cell.region_type, max_temp, normal_temp]], columns=INPUT_COLUMNS)
    pipeline = artifact["pipeline"]
    prediction = int(pipeline.predict(raw)[0])
    breakdown, neighbors = neighbor_breakdown(pipeline, raw)
    return {"prediction": LABEL_MAP[prediction], "rule_label": LABEL_MAP[int(label_observations(raw).iloc[0])], "departure": max_temp-normal_temp, "cell_id": cell.cell_id, "region_type": cell.region_type, "distance_km": distance, "breakdown": breakdown, "neighbors": neighbors}
