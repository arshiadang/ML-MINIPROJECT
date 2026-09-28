"""IMD annual 1-degree Tmax binary parser and Maharashtra cell selection."""
import calendar
import json
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape


def read_boundary(path):
    obj = json.loads(Path(path).read_text())
    if obj["type"] == "FeatureCollection":
        matches = [f for f in obj["features"] if f.get("properties", {}).get("shapeISO") == "IN-MH" or "maharashtra" in str(f.get("properties", {})).lower()]
        if len(matches) != 1:
            raise ValueError("Boundary must contain exactly one Maharashtra feature")
        obj = matches[0]
    geometry = shape(obj.get("geometry", obj))
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Invalid Maharashtra boundary")
    return geometry


def grid_cells(boundary_path):
    boundary = read_boundary(boundary_path)
    rows = []
    for i, lat in enumerate(np.arange(7.5, 38, 1)):
        for j, lon in enumerate(np.arange(67.5, 98, 1)):
            if boundary.covers(Point(lon, lat)):
                rows.append(dict(cell_id=f"{lat:.1f}_{lon:.1f}", lat=lat, lon=lon, i=i, j=j))
    if not rows:
        raise ValueError("No IMD grid centres inside boundary")
    return pd.DataFrame(rows)


def read_grd(path, year, cells, byte_order="<"):
    """Float32, day/latitude/longitude order; 99.9 is the IMD missing sentinel.

    Byte order is explicit because the source documentation does not state it.
    The acquisition manifest records the chosen order and range checks.
    """
    if byte_order not in ("<", ">"):
        raise ValueError("byte_order must be < or >")
    days = 366 if calendar.isleap(year) else 365
    raw = np.fromfile(path, dtype=f"{byte_order}f4")
    if raw.size != days * 31 * 31:
        raise ValueError(f"{path}: expected {days * 31 * 31 * 4} bytes, got {raw.nbytes}")
    values = raw.reshape(days, 31, 31)[:, cells.i, cells.j].astype(float)
    values[np.isclose(values, 99.9, atol=0.001)] = np.nan
    finite = values[np.isfinite(values)]
    if not finite.size or np.any((finite < -50) | (finite > 65)) or np.nanmax(np.abs(finite)) < 1:
        raise ValueError("Implausible temperatures: check byte order/layout")
    dates = pd.date_range(f"{year}-01-01", periods=days)
    out = pd.DataFrame({"date": np.repeat(dates, len(cells)), "cell_id": np.tile(cells.cell_id, days), "max_temp": values.ravel()})
    return out.merge(cells[["cell_id", "lat", "lon"]], on="cell_id", validate="many_to_one")


def load_years(raw_dir, boundary_path, start=1991, end=2025, byte_order="<"):
    cells = grid_cells(boundary_path)
    frames = [read_grd(Path(raw_dir) / f"{year}.GRD", year, cells, byte_order) for year in range(start, end + 1)]
    return pd.concat(frames, ignore_index=True), cells


def proposed_region_lookup(cells):
    """Transparent geographic proxy, not terrain- or IMD-validated.

    Coastal: longitude <74 E. Hilly: 74<=longitude<75 E and latitude<20 N.
    Plains: remaining cells. Editable lookup is the source of truth thereafter.
    """
    region = np.select([cells.lon < 74, (cells.lon < 75) & (cells.lat < 20)], ["coastal", "hilly"], default="plains")
    return pd.DataFrame({"cell_id": cells.cell_id, "region_type": region})
