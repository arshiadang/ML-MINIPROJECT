"""Acquire public IMD annual files and a geoBoundaries Maharashtra polygon."""
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
IMD = "https://www.imdpune.gov.in/cmpg/Griddata/maxtemp.php"


def download_year(year):
    path = ROOT / "data/raw" / f"{year}.GRD"
    expected = (366 if calendar.isleap(year) else 365) * 31 * 31 * 4
    for attempt in range(3):
        try:
            if not path.exists():
                response = requests.post(IMD, data={"maxtemp": year}, timeout=90)
                response.raise_for_status()
                if len(response.content) != expected:
                    raise ValueError(f"Unexpected download length for {year}: {len(response.content)}")
                path.write_bytes(response.content)
            content = path.read_bytes()
            if len(content) != expected:
                raise ValueError(f"Invalid cached file {path}")
            print(f"Downloaded/verified {year}", flush=True)
            return {"year": year, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "source": IMD}
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)


def boundary():
    api = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/"
    response = requests.get(api, timeout=60)
    response.raise_for_status()
    meta = response.json()
    (ROOT / "data/boundaries/source.json").write_text(json.dumps(meta, indent=2))
    response = requests.get(meta["gjDownloadURL"], timeout=120)
    response.raise_for_status()
    features = [f for f in response.json()["features"] if f["properties"].get("shapeISO") == "IN-MH"]
    if len(features) != 1:
        raise ValueError("Expected one Maharashtra feature")
    (ROOT / "data/boundaries/maharashtra.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download_year, range(1991, args.end_year + 1)))
    (ROOT / "data/raw/manifest.json").write_text(json.dumps({"byte_order": "little-endian float32", "files": records}, indent=2))
    boundary()
