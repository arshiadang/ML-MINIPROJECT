from datetime import date
from pathlib import Path
import json
import pandas as pd
from shapely.geometry import Point
import streamlit as st
from heatwave import RULE_NAME
from heatwave.data_loader import read_boundary
from heatwave.demo import predict_observation
from heatwave.model import load_artifact

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Maharashtra Heatwave Lab", page_icon="☀️", layout="wide")
st.title("Maharashtra Heatwave Lab")
st.caption("Machine Learning · KNN · Same-day severity classification · March–June")
st.info("Labels use the Operational Heatwave Severity Rules (Project Approximation). This classroom model approximates rules using their own input variables. It does not establish forecasting capability.")

@st.cache_resource
def resources(artifact_mtime, cells_mtime):
    return load_artifact(ROOT / "models/knn_heatwave.joblib"), pd.read_csv(ROOT / "data/processed/cells.csv")

artifact_path = ROOT / "models/knn_heatwave.joblib"
cells_path = ROOT / "data/processed/cells.csv"
if not artifact_path.exists() or not cells_path.exists():
    st.warning("Model is not ready. Run the data preparation and analysis notebook using the README commands.")
    st.stop()
try:
    artifact, cells = resources(artifact_path.stat().st_mtime, cells_path.stat().st_mtime)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()
if 2 not in artifact["pipeline"].classes_:
    st.warning("No Severe Heatwave examples occur in the training data. This fitted KNN cannot predict Severe Heatwave; compare the project rule label and read the evaluation limitation.")
predict_tab, evaluation_tab, method_tab = st.tabs(["Classify a day", "Evaluation", "Method"])
with predict_tab:
    year = st.selectbox("Year", list(range(2025, 2014, -1)))
    with st.form("observation"):
        left, right = st.columns(2)
        with left:
            observation_date = st.date_input("Date (March–June)", value=date(year, 5, 15), min_value=date(year, 3, 1), max_value=date(year, 6, 30))
            max_temp = st.number_input("Maximum temperature (°C)", min_value=0., max_value=60., value=43., step=.1)
            normal_temp = st.number_input("Normal maximum temperature (°C)", min_value=0., max_value=55., value=37., step=.1)
        with right:
            lat = st.number_input("Latitude (°N)", min_value=15., max_value=23., value=18.5, step=.01)
            lon = st.number_input("Longitude (°E)", min_value=72., max_value=81., value=75.5, step=.01)
            st.caption("Region type comes from the nearest Maharashtra grid cell. Use that cell's climatological normal when available.")
            st.metric("Departure (°C)", f"{max_temp-normal_temp:+.1f}")
        submitted = st.form_submit_button("Classify observation", type="primary")
    if submitted:
        if not 3 <= observation_date.month <= 6:
            st.error("Choose a date between 1 March and 30 June.")
        elif not read_boundary(ROOT / "data/boundaries/maharashtra.geojson").covers(Point(lon, lat)):
            st.error("Choose a location inside Maharashtra.")
        else:
            result = predict_observation(artifact, cells, observation_date.replace(year=year), lat, lon, max_temp, normal_temp)
            a, b, c = st.columns(3)
            a.metric("KNN severity", result["prediction"])
            b.metric("Project rule label", result["rule_label"])
            c.metric("Departure (°C)", f"{result['departure']:+.1f}")
            st.caption(f"Nearest cell {result['cell_id']} · {result['region_type']} · {result['distance_km']:.1f} km away")
            st.subheader("Neighbour vote breakdown")
            st.dataframe(result["breakdown"], use_container_width=True)
            st.caption("Vote shares reflect the model's weighting. They are not calibrated probabilities.")
            with st.expander("Inspect K neighbours"):
                st.dataframe(result["neighbors"], use_container_width=True)
with evaluation_tab:
    report_dir = ROOT / "reports/results"
    if (report_dir / "metrics.csv").exists():
        st.dataframe(pd.read_csv(report_dir / "metrics.csv", index_col=0), use_container_width=True)
        st.dataframe(pd.read_csv(ROOT / "data/processed/class_counts.csv", index_col=0))
        for name in ["confusion_matrix", "k_curves", "k_curves_selected", "decision_regions", "boundary_errors", "noise_robustness"]:
            path = report_dir / f"{name}.png"
            if path.exists():
                st.image(str(path))
    else:
        st.info("Execute notebooks/analysis.ipynb to generate evaluation figures.")
with method_tab:
    st.subheader(RULE_NAME)
    st.markdown("Base temperatures: plains 40°C, coastal 37°C, hilly 30°C. At or above the base, departure ≥4.5°C gives Heatwave and ≥6.5°C gives Severe Heatwave. When the normal is ≥40°C, Tmax ≥45°C gives Heatwave and ≥47°C gives Severe. The higher severity wins.")
    st.markdown("Normals pool a ±7-day window over 1991–2014, excluding 29 February. Training uses 2015–2022, testing uses 2023–2025. Tuning leaves out one training year at a time. The geographic region lookup is a project proxy, not an IMD region classification.")
    st.json(artifact.get("metadata", {}))
