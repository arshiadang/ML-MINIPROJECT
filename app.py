"""Local classroom demo. Predictions always use the saved shared pipeline."""
from datetime import date
from pathlib import Path
import json
import html
import pandas as pd
import pydeck as pdk
from shapely.geometry import Point
import streamlit as st
from heatwave import RULE_NAME
from heatwave.data_loader import read_boundary
from heatwave.demo import predict_observation, nearest_cell
from heatwave.model import load_artifact

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports/results"
st.set_page_config(page_title="Heatwave Lab · Maharashtra", page_icon="☀️", layout="wide")
st.markdown('''<style>
.stApp {background:#F7F8F4}
.block-container {max-width:1280px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3 {letter-spacing:-.035em}
[data-testid="stSidebar"] {background:#e9eee6;border-right:1px solid #d5dfd2}
[data-testid="stMetric"] {background:#fff;border:1px solid #dbe3d7;border-radius:14px;padding:18px}
[data-testid="stMetricValue"] {font-size:1.8rem}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:16px}
.stButton>button {border-radius:10px;font-weight:600;min-height:44px}
.hero {background:#173e38;color:#fff;padding:32px 36px;border-radius:20px;margin:0 0 25px}
.hero h1 {color:#fff;font-size:2.65rem;margin:4px 0 8px;padding:0;line-height:1.16}
.hero p {color:#d3e3d6;max-width:750px;margin:0;line-height:1.6}
.eyebrow {font-size:.73rem;letter-spacing:.17em;text-transform:uppercase;font-weight:700;color:#adcfb5;margin-bottom:10px}
.result {padding:25px;border-radius:16px;background:#e5f0e7;border:1px solid #c5ddc9;margin:12px 0 18px}
.result.heat {background:#fff1d9;border-color:#e5ca91}
.result .label {font-size:.8rem;text-transform:uppercase;letter-spacing:.1em}
.result h2 {font-size:2.2rem;margin:4px 0 8px;padding:0;color:#183d38}
.result p {margin:0;color:#496259}
.empty {padding:38px 28px;background:#f0f3eb;border:1px dashed #b5c7b2;border-radius:16px;margin-top:18px;line-height:1.7}
.small-note {color:#66786e;font-size:.83rem}
@media(max-width:850px) {[data-testid="stHorizontalBlock"] {flex-wrap:wrap} [data-testid="stColumn"] {width:100% !important;flex:1 1 100% !important;min-width:0 !important}}
@media(max-width:640px) {.hero{padding:24px}.hero h1{font-size:2rem}.block-container{padding-top:1.2rem}}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def resources(artifact_mtime, cells_mtime):
    return load_artifact(ROOT / "models/knn_heatwave.joblib"), pd.read_csv(ROOT / "data/processed/cells.csv")

@st.cache_data
def read_csv(path, modified):
    return pd.read_csv(path)


def csv(name):
    path = OUT / name
    return read_csv(str(path), path.stat().st_mtime)


def figure(name, caption=None):
    path = OUT / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info("This figure is not available. Run the analysis notebook to generate it.")


def region_deck(cells, selected=None):
    colors = {"coastal": [22,139,139], "hilly": [173,118,61], "plains": [97,125,186]}
    points = cells.copy()
    points["color"] = points.region_type.map(colors)
    geometry = json.loads((ROOT / "data/boundaries/maharashtra.geojson").read_text())
    layers = [pdk.Layer("GeoJsonLayer", geometry, filled=True, stroked=True, get_fill_color=[231,237,221,180], get_line_color=[94,117,103], line_width_min_pixels=1),
              pdk.Layer("ScatterplotLayer", points, get_position="[lon, lat]", get_fill_color="color", get_radius=6500, radius_min_pixels=5, pickable=True)]
    if selected is not None:
        layers.append(pdk.Layer("ScatterplotLayer", pd.DataFrame([{"lat": selected[0], "lon": selected[1], "cell_id": "Your location", "region_type": "Entered coordinates"}]), get_position="[lon, lat]", get_fill_color=[201,72,61], get_radius=4500, radius_min_pixels=5, pickable=True))
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=19, longitude=76.7, zoom=5.5), map_style=None, tooltip={"text":"{cell_id}\n{region_type}"})


artifact_path = ROOT / "models/knn_heatwave.joblib"
cells_path = ROOT / "data/processed/cells.csv"
if not artifact_path.exists() or not cells_path.exists():
    st.title("Maharashtra Heatwave Lab")
    st.info("Model is not ready. Follow the preparation commands in README.md.")
    st.stop()
try:
    artifact, cells = resources(artifact_path.stat().st_mtime, cells_path.stat().st_mtime)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()
knn = artifact["pipeline"].named_steps["knn"]
missing_severe = 2 not in artifact["pipeline"].classes_
with st.sidebar:
    st.markdown("## ☀ Heatwave Lab")
    st.caption("MAHARASHTRA / CLIMATE CLASSROOM")
    page = st.radio("Explore", ["Classify a day", "Model evidence", "Region map", "Method & data"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Model snapshot**")
    st.caption(f"K = {knn.n_neighbors} · {knn.metric.title()}\n\n{knn.weights.title()} neighbour voting")
    st.markdown("**Data window**")
    st.caption("26 Maharashtra cells\n\nMarch–June, 2015–2025\n\nTrain 2015–2022 · Test 2023–2025")
    st.divider()
    st.caption("A same-day rule approximation. No live weather feed or independent forecasting claim.")

if page == "Classify a day":
    st.markdown('''<div class="hero"><div class="eyebrow">A closer look at a hotter day</div><h1>How does this day classify?</h1><p>Explore a Maharashtra observation. Compare the KNN prediction with our project rules, then see the neighbours behind the vote.</p></div>''', unsafe_allow_html=True)
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.subheader("01 / Weather observation")
        st.caption("Choose a sample or enter your own readings. Values are illustrative until you change them.")
        def preset(t, n):
            st.session_state["tmax"] = t
            st.session_state["normal"] = n
        samples = st.columns(3)
        samples[0].button("Typical day", on_click=preset, args=(36.,37.), use_container_width=True)
        samples[1].button("Heatwave case", on_click=preset, args=(43.,37.), use_container_width=True)
        samples[2].button("Severe rule case", on_click=preset, args=(48.,40.), use_container_width=True)
        with st.container(border=True):
            a,b = st.columns([1,2])
            year = a.selectbox("Year", list(range(2025,2014,-1)))
            observation_date = b.date_input("Date · March–June", value=date(year,5,15), min_value=date(year,3,1), max_value=date(year,6,30), key=f"date_{year}")
            a,b = st.columns(2)
            lat = a.number_input("Latitude (°N)", min_value=15.,max_value=23.,value=18.5,step=.01,format="%.2f")
            lon = b.number_input("Longitude (°E)", min_value=72.,max_value=81.,value=75.5,step=.01,format="%.2f")
            cell, distance = nearest_cell(cells,lat,lon)
            st.caption(f"Nearest grid cell: {cell.cell_id} · {cell.region_type.title()} proxy · {distance:.1f} km")
            st.session_state.setdefault("tmax",43.)
            st.session_state.setdefault("normal",37.)
            a,b = st.columns(2)
            max_temp = a.number_input("Maximum temperature (°C)",min_value=0.,max_value=60.,step=.1,key="tmax")
            normal_temp = b.number_input("Normal maximum temperature (°C)",min_value=0.,max_value=55.,step=.1,key="normal")
            st.caption("Normal temperature is the climatological reference, not today's minimum temperature.")
            departure = max_temp-normal_temp
            st.markdown(f"**Calculated departure: {departure:+.1f}°C**")
            submitted = st.button("Classify observation →",type="primary",use_container_width=True)
        st.caption("Inputs stay local. Region type is assigned by the nearest grid cell using the documented geographic proxy.")
    signature = (str(observation_date),lat,lon,max_temp,normal_temp)
    if submitted:
        if not 3 <= observation_date.month <= 6:
            st.error("Choose a date from 1 March to 30 June.")
            st.session_state.pop("prediction",None)
        elif not read_boundary(ROOT / "data/boundaries/maharashtra.geojson").covers(Point(lon,lat)):
            st.error("This location falls outside Maharashtra. Choose a point inside the state.")
            st.session_state.pop("prediction",None)
        else:
            st.session_state["prediction"] = (signature,predict_observation(artifact,cells,observation_date,lat,lon,max_temp,normal_temp))
    with right:
        st.subheader("02 / Classification & neighbours")
        saved = st.session_state.get("prediction")
        if saved and saved[0] == signature:
            result = saved[1]
            cls = "heat" if result['prediction'] != 'Normal' else ''
            st.markdown(f'<div class="result {cls}"><div class="label">KNN prediction</div><h2>{html.escape(result["prediction"])}</h2><p>{knn.n_neighbors} neighbour · {knn.metric} distance · {knn.weights} voting</p></div>',unsafe_allow_html=True)
            a,b = st.columns(2)
            a.metric("Project rule label",result["rule_label"])
            b.metric("Departure",f"{result['departure']:+.1f}°C")
            if result["prediction"] != result["rule_label"]:
                st.warning("Model and rule disagree. The rule label is calculated directly from your inputs; the model can only vote among classes present in training.")
            else:
                st.success("The model and project rule agree for these inputs.")
            st.markdown("**Neighbour votes**")
            breakdown = result["breakdown"].reset_index(names="Severity").rename(columns={"neighbors":"Neighbours","vote_share":"Vote share"})
            st.dataframe(breakdown,hide_index=True,use_container_width=True,column_config={"Vote share":st.column_config.ProgressColumn("Vote share",min_value=0,max_value=1,format="%.2f")})
            st.caption("Vote shares reflect neighbour weighting, not calibrated confidence. At K=1, one neighbour supplies the entire vote.")
            with st.expander("Inspect distances and neighbour classes"):
                st.dataframe(result["neighbors"],hide_index=True,use_container_width=True)
        else:
            message = "Your inputs changed. Classify again to refresh the result." if saved else "Enter an observation and select Classify observation to see the prediction, rule comparison and neighbour votes."
            st.markdown(f'<div class="empty"><strong>Your result will appear here</strong><br>{message}</div>',unsafe_allow_html=True)
        if missing_severe:
            st.warning("Severe class unavailable in KNN: training contains zero Severe examples. The project rule can still label an input Severe.")
        with st.expander("Where is this observation?"):
            st.pydeck_chart(region_deck(cells,(lat,lon)),use_container_width=True)
            st.caption("Red: entered location. Teal: coastal proxy. Brown: hilly proxy. Blue: plains proxy. Boundary map only; no terrain layer.")
    st.caption(RULE_NAME + ". Labels derive from the model's inputs; this demonstrates rule approximation, not independent forecasting.")

elif page == "Model evidence":
    st.markdown('''<div class="hero"><div class="eyebrow">Held-out evidence / 2023–2025</div><h1>Read the scores in context.</h1><p>Rare-class performance matters more than overall accuracy. These results come from the saved analysis, with no fitting in the app.</p></div>''',unsafe_allow_html=True)
    if not (OUT / "metrics.csv").exists():
        st.info("Execute the analysis notebook to generate evaluation evidence.")
        st.stop()
    per = csv("classification_report.csv").set_index("Unnamed: 0")
    hw = per.loc["Heatwave"]
    two = per.loc[["Normal","Heatwave"],"f1-score"].mean()
    three = per.loc["macro avg","f1-score"]
    a,b,c,d = st.columns(4)
    a.metric("Heatwave F1",f"{hw['f1-score']:.2f}")
    b.metric("Two-class macro-F1",f"{two:.2f}")
    c.metric("Three-class macro-F1",f"{three:.2f}")
    d.metric("Severe recall","N/A",help="No Severe observations in the original test labels.")
    st.info("CV macro-F1 0.5533 averages eight held-out training-year scores from seven-year fits. Test macro-F1 0.5999 scores pooled 2023–2025 observations after fitting all eight training years. Both use the same three-class definition; different years, training sizes and aggregation explain why they need not match.")
    st.caption("0.80 scores Heatwave alone. 0.90 averages Normal and Heatwave. 0.60 includes a zero contribution for unsupported Severe under the fixed three-class scoring convention.")
    perf,kview,noise,errors = st.tabs(["Performance","K selection","Reading noise","Error locations"])
    with perf:
        st.markdown(f"**Heatwave recall {hw['recall']:.1%} · precision {hw['precision']:.1%} · 21 test cases.** One missed case changes recall by 4.76 percentage points.")
        a,b = st.columns([1.35,1])
        with a: figure("confusion_matrix")
        with b:
            st.markdown("**Class support**")
            st.dataframe(csv("class_counts.csv"),hide_index=True,use_container_width=True)
            st.markdown("**Majority-class comparison**")
            metrics = csv("metrics.csv").rename(columns={"Unnamed: 0":"Model","macro_f1":"Three-class macro-F1","accuracy":"Accuracy","severe_recall":"Severe recall","severe_support":"Severe cases"})
            st.dataframe(metrics,hide_index=True,use_container_width=True)
            st.caption("The baseline always predicts the most frequent training class. High accuracy alone would mask missed Heatwaves.")
    with kview:
        figure("k_curves")
        st.markdown("**K=1 is the observed peak, not a flat-curve tie.** Euclidean/uniform macro-F1 is 0.553342 at K=1 versus 0.535592 at K=5. Across paired year folds, K=1 wins three and ties five.")
        st.caption("The gap is 0.017749, with fold variation around 0.10. This does not establish a statistically distinct optimum. Euclidean beats Manhattan by only 0.000260 at K=1; uniform and distance voting tie with one neighbour.")
        if (OUT/"k_fold_review.csv").exists():
            with st.expander("Eight paired year folds"):
                st.dataframe(csv("k_fold_review.csv"),hide_index=True,use_container_width=True)
    with noise:
        st.markdown("**Noise can create Severe rule labels. The trained KNN still cannot predict them.**")
        st.caption("Only test maximum temperature changes. Normal stays fixed; departure and project labels are recomputed. Five fixed seeds per noise level.")
        figure("noise_robustness")
        if (OUT/"noise_severe_support.csv").exists():
            st.dataframe(csv("noise_severe_support.csv"),hide_index=True,use_container_width=True)
        st.info("Recall against noisy Severe labels is 0.0000 whenever those labels exist. At σ=0.5°C, seed 23 has none, so the mean uses four defined recalls. Against original labels, Severe recall is always N/A.")
    with errors:
        figure("boundary_errors")
        st.markdown("**All eight errors fall within 0.25°C of a candidate threshold.** Five Heatwave cases are missed and three Normal cases are false alarms. The largest recorded threshold distance among errors is 0.0702°C.")
        st.caption("Candidate-threshold distance is a diagnostic proxy, not exact distance to the active decision boundary.")
        with st.expander("Separate two-feature decision-region illustration"):
            figure("decision_regions","Plot-only KNN fitted on plains training observations. This is not a slice of the deployed pipeline.")
    st.download_button("Download the written report",(ROOT/"reports/report.md").read_text(),file_name="heatwave_report.md",mime="text/markdown")

elif page == "Region map":
    st.markdown('''<div class="hero"><div class="eyebrow">26 cells / one explicit approximation</div><h1>Put the region lookup on the map.</h1><p>Inspect the assignment used by the existing model. Colours show project coordinate bands, not verified terrain categories.</p></div>''',unsafe_allow_html=True)
    a,b,c = st.columns(3)
    for col,region in zip([a,b,c],["coastal","hilly","plains"]):
        col.metric(region.title()+" proxy",str(int((cells.region_type==region).sum()))+" cells")
    st.pydeck_chart(region_deck(cells),use_container_width=True)
    st.caption("Hover over a cell to see its ID and region. Teal: coastal. Brown: hilly. Blue: plains. Boundary: geoBoundaries / DataMeet, CC BY 2.5 India. No basemap or elevation layer.")
    st.markdown("**Map review:** all 26 centres fall inside the Maharashtra polygon and agree with the stored lookup. The cutoff bands are visibly coarse. The northern coastal-assigned cell (20.5°N, 73.5°E) and the hilly column need geographic validation before treating these names as terrain facts.")
    st.info("Decision: retain the coordinate proxy as an accepted mini-project limitation (Option A). Cell 20.5_73.5 is a flagged geographic assignment, not a validated coastal location. No measured coastal distance or elevation is used. The lookup and model remain unchanged; geographic refinement is future work.")
    with st.expander("Labelled map and full cell lookup"):
        figure("region_lookup_map")
        st.dataframe(cells[["cell_id","lat","lon","region_type"]],hide_index=True,use_container_width=True)
    st.download_button("Download region lookup",(ROOT/"data/region_lookup.csv").read_text(),file_name="region_lookup.csv",mime="text/csv")

else:
    st.markdown('''<div class="hero"><div class="eyebrow">How the experiment works</div><h1>Transparent inputs. Explicit rules.</h1><p>A classroom study of how closely nearest-neighbour voting approximates predefined severity thresholds.</p></div>''',unsafe_allow_html=True)
    st.subheader(RULE_NAME)
    st.markdown("The maximum temperature must first reach the region's base: **40°C plains, 37°C coastal, 30°C hilly**. Below the base, the label is Normal.")
    st.table(pd.DataFrame({"Rule":["Departure from normal","Absolute temperature, if normal ≥40°C"],"Heatwave":["≥4.5°C","Maximum ≥45°C"],"Severe Heatwave":["≥6.5°C","Maximum ≥47°C"]}))
    st.caption("The higher applicable severity wins. These project rules have not been verified against IMD criteria and omit station-level consecutive-day declaration requirements.")
    a,b = st.columns(2)
    with a:
        st.subheader("Data & normal")
        st.markdown("IMD annual maximum-temperature files, 1991–2025. The normal pools ±7 calendar days over 1991–2014, excluding 29 February. It differs from IMD's official normal.")
        st.markdown("Region proxy: longitude <74°E is coastal; 74–75°E below 20°N is hilly; remaining cells are plains.")
    with b:
        st.subheader("Training & evaluation")
        st.markdown("Train: 2015–2022. Test: 2023–2025. Eight leave-one-year-out folds compare 52 KNN configurations. Every fold fits its own preprocessing. No test observations enter the normal or model fitting.")
        st.markdown("Labels derive from model inputs. Scores describe rule approximation and do not validate independent forecasting.")
    with st.expander("Saved artifact details"):
        st.json(artifact.get("metadata",{}))
