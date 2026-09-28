"""Shared implementation for notebook, CLI and Streamlit."""
__version__ = "0.1.0"
LABEL_MAP = {0: "Normal", 1: "Heatwave", 2: "Severe Heatwave"}
RULE_NAME = "Operational Heatwave Severity Rules (Project Approximation)"
INPUT_COLUMNS = ["date", "lat", "lon", "region_type", "max_temp", "normal_temp"]
