import streamlit as st
import os
from PIL import Image
from streamlit_autorefresh import st_autorefresh

# ------------------------------
# Dashboard Title
# ------------------------------

st.title("Self-Healing ML Pipeline Dashboard")

# auto refresh every 5 seconds
st_autorefresh(interval=5000, key="refresh")

# ------------------------------
# Model Registry Section
# ------------------------------

st.header("Model Registry")

models_folder = "models"

if os.path.exists(models_folder):

    models = os.listdir(models_folder)

    if models:

        models = sorted(models)

        latest_model = models[-1]

        st.success(f"Latest Model Version: {latest_model}")

        st.subheader("Available Models")

        for model in models:
            st.write(model)

    else:

        st.warning("No models saved yet")

else:

    st.warning("Models folder not found")


# ------------------------------
# Drift Visualization
# ------------------------------

st.header("Drift Visualization")

image_path = "drift_amount.png"

if os.path.exists(image_path):

    image = Image.open(image_path)

    st.image(image, caption="Feature Distribution Drift (Amount)")

else:

    st.warning("No drift visualization available yet")

import json

st.header("Drift Alerts")

drift_file = "logs/drift_status.json"

if os.path.exists(drift_file):

    with open(drift_file, "r") as f:
        drift_features = json.load(f)

    if drift_features:

        st.error("Drift detected in features:")

        for feature in drift_features:
            st.write(f"⚠ {feature}")

    else:

        st.success("No drift detected")

else:

    st.warning("No drift data available yet")


# ------------------------------
# Pipeline Logs
# ------------------------------

st.header("Recent Pipeline Events")

log_file = "logs/pipeline_log.txt"

if os.path.exists(log_file):

    with open(log_file, "r") as f:

        logs = f.readlines()

    # show last 10 logs
    for log in logs[-10:]:

        st.text(log.strip())

else:

    st.warning("No logs found")


# ------------------------------
# Pipeline Status
# ------------------------------

st.header("Pipeline Status")

st.write("✔ Model Training")
st.write("✔ Drift Detection")
st.write("✔ Automatic Retraining")
st.write("✔ Model Versioning")
st.write("✔ Transaction Streaming")
st.write("✔ Monitoring Dashboard")