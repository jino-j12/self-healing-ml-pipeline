import streamlit as st
import os
from PIL import Image

st.title("Self-Healing ML Pipeline Dashboard")

st.header("Model Registry")

models_folder = "models"

if os.path.exists(models_folder):

    models = os.listdir(models_folder)

    if models:
        latest_model = sorted(models)[-1]

        st.success(f"Latest Model Version: {latest_model}")

        st.write("Available Models:")

        for model in models:
            st.write(model)

    else:
        st.warning("No models saved yet")

else:
    st.warning("Models folder not found")

st.header("Pipeline Status")

st.write("✔ Model Training")
st.write("✔ Drift Detection")
st.write("✔ Automatic Retraining")
st.write("✔ Model Versioning")


st.header("Drift Visualization")

image_path = "drift_amount.png"

if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption="Drift Detection: Amount Feature")
else:
    st.warning("No drift visualization available yet")

st.header("Recent Pipeline Events")

log_file = "logs/pipeline_log.txt"

if os.path.exists(log_file):
    with open(log_file, "r") as f:
        logs = f.readlines()

    # show last 10 events
    for log in logs[-10:]:
        st.text(log.strip())

else:
    st.warning("No logs found")