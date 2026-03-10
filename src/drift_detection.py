from scipy.stats import ks_2samp
from src.logger import log_event
import json
import os

def detect_drift(X_train, X_new):

    log_event("Checking for data drift...")

    drifted_features = []

    for column in X_train.columns:

        stat, p_value = ks_2samp(X_train[column], X_new[column])

        if p_value < 0.05:

            log_event(f"Drift detected in feature: {column}")
            drifted_features.append(column)

    # save drift info
    os.makedirs("logs", exist_ok=True)

    with open("logs/drift_status.json", "w") as f:
        json.dump(drifted_features, f)

    return len(drifted_features) > 0