from scipy.stats import ks_2samp

def detect_drift(X_train, X_new):
    print("\nChecking for data drift...")

    drift_detected = False

    for column in X_train.columns:
        stat, p_value = ks_2samp(X_train[column], X_new[column])

        if p_value < 0.05:
            print(f"Drift detected in feature: {column}")
            drift_detected = True

    return drift_detected