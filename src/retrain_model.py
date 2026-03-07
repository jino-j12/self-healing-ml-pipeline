import pandas as pd
import joblib
import os

def retrain_model(model, X_train, y_train, X_new, y_new):

    print("\nDrift detected! Retraining model with new data...")

    combined_X = pd.concat([X_train, X_new])
    combined_y = pd.concat([y_train, y_new])

    model.fit(combined_X, combined_y)

    # save new model version
    os.makedirs("models", exist_ok=True)

    existing_models = len(os.listdir("models"))

    model_version = existing_models + 1

    model_path = f"models/model_v{model_version}.pkl"

    joblib.dump(model, model_path)

    print(f"New model saved as {model_path}")

    return model