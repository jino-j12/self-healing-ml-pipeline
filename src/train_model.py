from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(X_train, y_train):

    model = RandomForestClassifier(n_estimators=50, n_jobs=-1)

    print("Training model...")
    model.fit(X_train, y_train)

    # create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # count existing models
    existing_models = len(os.listdir("models"))

    model_version = existing_models + 1

    model_path = f"models/model_v{model_version}.pkl"

    joblib.dump(model, model_path)

    print(f"Model saved as {model_path}")

    return model