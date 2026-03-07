from sklearn.metrics import classification_report
from src.preprocessing import load_and_split_data
from src.train_model import train_model
from src.drift_detection import detect_drift
from src.retrain_model import retrain_model

# load data
X_train, y_train, X_new, y_new = load_and_split_data("data/creditcard.csv")

# train model
model = train_model(X_train, y_train)

# predictions
predictions = model.predict(X_new)

print("\nModel performance on new data:")
print(classification_report(y_new, predictions))

# drift detection
drift = detect_drift(X_train, X_new)

# self healing
if drift:
    model = retrain_model(model, X_train, y_train, X_new, y_new)
else:
    print("No drift detected. Model remains unchanged.")