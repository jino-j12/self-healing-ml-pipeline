# 🧠 Self-Healing ML Pipeline

> An intelligent, autonomous machine learning pipeline that detects data drift, automatically retrains itself, and serves real-time fraud detection predictions — all without human intervention.

---

## 📌 Overview

The **Self-Healing ML Pipeline** is a production-inspired system built around credit card fraud detection. It trains a Random Forest classifier on historical transaction data, monitors incoming data for statistical drift using the **Kolmogorov-Smirnov test**, and automatically retrains the model when drift is detected. All activity is logged, versioned, and visualized through a live Streamlit dashboard and a FastAPI REST endpoint.

This project demonstrates key **MLOps concepts**:
- Automated model training & versioning
- Statistical data drift detection
- Self-healing retraining loop
- Real-time transaction stream simulation
- Live monitoring dashboard
- REST API for inference

---

## 🏗️ Architecture

```
self_healing_ml_pipeline/
│
├── main.py                  # Entry point — runs training, drift check, and stream loop
├── api.py                   # FastAPI REST API for fraud predictions
├── dashboard.py             # (Simple) Streamlit dashboard entry point
│
├── src/
│   ├── preprocessing.py     # Data loading and train/new split
│   ├── train_model.py       # Model training + versioned saving
│   ├── drift_detection.py   # KS-test based drift detection per feature
│   ├── retrain_model.py     # Automatic retraining on combined dataset
│   ├── visualize_drift.py   # Feature distribution comparison chart
│   ├── transaction_stream.py # Simulates real-time transaction batches
│   ├── logger.py            # Timestamped event logger
│   └── dashboard.py         # Full-featured Streamlit dashboard (auto-refresh)
│
├── data/
│   └── creditcard.csv       # Kaggle credit card fraud dataset (~284K rows)
│
├── models/
│   └── model_v*.pkl         # Auto-versioned trained model files
│
├── logs/
│   ├── pipeline_log.txt     # Timestamped pipeline activity log
│   └── drift_status.json    # JSON list of features where drift was found
│
└── drift_amount.png         # Histogram comparison plot for the Amount feature
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Auto Training** | Trains a Random Forest classifier (50 estimators) on 200K rows of transaction data |
| 📊 **Data Drift Detection** | Uses the Kolmogorov-Smirnov (KS) test to detect distribution shifts in all 30 features |
| 🔁 **Self-Healing Retraining** | When drift is detected, automatically combines old + new data and retrains the model |
| 🗂️ **Model Versioning** | Every trained or retrained model is saved as `model_v<N>.pkl` (20 versions accumulated so far) |
| 📡 **Transaction Streaming** | Simulates real-time data ingestion in configurable batches of 500 rows with a 2-second delay |
| 📈 **Drift Visualization** | Generates histogram overlays comparing training vs incoming data distributions |
| 📝 **Structured Logging** | All events are timestamped and written to `logs/pipeline_log.txt` |
| 🖥️ **Live Dashboard** | Streamlit dashboard with 5-second auto-refresh showing model registry, drift alerts, and logs |
| 🌐 **REST API** | FastAPI endpoint for real-time fraud prediction with probability scores |

---

## 🔄 Pipeline Flow

```
Load Data (creditcard.csv)
        │
        ▼
  Train Model ──────────────────────────────────────────┐
        │                                               │
        ▼                                               │
 Predict on New Data                              Save model_v<N>.pkl
        │
        ▼
 Drift Detection (KS-test per feature, p < 0.05)
        │
   ┌────┴─────┐
   │          │
 Drift?      No Drift
   │          │
   ▼          ▼
Retrain    Keep Model
  Model       │
   │          │
   └────┬─────┘
        │
        ▼
 Stream Transactions (500/batch, simulate real-time)
        │
        ▼
  Log All Events → pipeline_log.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pip`
- The [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`)

### 1. Clone the Repository

```bash
git clone https://github.com/jino-j12/self-healing-ml-pipeline.git
cd self_healing_ml_pipeline
```

### 2. Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install scikit-learn pandas numpy scipy matplotlib joblib fastapi uvicorn streamlit streamlit-autorefresh pillow
```

### 4. Add the Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
data/creditcard.csv
```

> ⚠️ The dataset is excluded from version control via `.gitignore` due to its size (~144 MB).

---

## ▶️ Running the Pipeline

### Run the Full Self-Healing Pipeline

```bash
python main.py
```

This will:
1. Load and split the dataset (200K train / remaining as new data)
2. Train the initial Random Forest model and save it as a versioned `.pkl`
3. Evaluate performance on the new data subset
4. Run KS-test drift detection across all features
5. Generate the `drift_amount.png` visualization
6. Auto-retrain if drift is detected, saving a new model version
7. Stream all transactions in batches of 500 with simulated delay

---

### Launch the Monitoring Dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard provides (auto-refreshes every 5 seconds):
- **Model Registry** — lists all saved model versions, highlights latest
- **Drift Visualization** — displays feature distribution comparison chart
- **Drift Alerts** — shows which features have drifted (from `drift_status.json`)
- **Recent Pipeline Events** — last 10 entries from `pipeline_log.txt`
- **Pipeline Status** — checklist of all active components

---

### Start the Prediction API

```bash
uvicorn api:app --reload
```

API will be available at `http://127.0.0.1:8000`

#### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict fraud for a transaction |

#### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, -0.10, -0.13, -0.19, -0.18, -0.14, -0.28, -0.56, -0.28, -2.14, 1.79, 0.27, 1.81, 0.50, -0.30, 0.28, 0.11, -0.29, -0.11, 0.04, 149.62]}'
```

#### Example Response

```json
{
  "prediction": "Fraud",
  "fraud_probability": 0.94
}
```

> ℹ️ The API automatically loads the **latest model version** from the `models/` folder at startup.

---

## 🧪 How Drift Detection Works

The pipeline uses the **Kolmogorov-Smirnov (KS) two-sample test** from `scipy.stats`:

```python
stat, p_value = ks_2samp(X_train[column], X_new[column])

if p_value < 0.05:
    # Drift detected in this feature
```

- Each of the **30 features** (Time, V1–V28, Amount) is tested independently
- A p-value below `0.05` indicates statistically significant distribution shift
- Drifted feature names are saved to `logs/drift_status.json` for the dashboard
- If **any** feature drifts, a full model retraining is triggered

---

## 📂 Data

The dataset used is the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset:

| Property | Value |
|---|---|
| Total rows | ~284,807 transactions |
| Features | 30 (Time, V1–V28 via PCA, Amount) |
| Label column | `Class` (0 = Normal, 1 = Fraud) |
| Train split | First 200,000 rows |
| New data split | Remaining rows (~84,807) |
| Fraud rate | ~0.17% (highly imbalanced) |

---

## 🗃️ Model Registry

All models are saved automatically to the `models/` folder with sequential versioning:

```
models/
├── model_v1.pkl   (~2.3 MB) — Initial trained model
├── model_v2.pkl   (~3.1 MB) — After first retraining
├── ...
└── model_v20.pkl  (~1.5 MB) — Latest version
```

Currently **20 model versions** have been accumulated across multiple pipeline runs.

---

## 📜 Logs

### `logs/pipeline_log.txt`
Timestamped log of every pipeline event:
```
[2026-03-07 21:49:01] Checking for data drift...
[2026-03-07 21:49:01] Drift detected in feature: Time
[2026-03-07 21:49:01] Drift detected in feature: Amount
[2026-03-07 22:06:26] Processed batch of 500 transactions
...
```

### `logs/drift_status.json`
JSON array of feature names where drift was detected in the latest run:
```json
["Time", "V1", "V2", "V3", ..., "Amount"]
```

---

## 🛠️ Module Reference

### `src/preprocessing.py`
Loads `creditcard.csv`, splits into training (first 200K rows) and new data, separates features from labels.

### `src/train_model.py`
Trains a `RandomForestClassifier(n_estimators=50, n_jobs=-1)`, auto-increments version number, saves to `models/model_v<N>.pkl`.

### `src/drift_detection.py`
Applies KS-test to each feature column. Logs drifted features, saves to `drift_status.json`, returns `True` if any drift is found.

### `src/retrain_model.py`
Concatenates old training data + new data, retrains the existing model object in-place, saves new versioned `.pkl`.

### `src/visualize_drift.py`
Plots overlapping histograms (train vs new) for a specified feature using `matplotlib`, saves to `drift_amount.png`.

### `src/transaction_stream.py`
Generator that yields batches of 500 rows from the CSV with a 2-second `time.sleep` delay to simulate live data arrival.

### `src/logger.py`
Appends `[YYYY-MM-DD HH:MM:SS] <message>` entries to `logs/pipeline_log.txt` and prints to stdout.

### `src/dashboard.py`
Full Streamlit dashboard with `streamlit-autorefresh` (5s interval), showing model registry, drift chart, alerts, and logs.

### `api.py`
FastAPI app that loads the latest model at startup. Exposes `POST /predict` which accepts a list of 30 feature values and returns `prediction` ("Fraud" / "Normal") and `fraud_probability`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | Random Forest classifier, metrics |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scipy` | KS-test for drift detection |
| `matplotlib` | Drift visualization plots |
| `joblib` | Model serialization (.pkl files) |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server for FastAPI |
| `streamlit` | Interactive monitoring dashboard |
| `streamlit-autorefresh` | Auto-refresh component for dashboard |
| `Pillow` | Image rendering in Streamlit |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Jino J** — [GitHub](https://github.com/jino-j12)

> Built as a demonstration of real-world MLOps principles: automated drift detection, self-healing retraining loops, model versioning, and continuous monitoring.
