```markdown
🌍 Urban Air Quality Prediction — PM2.5 → AQI Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-green)
![RandomForest](https://img.shields.io/badge/RandomForest-Model-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> A machine learning project to predict real-time Air Quality Index (AQI) from PM2.5 data using RandomForest and LightGBM, with integration to OpenAQ API.**

---

## 📘 Overview

This notebook demonstrates a complete workflow for **urban air quality modeling**, including:
- Data ingestion from OpenAQ (or local CSV)
- Data cleaning, resampling, and feature engineering
- Machine learning model training (RandomForest & LightGBM)
- Model evaluation (MAE, RMSE, risk category classification)
- Real-time AQI prediction using OpenAQ sensor data
- Color-coded AQI visualization and risk level interpretation

The model predicts **next-hour AQI values** based on PM2.5 trends and rolling averages, enabling early warnings for unhealthy air conditions.

---

 🚀 Features

✅ Fetch live PM2.5 readings from [OpenAQ API](https://docs.openaq.org/)  
✅ Train and compare **RandomForest** and **LightGBM** models  
✅ Evaluate performance using MAE, RMSE, and AQI category reports  
✅ Perform real-time prediction and AQI risk scoring  
✅ Generate color-coded dashboards for easier interpretation  
✅ Save models (`.pkl`) and processed datasets to Google Drive  

---

📊 Model Performance

| Model         | MAE (↓) | RMSE (↓) | Remarks |
|----------------|----------|-----------|----------|
| RandomForest   | 23.25    | 43.70     | Stable, interpretable |
| LightGBM       | 21.70    | 42.04     | Slightly better accuracy |

📈 Although LightGBM achieved lower MAE/RMSE, RandomForest’s predictions visually aligned better with actual AQI patterns.

---

## 🧠 Model Pipeline

```

Raw PM2.5 Data (OpenAQ/CSV)
↓
Resampling & Cleaning (Hourly)
↓
Feature Engineering (lags, rolling means)
↓
Train/Test Split
↓
Model Training (RF + LGBM)
↓
Evaluation + Visualization
↓
Real-Time Prediction via OpenAQ API

```

---

🌡️ Real-Time AQI Demo

The notebook includes a **real-time prediction cell** that:
- Asks for your OpenAQ API key (`input()` prompt)
- Fetches the latest PM2.5 reading by `sensor_id`
- Predicts AQI using both models and blends them
- Displays a color-coded summary (green → red → purple)
- Calculates **official formula-based AQI** for comparison

Example output:

```

📡 Latest PM2.5: 270.0 µg/m³
🤖 Model Blended AQI: 165.6  → Unhealthy
📏 Official AQI: 316.5       → Hazardous

````

---

## 🔧 Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/urban-air-quality-prediction.git
cd urban-air-quality-prediction
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

**Core Libraries:**

* pandas, numpy, scikit-learn
* lightgbm, matplotlib, seaborn
* requests, joblib, IPython

### 3️⃣ Run the notebook

```bash
jupyter notebook urban_aqi_prediction.ipynb
```

### 4️⃣ (Optional) Use OpenAQ API

Get your free API key from [https://api.openaq.org](https://api.openaq.org).
When prompted, enter your key to enable real-time predictions.

---

## 💾 Saving Models

After training, models are saved as:

```
/urban_aqi_model/
│── randomforest_aqi_model.pkl
│── lightgbm_aqi_model.pkl
│── daily_pm25_aqi.csv
```

---

## 🎨 Output Visualization

Includes:

* 📈 Actual vs Predicted AQI (time series)
* 🌈 Feature importance (LightGBM)
* 🧾 Classification Report (AQI risk categories)
* 🟢🟡🟠🔴 Color-coded AQI summary cards for real-time predictions

---

## 🧩 Folder Structure

```
urban-air-quality-prediction/
│
├── urban_aqi_prediction.ipynb
├── requirements.txt
├── README.md
├── data/
│   └── sample_aqi_data.csv
└── models/
    ├── randomforest_aqi_model.pkl
    └── lightgbm_aqi_model.pkl
```

---

## ⚠️ Notes

* Ensure your data is **hourly averaged** before training.
* OpenAQ free API may have rate limits (60 req/min).
* For higher reliability, blend both models as done in the notebook.

---

## 📜 License

MIT License — free to use, modify, and share with attribution.

---

## 👩‍💻 Author

**Manasvi Awasthi**
💡 *Environmental Data Science | Air Quality Modeling | ML Research*

---
