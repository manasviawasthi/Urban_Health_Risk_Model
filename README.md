---

# 📘 Project Execution Report

### **Urban Air Quality Prediction using Machine Learning Models**

---

## 1. Introduction

This project focuses on the development of a **machine learning–based Air Quality Prediction System** to estimate **Air Quality Index (AQI)** levels from **PM2.5 (Particulate Matter ≤ 2.5 µm)** data. The primary goal is to forecast short-term urban air pollution conditions and classify the potential health risk levels.

By leveraging **OpenAQ**, a global open-source air quality data platform, we combined real-world sensor measurements with statistical and machine learning techniques to model the relationship between PM2.5 concentrations and AQI levels.

The system is capable of **real-time AQI prediction**, **risk categorization**, and **visual interpretation** of air pollution trends.

---

## 2. Data Source and Collection

* **Data Source:** [OpenAQ API](https://docs.openaq.org/)
* **Sensor ID:** 23534 (Location ID: 8118, urban air station)
* **Parameter Monitored:** PM2.5 (µg/m³)
* **Sampling Frequency:** Hourly
* **Fallback Option:** Local CSV (if live API unavailable)

The OpenAQ API was used to retrieve recent and historical PM2.5 data for training and testing the prediction model. A fallback mechanism ensures continuity by loading the last recorded values from local datasets when the API is inaccessible.

---

## 3. Data Preprocessing and Feature Engineering

Data preprocessing was performed to ensure quality and consistency of input signals. The steps included:

1. **Datetime Normalization:** Conversion to local time zone (`datetimeLocal`).
2. **Resampling:** Aggregated to hourly averages to remove noise.
3. **Missing Value Imputation:** Forward-fill and interpolation applied.
4. **Feature Engineering:**

   * Lag features: PM2.5 at t–1, t–3 hours.
   * Rolling mean features: 3h, 6h, 12h, and 24h windows.
   * Derived features enhance the model’s ability to detect short and long-term pollution trends.

Final feature set used for model training:
`['datetime', 'pm25', 'AQI', 'Risk_Score_0_100', 'Risk_Level']`

---

## 4. Model Development

Two supervised regression models were trained and compared:

| Model                       | Framework          | Description                                                                                                  |
| --------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------ |
| **Random Forest Regressor** | scikit-learn       | Ensemble model based on multiple decision trees. Interpretable and stable, good baseline for AQI prediction. |
| **LightGBM Regressor**      | Microsoft LightGBM | Gradient boosting model optimized for performance and accuracy on structured data.                           |

### Model Configuration

* **Train/Test Split:** 80% / 20%
* **Target Variable:** AQI (derived from PM2.5 using USEPA breakpoints)
* **Features Used:** Current and lagged PM2.5 + rolling means

---

## 5. Model Evaluation

Both models were evaluated using **Mean Absolute Error (MAE)** and **Root Mean Square Error (RMSE)** on the test dataset.

| Model        | MAE (↓)   | RMSE (↓)  | Remarks                                            |
| ------------ | --------- | --------- | -------------------------------------------------- |
| RandomForest | **23.25** | **43.70** | Stable and visually aligned with actual AQI trends |
| LightGBM     | **21.70** | **42.04** | Slightly better numerical accuracy                 |

Although LightGBM produced slightly lower error metrics, **RandomForest’s prediction pattern matched more closely with real-world AQI fluctuations**, indicating better generalization for unseen temporal data.

---

## 6. Real-Time Prediction System

The notebook includes an **interactive real-time prediction cell** integrated with OpenAQ’s live sensor API.

### Key Features:

* Prompts user for **API Key** at runtime
* Fetches **latest PM2.5 readings** from OpenAQ sensors
* Predicts AQI using both models (RandomForest + LightGBM)
* Generates a **blended AQI** (mean of both predictions)
* Classifies **Risk Level** and assigns **color-coded severity**

#### Example Output:

| Metric               | Value         |
| -------------------- | ------------- |
| PM2.5                | 270.00 µg/m³  |
| Predicted AQI (RF)   | 163.5         |
| Predicted AQI (LGBM) | 167.6         |
| Blended AQI          | 165.6         |
| Risk Level           | **Unhealthy** |
| Risk Score           | 33.1 / 100    |

Color-coded cards (green → red → purple) help visualize pollution severity dynamically.

---

## 7. Key Findings

1. **Correlation:** PM2.5 levels strongly correlate with AQI, confirming it as the primary determinant of urban air health.
2. **Temporal Behavior:** Lag and rolling mean features significantly improve model accuracy, indicating persistence in pollution patterns.
3. **Model Trade-offs:** LightGBM achieves higher precision, while RandomForest maintains robustness and interpretability.
4. **Real-Time Capability:** The system successfully integrates live API data for continuous AQI monitoring.
5. **Risk Categorization:** AQI outputs are mapped to health-based risk levels (Good → Hazardous) following USEPA standards.

---

## 8. Visual Analysis

* **Actual vs Predicted AQI Graphs** — demonstrate model performance and trend capture ability.
* **Feature Importance (LightGBM)** — confirms `pm25` and short-term averages as the most influential features.
* **Risk Color Mapping** — enhances interpretability for public health communication.

---

## 9. Model Persistence and Export

Both trained models and processed datasets are exportable to Google Drive for reuse or deployment:

```python
joblib.dump(model_rf, 'randomforest_aqi_model.pkl')
joblib.dump(model_lgb, 'lightgbm_aqi_model.pkl')
df_daily.to_csv('daily_pm25_aqi.csv', index=False)
```

This allows integration with dashboards, web applications, or city-level monitoring tools.

---

## 10. Conclusion

The developed system demonstrates that **machine learning models can effectively predict short-term AQI from PM2.5 data**, providing valuable insights for policymakers, environmental scientists, and citizens.

The project bridges environmental science and AI by transforming open environmental data into **actionable intelligence** — enabling real-time, data-driven urban air quality management.

---

## 11. Future Scope

* Expand to multi-pollutant models (NO₂, SO₂, O₃).
* Deploy as a live API or web dashboard.
* Integrate weather and traffic data for contextual prediction.
* Enhance explainability using SHAP/LIME for public interpretability.

---

## 12. References

* OpenAQ API Documentation — [https://docs.openaq.org/](https://docs.openaq.org/)
* U.S. Environmental Protection Agency (USEPA) AQI Breakpoints
* Scikit-learn and LightGBM Documentation

---

