# Wildlife Biodiversity Forecasting (GCP + ML Pipeline)

## Overview

This project builds an end-to-end machine learning pipeline to **forecast biodiversity trends** using the Living Planet Index dataset.

The system predicts the **next-year biodiversity index** for different regions and ecosystems, and transforms predictions into actionable insights using a **Delta metric**.

---

## Objective

* Analyze historical biodiversity data
* Forecast future biodiversity trends
* Identify regions and ecosystems at highest risk
* Visualize insights through dashboards

---

## Dataset

The dataset contains biodiversity index values over time across:

### Geographic Regions

* Africa
* Europe and Central Asia
* North America
* Asia and Pacific
* Latin America and the Caribbean

### Ecosystem Groups

* Freshwater
* World

The **Average_Index** represents biodiversity health:

* Higher → healthier ecosystems
* Lower → biodiversity loss

---

## Pipeline Architecture

```text
Raw CSV (GCS)
      ↓
Feature Engineering (lags, rolling stats)
      ↓
Model Training (Linear Regression)
      ↓
Batch Prediction (Vertex AI)
      ↓
Predictions saved to GCS
      ↓
BigQuery (SQL transformations)
      ↓
Looker Studio Dashboard
```

---

## Feature Engineering

To capture time-series behavior, the following features are created:

* Lag features: `lag_1`, `lag_2`, `lag_3`
* Rolling statistics:

  * Mean (3, 5)
  * Standard deviation (3, 5)
* Trend feature:

  * `trend_slope_5` (linear regression slope over last 5 points)

These features allow the model to learn:

* Momentum (recent behavior)
* Trend direction (growth/decline)
* Volatility (stability vs fluctuations)

---

## Model

* Model: **Linear Regression**
* Input: Engineered features
* Output: Predicted biodiversity index for next year

---

## Key Metric: Delta

```text
Delta = Predicted_Next_Index - Average_Index
```

Interpretation:

* **Delta < 0** → biodiversity expected to decline
* **Delta > 0** → biodiversity expected to improve

This transforms raw predictions into **decision-ready insights**.

---

## Outputs

Stored in Google Cloud Storage:

* `model.pkl` → trained model
* `predictions.csv` → prediction results

---

## Data Transformation (BigQuery)

The predictions are split into:

* `regional_predictions` → geographic regions
* `ecosystem_predictions` → ecosystems

SQL is used to:

* Compute Delta
* Separate semantic groups

---

## Visualization (Looker Studio)

Two main dashboards:

### 1. Regional Forecast

* Shows predicted biodiversity decline by region
* Identifies high-risk regions

### 2. Ecosystem Forecast

* Shows trends for ecosystems (e.g., freshwater)
* Highlights global environmental patterns

---

## Key Insights

* Biodiversity is predicted to **decline across all regions**
* **Africa and North America** show the largest expected decreases
* **Latin America** appears relatively more stable
* Ecosystems such as **Freshwater** also show declining trends

---

## Limitations

* Model is based only on historical trends (no external factors)
* Does not include climate, policy, or human activity variables
* Assumes future follows past patterns


---

## Tech Stack

* Python (pandas, numpy, scikit-learn)
* Google Cloud Platform:

  * Vertex AI (training & batch prediction)
  * Cloud Storage (data & artifacts)
  * BigQuery (data transformation)
  * Looker Studio (visualization)

---

## Conclusion

This project demonstrates how to build a **production-style ML pipeline**, transforming raw data into actionable insights through:

* Feature engineering
* Predictive modeling
* Cloud-based processing
* Data visualization

It highlights how machine learning can support **environmental decision-making**.
