import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


BUCKET_NAME = "wildlife-forecasting-thomas"

MODEL_PATH = f"gs://{BUCKET_NAME}/outputs/model.pkl"
DATA_PATH = f"gs://{BUCKET_NAME}/living-planet-index.csv"
OUTPUT_PATH = f"gs://{BUCKET_NAME}/outputs/predictions.csv"


FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_mean_5",
    "rolling_std_3",
    "rolling_std_5",
    "trend_slope_5",
]


def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")

    df.columns = [
        "Region",
        "Year",
        "Average_Index",
        "Upper_Index",
        "Lower_Index",
    ]

    df["Year"] = df["Year"].astype(int)
    df["Average_Index"] = df["Average_Index"].astype(float)
    df["Upper_Index"] = df["Upper_Index"].astype(float)
    df["Lower_Index"] = df["Lower_Index"].astype(float)

    df = df.sort_values(["Region", "Year"]).reset_index(drop=True)

    return df


def compute_slope(series):
    y = series.values
    x = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    return model.coef_[0]


def create_features(df):
    df = df.copy()
    df = df.sort_values(["Region", "Year"])

    grouped = df.groupby("Region")["Average_Index"]

    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = grouped.shift(lag)

    df["rolling_mean_3"] = grouped.transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    df["rolling_mean_5"] = grouped.transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    df["rolling_std_3"] = grouped.transform(
        lambda x: x.shift(1).rolling(3).std()
    )

    df["rolling_std_5"] = grouped.transform(
        lambda x: x.shift(1).rolling(5).std()
    )

    df["trend_slope_5"] = grouped.transform(
        lambda x: x.shift(1).rolling(5).apply(compute_slope, raw=False)
    )

    df = df.dropna().reset_index(drop=True)

    return df


def load_model():
    local_model_path = "model.pkl"

    if not os.path.exists(local_model_path):
        os.system(f"gsutil cp {MODEL_PATH} {local_model_path}")

    return joblib.load(local_model_path)


def predict():
    df = load_data()
    df_features = create_features(df)

    latest_rows = (
        df_features.sort_values(["Region", "Year"])
        .groupby("Region")
        .tail(1)
        .copy()
    )

    X = latest_rows[FEATURES]

    model = load_model()
    predictions = model.predict(X)

    latest_rows["Predicted_Next_Index"] = predictions
    latest_rows["Prediction_For_Year"] = latest_rows["Year"] + 1

    output_cols = [
        "Region",
        "Year",
        "Prediction_For_Year",
        "Average_Index",
        "Predicted_Next_Index",
    ]

    predictions_df = latest_rows[output_cols]

    predictions_df.to_csv("predictions.csv", index=False)
    os.system(f"gsutil cp predictions.csv {OUTPUT_PATH}")

    print(predictions_df)
    print(f"Predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    predict()