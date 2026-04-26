import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data():
    csv_path = "gs://wildlife-forecasting-thomas/living-planet-index.csv"

    df = pd.read_csv(csv_path, sep=";")

    df.columns = [
        "Region",
        "Year",
        "Average_Index",
        "Upper_Index",
        "Lower_Index"
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

    # Lag features: previous years
    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Rolling features: use ONLY past values, not current target
    df["rolling_mean_3"] = grouped.transform(lambda x: x.shift(1).rolling(3).mean())
    df["rolling_mean_5"] = grouped.transform(lambda x: x.shift(1).rolling(5).mean())

    df["rolling_std_3"] = grouped.transform(lambda x: x.shift(1).rolling(3).std())
    df["rolling_std_5"] = grouped.transform(lambda x: x.shift(1).rolling(5).std())

    df["trend_slope_5"] = grouped.transform(
        lambda x: x.shift(1).rolling(5).apply(compute_slope, raw=False)
    )

    df_model = df.dropna().reset_index(drop=True)

    return df_model

def train_and_evaluate(df_model, split_year=2005):
    features = [
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "rolling_mean_5",
        "rolling_std_3",
        "rolling_std_5",
        "trend_slope_5"
    ]

    target = "Average_Index"

    train_df = df_model[df_model["Year"] <= split_year]
    test_df = df_model[df_model["Year"] > split_year]

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

    print("Linear Regression RMSE:", rmse_lr)
    print("XGBoost RMSE:", rmse_xgb)

    return lr_model, rmse_lr, y_pred_lr, test_df, y_test

def save_model(model):
    local_model_path = "model.pkl"
    gcs_model_path = "gs://wildlife-forecasting-thomas/outputs/model.pkl"

    joblib.dump(model, local_model_path)

    os.system(f"gsutil cp {local_model_path} {gcs_model_path}")

    print(f"Model saved to: {gcs_model_path}")


def plot_predictions(test_df, y_test, y_pred):
    plt.figure(figsize=(10, 5))

    plt.plot(test_df["Year"], y_test.values, label="Actual")
    plt.plot(test_df["Year"], y_pred, label="Predicted")

    plt.xlabel("Year")
    plt.ylabel("Average Index")
    plt.title("Actual vs Predicted Population Index")
    plt.legend()

    local_plot_path = "predictions_plot.png"
    gcs_plot_path = "gs://wildlife-forecasting-thomas/outputs/predictions_plot.png"

    plt.savefig(local_plot_path)
    os.system(f"gsutil cp {local_plot_path} {gcs_plot_path}")

    print(f"Plot saved to: {gcs_plot_path}")
    plt.show()


def main():
    df = load_data()

    print("Loaded data:")
    print(df.head())
    print(df.info())

    df_model = create_features(df)

    print("Model-ready data:")
    print(df_model.head())
    print("Shape:", df_model.shape)

    model, rmse, y_pred, test_df, y_test = train_and_evaluate(df_model)

    print("RMSE:", rmse)

    plot_predictions(test_df, y_test, y_pred)

    save_model(model)


if __name__ == "__main__":
    main()
