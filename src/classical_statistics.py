import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def run_linear_regression(filename="synthetic_data.csv"):
    """
    Classical statistical model: Linear Regression
    """

    # Resolve dataset path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(base_dir, "dataset", filename)

    # Load dataset
    data = pd.read_csv(dataset_path)
    X = data["x"].values.reshape(-1, 1)
    y = data["y"].values

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Error metric (Mean Squared Error)
    mse = mean_squared_error(y, y_pred)

    print("📊 Classical Statistics: Linear Regression")
    print(f"Estimated model: y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")

    # Visualization
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Measured Data", alpha=0.6)
    plt.plot(X, y_pred, color="red", label="Linear Regression Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Classical Statistical Model: Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    return mse


if __name__ == "__main__":
    run_linear_regression()
