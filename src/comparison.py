import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def compare_models(filename="synthetic_data.csv"):
    # Resolve dataset path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(base_dir, "dataset", filename)

    # Load data
    data = pd.read_csv(dataset_path)
    X = data["x"].values.reshape(-1, 1)
    y = data["y"].values

    # ===============================
    # Classical Statistical Model
    # ===============================
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_lin = lin_model.predict(X)
    mse_lin = mean_squared_error(y, y_lin)

    # ===============================
    # ANN Model
    # ===============================
    ann_model = Sequential([
        Dense(32, activation="relu", input_shape=(1,)),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    ann_model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="mse"
    )

    ann_model.fit(X, y, epochs=300, batch_size=16, verbose=0)
    y_ann = ann_model.predict(X).flatten()
    mse_ann = mean_squared_error(y, y_ann)

    # ===============================
    # Results
    # ===============================
    print("\n📊 MODEL COMPARISON RESULTS")
    print(f"Classical Linear Regression MSE : {mse_lin:.6f}")
    print(f"ANN Model MSE                  : {mse_ann:.6f}")

    # ===============================
    # Visualization
    # ===============================
    plt.figure(figsize=(9, 6))
    plt.scatter(X, y, label="Measured Data", alpha=0.5)
    plt.plot(X, y_lin, color="red", label="Linear Regression")
    plt.plot(X, y_ann, color="green", label="ANN Approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Classical Statistics vs Modern Statistics (ANN)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    compare_models()
