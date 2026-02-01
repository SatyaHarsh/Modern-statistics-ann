import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def run_ann_model(filename="synthetic_data.csv"):
    """
    ANN-based nonlinear statistical model
    """

    # Resolve dataset path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(base_dir, "dataset", filename)

    # Load dataset
    data = pd.read_csv(dataset_path)
    X = data["x"].values.reshape(-1, 1)
    y = data["y"].values

    # Build ANN (Feedforward MLP)
    model = Sequential([
        Dense(32, activation="relu", input_shape=(1,)),
        Dense(32, activation="relu"),
        Dense(1)  # Linear output neuron (regression)
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="mse"
    )

    # Train model
    history = model.fit(
        X, y,
        epochs=300,
        batch_size=16,
        verbose=0
    )

    # Predictions
    y_pred = model.predict(X).flatten()

    # Error metric
    mse = mean_squared_error(y, y_pred)

    print("Modern Statistics: ANN Model")
    print(f"Mean Squared Error (MSE): {mse:.6f}")

    # Visualization
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Measured Data", alpha=0.6)
    plt.plot(X, y_pred, color="green", label="ANN Approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("ANN-Based Nonlinear Statistical Model")
    plt.legend()
    plt.grid(True)
    plt.show()

    return mse


if __name__ == "__main__":
    run_ann_model()
