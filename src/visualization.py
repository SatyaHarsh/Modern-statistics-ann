import pandas as pd
import matplotlib.pyplot as plt
import os


def visualize_data(filename="synthetic_data.csv"):
    """
    Visualizes the generated synthetic dataset.
    """

    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(base_dir, "dataset", filename)

    data = pd.read_csv(dataset_path)

    plt.figure(figsize=(8, 5))
    plt.scatter(data["x"], data["y"], alpha=0.6)
    plt.title("Synthetic Data: Non-linear Relationship")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    visualize_data()
