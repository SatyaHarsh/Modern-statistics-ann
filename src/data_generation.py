import numpy as np
import pandas as pd
import os


def generate_synthetic_data(
    n_samples: int = 200,
    noise_std: float = 0.1,
    random_seed: int = 42
):
    """
    Generates synthetic non-linear data for modern statistics experiments.

    Mathematical model:
        y = sin(x) + ε
        where ε ~ N(0, noise_std^2)

    Parameters:
        n_samples (int): Number of data points
        noise_std (float): Standard deviation of Gaussian noise
        random_seed (int): Seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame with columns ['x', 'y']
    """

    np.random.seed(random_seed)

    x = np.linspace(-3, 3, n_samples)
    noise = np.random.normal(0, noise_std, size=n_samples)
    y = np.sin(x) + noise

    data = pd.DataFrame({
        "x": x,
        "y": y
    })

    return data


def save_dataset(data: pd.DataFrame, filename: str = "synthetic_data.csv"):
    """
    Saves dataset to the dataset folder.
    """

    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    file_path = os.path.join(dataset_dir, filename)
    data.to_csv(file_path, index=False)

    print(f"✅ Dataset saved at: {file_path}")


if __name__ == "__main__":
    data = generate_synthetic_data()
    save_dataset(data)
