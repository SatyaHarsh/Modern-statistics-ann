Modern Statistics with Artificial Neural Networks (ANN)

1. Introduction

Classical statistical models are widely used for system modeling and data analysis. However, such models often rely on strong assumptions, such as linearity and predefined probability distributions, which limit their ability to represent complex real-world systems.

Artificial Neural Networks (ANNs) overcome these limitations by learning nonlinear input–output relationships directly from data. Due to this capability, ANNs can be interpreted as modern statistical models, especially in the context of nonlinear system identification and data-driven modeling.

This project demonstrates the use of ANN as a modern statistical tool by comparing it with a classical statistical approach using a simulation-based example.

2. Project Objectives

The main objectives of this project are:

- To explain the theoretical background of ANN as modern statistical models

- To discuss realization and identification properties of ANN

- To generate a synthetic dataset representing a nonlinear system

- To implement a classical statistical model (linear regression)

- To implement an ANN-based nonlinear model

- To compare both approaches using quantitative error metrics and visualization

3. Theoretical Background
3.1 Classical Statistics

Classical statistical modeling techniques, such as linear regression, assume a linear relationship between input and output variables. Parameters are estimated using methods like least squares, and model performance is evaluated using error measures such as Mean Squared Error (MSE).

While effective for linear systems, these models perform poorly when the underlying system behavior is nonlinear.

3.2 Artificial Neural Networks as Modern Statistics

Artificial Neural Networks can be viewed as nonlinear statistical models that approximate unknown functions through data-driven learning.

Key characteristics:

- Nonlinear function approximation

- Parameter learning through error minimization

- No explicit assumption about data distribution

- Robustness to noise

Training an ANN corresponds to statistical parameter identification, where weights and biases are adapted to minimize an expected loss function (MSE).

4. Dataset Generation

A synthetic dataset is generated using the nonlinear model:
y=sin(x)+ε

where:
x is the input variable
ε is Gaussian noise

This dataset represents a nonlinear system that cannot be accurately modeled using linear statistical methods.

5. Project Structure
modern-statistics-ann/
│
├── dataset/
│   └── synthetic_data.csv
│
├── src/
│   ├── data_generation.py
│   ├── visualization.py
│   ├── classical_statistics.py
│   ├── ann_model.py
│   └── comparison.py
│
├── README.md
├── .gitignore
└── requirements.txt

6. Classical Statistical Model

A linear regression model is used as a baseline classical statistical approach.

- Assumes linear input–output relationship

- Parameters estimated using least squares

- Performance evaluated using Mean Squared Error (MSE)

Due to the nonlinear nature of the dataset, the linear model exhibits significant approximation error.

7. ANN-Based Statistical Model

A feed-forward Artificial Neural Network (Multilayer Perceptron) is implemented:

- Two hidden layers with ReLU activation

- One linear output neuron

- Trained using backpropagation

- Loss function: Mean Squared Error (MSE)

The ANN successfully learns the nonlinear relationship and achieves significantly lower error compared to the classical model.

8. Simulation Results and Comparison

Both models are evaluated using the same dataset.

Key Observations:

- Linear regression fails to capture nonlinear behavior

- ANN closely approximates the true system dynamics

- ANN achieves a much lower MSE

- Visualization clearly shows ANN superiority

This confirms ANN as a powerful modern statistical modeling tool.

9. Conclusion

This project demonstrates that Artificial Neural Networks function as effective modern statistical models for nonlinear system identification.

While classical statistical methods rely on strict assumptions and are limited to linear behavior, ANN learns complex relationships directly from data. The simulation results clearly show the advantages of ANN in terms of accuracy, flexibility, and modeling capability.

10. Technologies Used

- Python 3.11

- NumPy

- Pandas

- Matplotlib

- Scikit-learn

- TensorFlow (Keras)

11. How to Run the Project
# Activate virtual environment
source venv/bin/activate

# Generate dataset
python src/data_generation.py

# Visualize data
python src/visualization.py

# Run classical statistical model
python src/classical_statistics.py

# Run ANN model
python src/ann_model.py

# Compare both models
python src/comparison.py