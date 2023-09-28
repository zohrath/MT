import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)
from tensorflow import keras
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model = keras.Sequential(
    [
        keras.layers.Dense(
            4, activation="relu", input_shape=(2,), bias_initializer="zeros"
        ),
        keras.layers.Dense(4, activation="relu", bias_initializer="zeros"),
        keras.layers.Dense(4, activation="relu", bias_initializer="zeros"),
        keras.layers.Dense(1),
    ]
)

weights = model.get_weights()
num_dimensions = sum(w.size for w in weights)


def get_regression_data():
    df = pd.read_csv("fake_reg.csv")

    X = df[["feature1", "feature2"]].values
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    return X_train, X_test, y_train, y_test


def get_fingerprinted_data_noisy():
    df = pd.read_csv("fingerprints-noisy.csv", delimiter=",")
    df = df[(df['X'] % 2 == 0) | (df['Y'] % 2 == 0)]
    df = df.drop(
        [
            "AP1_dev",
            "AP2_dev",
            "AP3_dev",
            "AP1_dist_dev",
            "AP2_dist_dev",
            "AP3_dist_dev",
            "id"
        ],
        axis=1,
    )

    free_variables = df.drop(["X", "Y"], axis=1).values
    dependent_variables = df[["X", "Y"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        free_variables, dependent_variables, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def get_fingerprinted_data():
    df = pd.read_csv("fingerprints.csv", delimiter=",")
    df = df[(df['X'] % 2 == 0) | (df['Y'] % 2 == 0)]
    df = df.drop(
        [
            "AP1_dev",
            "AP2_dev",
            "AP3_dev",
            "AP1_dist_dev",
            "AP2_dist_dev",
            "AP3_dist_dev",
        ],
        axis=1,
    )

    free_variables = df.drop(["X", "Y"], axis=1).values
    dependent_variables = df[["X", "Y"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        free_variables, dependent_variables, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def get_log_distance_path_loss_data():
    df = pd.read_csv("fingerprints.csv", delimiter=",")

    ref_x = 7.5
    ref_y = -1
    df["AP1_distance_ref"] = (
        (df["X"] - ref_x) ** 2 + (df["Y"] - ref_y) ** 2) ** 0.5

    df = df.drop(
        [
            "AP1_dev",
            "AP2_dev",
            "AP3_dev",
            "AP1_dist",
            "AP2_dist",
            "AP3_dist",
            "AP1_dist_dev",
            "AP2_dist_dev",
            "AP3_dist_dev",
            "X",
            "Y",
        ],
        axis=1,
    )
    print(df.head())
    # Reference coordinates

    X_train, X_test, y_train, y_test = train_test_split(
        df["AP1_rss"], df["AP1_distance_ref"], test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def ann_node_count_fitness(num_nodes_per_layer):
    num_nodes_per_layer = np.round(num_nodes_per_layer)
    if len(num_nodes_per_layer) < 1:
        raise ValueError("Number of layers must be at least 1.")

    X_train, X_test, y_train, y_test, scaler = get_fingerprinted_data()

    model = keras.Sequential(
        [
            keras.layers.Dense(num_nodes_per_layer[0], activation="relu"),
        ]
    )

    if len(num_nodes_per_layer) > 1:
        for nodeCount in num_nodes_per_layer[1:]:
            model.add(keras.layers.Dense(nodeCount, activation="relu"))

    model.add(keras.layers.Dense(2))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=100,
              validation_data=(X_test, y_test), verbose=0)
    # losses = pd.DataFrame(model.history.history)

    # # Extract loss and validation loss columns
    # train_loss = losses["loss"]
    # val_loss = losses["val_loss"]

    # # Create a plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_loss, label="Training Loss")
    # plt.plot(val_loss, label="Validation Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training and Validation Loss Over Epochs")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    predictions = model.predict(X_test, verbose=0)

    # print("RMSE: ", np.sqrt(mean_squared_error(y_test, predictions)))
    # print("Explained variance score: ", explained_variance_score(y_test, predictions))

    return mean_squared_error(y_test, predictions)


def linear_regression(particle_position, X_test, y_test):
    feature1, feature2, intercept = particle_position

    y_pred = feature1 * X_test[:, 0] + feature2 * X_test[:, 1] + intercept

    mse = np.mean((y_test - y_pred) ** 2)

    # Return the MSE
    return mse


X_train, _, y_train, _, _ = get_fingerprinted_data()


def ann_weights_fitness_function(particle, model):

    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        num_weights = weights.size
        num_biases = biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = particle[:num_weights]
        sliced_biases = particle[num_weights: num_weights + num_biases]

        # Update the continuous_values array for the next iteration
        particle = particle[num_weights + num_biases:]

        # Set the sliced weights and biases in the layer
        layer.set_weights(
            [sliced_weights.reshape(weights.shape),
             sliced_biases.reshape(biases.shape)]
        )
    model.compile(optimizer="adam", loss="mse")

    # Evaluate the model and get the evaluation metrics
    evaluation_metrics = model.evaluate(X_train, y_train, verbose=0)
    rmse = np.sqrt(evaluation_metrics)

    return rmse


def log_path_loss_model_fitness_function(particle, X_train, y_train):
    A = particle[0]
    n = particle[1]

    # Predict distances using the path loss model
    predicted_distances = 10 ** ((X_train - A) / (10 * n))

    # Calculate RMSE between predicted and actual distances for the AP's dataset
    rmse = np.sqrt(mean_squared_error(y_train, predicted_distances))

    return rmse


def sphere(x):
    return sum(x_i**2 for x_i in x)


def rosenbrock(args):
    return sum(
        (1 - x) ** 2 + 100 * (y - x**2) ** 2 for x, y in zip(args[::2], args[1::2])
    )


def rastrigin(X):
    A = 10
    return A + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])


def schwefel(args):
    return 418.9829 * len(args) - sum(x * np.sin(np.sqrt(abs(x))) for x in args)


def griewank(args):
    term1 = sum(x**2 for x in args) / 4000
    term2 = np.prod(list(np.cos(x / np.sqrt(i + 1))
                    for i, x in enumerate(args)))
    return term1 - term2 + 1


def penalized1(args):
    penalty_term = (
        0.1
        * np.sqrt(sum(x**2 for x in args))
        * np.sin(50 * np.sqrt(sum(x**2 for x in args)))
    )
    return -((1 + penalty_term) ** 2)


def step(args, lower_bound=-5.0, upper_bound=5.0):
    for x in args:
        if not (lower_bound <= x <= upper_bound):
            return 1
    return 0
