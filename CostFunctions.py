from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

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


def build_keras_model(num_hidden_layers):
    if num_hidden_layers < 1:
        raise ValueError("Number of hidden layers must be at least 1.")

    model = keras.Sequential(
        [
            keras.layers.Dense(4, activation="relu", input_shape=(2,)),
        ]
    )

    for _ in range(num_hidden_layers - 1):
        model.add(keras.layers.Dense(4, activation="relu"))

    model.add(keras.layers.Dense(1))

    return model


def replace_hidden_layer_nodes(model, num_nodes):
    if len(num_nodes) != len(model.layers):
        raise ValueError(
            "Length of num_nodes array must match the number of layers in the model."
        )

    new_model = keras.models.clone_model(model)
    new_model.build()  # Build the new model to initialize the weights

    # Initialize GlorotUniform initializer with unique seeds for each layer
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    # Iterate through the layers and update the number of nodes for each layer
    for layer, nodes in zip(new_model.layers[1:-1], num_nodes[1:-1]):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.units = nodes
            # Rebuild the layer with the updated number of nodes and the unique initializer
            layer.build(layer.input_shape)
            layer.kernel_initializer = initializer
            layer.bias_initializer = initializer

    return new_model


def get_regression_data():
    df = pd.read_csv("fake_reg.csv")

    X = df[["feature1", "feature2"]].values
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    return X_train, X_test, y_train, y_test


def linear_regression(particle_position, X_test, y_test):
    feature1, feature2, intercept = particle_position

    y_pred = feature1 * X_test[:, 0] + feature2 * X_test[:, 1] + intercept

    mse = np.mean((y_test - y_pred) ** 2)

    # Return the MSE
    return mse


def ann_cost_function(particle):
    X_train, X_test, y_train, y_test = get_regression_data()
    model = build_keras_model(2)
    start = 0
    for layer in model.layers:
        num_weights = layer.get_weights()[0].size
        num_biases = layer.get_weights()[1].size
        num_params = num_weights + num_biases

        sliced_array = particle[start : start + num_params]
        weights = sliced_array[:num_weights]
        biases = sliced_array[num_weights:]

        weight_shape = layer.get_weights()[0].shape
        bias_shape = layer.get_weights()[1].shape
        reshaped_weights = np.reshape(weights, weight_shape)
        reshaped_biases = np.reshape(biases, bias_shape)

        layer.set_weights([reshaped_weights, reshaped_biases])
        start += num_params

    model.compile(optimizer="adam", loss="mse")

    training_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)

    return training_score


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
    term2 = np.prod(list(np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(args)))
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
