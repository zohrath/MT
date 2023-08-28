import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


from CostFunctions import get_fingerprinted_data

# Med upper bound 2.0 och lower bound -1.0:

# Given array of weights and biases
weights_biases = np.array(
    [
        1.6826488,
        1.96462035,
        0.3848043,
        2.0,
        -1.0,
        -0.99999844,
        1.16212366,
        -0.80736833,
        1.59543145,
        1.33724797,
        -1.0,
        0.54035601,
        2.0,
        -1.0,
        1.93461477,
        -1.0,
        1.99523898,
        1.8594115,
        2.0,
        -0.83455676,
        -0.6496651,
        1.86875475,
        2.0,
        -1.0,
        -1.0,
        0.51905599,
        1.62208914,
        -0.27110193,
        -1.0,
        1.26428533,
        -0.84735914,
        -0.99980285,
        1.15410748,
        -1.0,
        -0.61770491,
        -0.67259027,
        -0.39754078,
        -0.39821655,
        -0.19091158,
        1.97692845,
        1.49676839,
        0.3350044,
        1.79952157,
        -0.27748941,
        1.74178691,
        1.85487574,
        -1.0,
        -0.91273735,
        1.52466019,
        -1.0,
        -0.08352284,
        1.0957222,
        1.16392206,
        -0.33065401,
        -0.99999457,
        2.0,
        1.77188668,
        -0.57062004,
        -0.85822845,
        -0.93297602,
        -0.68618921,
        -0.41915388,
        -0.20911534,
        1.71787912,
        -1.0,
        -0.90561035,
        1.72072348,
        -0.67156426,
        0.46055635,
        -0.34356725,
        1.70458077,
        0.02986414,
        2.0,
        0.88057392,
        -1.0,
        1.07055788,
        1.80367783,
        -0.22035074,
        -0.39350897,
        -1.0,
        -0.00613335,
        0.20704063,
        0.76266495,
        0.80190614,
        -0.63171097,
        -0.68709728,
        -0.37842605,
        -0.99592408,
        1.06777575,
        -0.40995542,
        2.0,
        1.72261297,
        -0.48582995,
        0.55482952,
        0.41807017,
        0.49329802,
        -0.78240454,
        -0.91182823,
        0.61626971,
        -0.16569766,
        -1.0,
        -0.90383385,
        -1.0,
        2.0,
        0.56422779,
        0.91778492,
        -0.00871026,
        1.04018665,
        -1.0,
        -0.6799783,
        -0.73280287,
        -0.99894187,
        1.26165199,
        -0.66302267,
        1.99173278,
        -0.98914432,
        0.44517297,
        -0.18063943,
        1.93900096,
        -0.01804117,
        -0.13576734,
        -0.11579808,
        1.16720299,
        0.92244695,
        -0.16274281,
        -1.0,
        -0.25697596,
        -0.61758342,
        0.63717002,
        -0.05562057,
        -0.00667171,
        1.34914113,
        -0.94688882,
        -1.0,
        -0.64101393,
        0.66624017,
        1.99151021,
        0.53037385,
        1.83764329,
        2.0,
    ]
)


def create_model():
    num_nodes_per_layer = [
        6,
        6,
        6,
    ]
    model = Sequential()
    model.add(Dense(num_nodes_per_layer[0], activation="relu", input_shape=(6,)))

    for nodes in num_nodes_per_layer[1:]:
        model.add(Dense(nodes, activation="relu"))

    model.add(Dense(2))  # Output layer

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return model


# Create a Keras Sequential model
model = create_model()

for layer in model.layers:
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    num_weights = weights.size
    num_biases = biases.size

    # Slice off values from the continuous_values array for weights and biases
    sliced_weights = weights_biases[:num_weights]
    sliced_biases = weights_biases[num_weights : num_weights + num_biases]

    # Update the continuous_values array for the next iteration
    weights_biases = weights_biases[num_weights + num_biases :]

    # Set the sliced weights and biases in the layer
    layer.set_weights(
        [sliced_weights.reshape(weights.shape), sliced_biases.reshape(biases.shape)]
    )


X_train, X_test, y_train, y_test, scaler = get_fingerprinted_data()


model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])


evaluation_metrics = model.evaluate(X_train, y_train, verbose=1)


# this should produce (1, 0)
some_position = [[75, 87, 80, 6920, 17112, 17286]]
transformed_some_position = scaler.transform(some_position)


coord_value = model.predict(transformed_some_position)


print(coord_value)
