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
        1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        0.69411563,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        0.99896655,
        -1.0,
        0.95054142,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        0.9999873,
        0.99998691,
        -1.0,
        0.75467262,
        -0.98227224,
        1.0,
        -1.0,
        -0.00382953,
        0.60909226,
        0.93793427,
        -0.99557881,
        0.2997539,
        -1.0,
        0.3692278,
        0.87057735,
        -0.98893649,
        0.52028137,
        1.0,
        1.0,
        0.93000502,
        -1.0,
        1.0,
        1.0,
        -1.0,
        0.23074096,
        1.0,
        1.0,
        0.07382197,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -0.30484712,
        0.99746915,
        -1.0,
        1.0,
        0.27437035,
        0.53743673,
        0.28392163,
        -0.05086395,
        1.0,
        1.0,
        1.0,
        1.0,
        -0.91141818,
        -1.0,
        0.74360141,
        1.0,
        -1.0,
        1.0,
        -0.02782531,
        -0.6310995,
        0.16481653,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -0.2591846,
        0.60563253,
        1.0,
        -0.53090531,
        -1.0,
        1.0,
        -1.0,
        -0.2583152,
        -0.14494746,
        0.15187139,
        1.0,
        1.0,
        0.99272684,
        1.0,
        1.0,
        0.99992295,
        -0.11467316,
        -0.70107806,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        0.08477322,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        0.20947837,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -0.57296701,
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
distance = np.linalg.norm(coord_value - np.array([[1, 0]]))

print(distance)
