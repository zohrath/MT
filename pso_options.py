import numpy as np
from keras.layers import Dense
from keras.models import Sequential


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

    model.add(Dense(2))

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    total_num_weights = sum(np.prod(w.shape) for w in model.get_weights()[::2])
    total_num_biases = sum(np.prod(b.shape) for b in model.get_weights()[1::2])
    ann_dimensions = total_num_weights + total_num_biases

    return model, ann_dimensions
