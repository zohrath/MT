import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from CostFunctions import get_fingerprinted_data


def get_final_model(model, particle):
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        num_weights = weights.size
        num_biases = biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = particle[:num_weights]
        sliced_biases = particle[num_weights : num_weights + num_biases]

        # Update the continuous_values array for the next iteration
        particle = particle[num_weights + num_biases :]

        # Set the sliced weights and biases in the layer
        layer.set_weights(
            [sliced_weights.reshape(weights.shape), sliced_biases.reshape(biases.shape)]
        )
    return model


def create_model(params):
    model = Sequential()
    model.add(Dense(params[0], activation="relu", input_shape=(6,)))

    for nodes in params[1:]:
        model.add(Dense(nodes, activation="relu"))

    model.add(Dense(2))  # Output layer

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return model


def ann_weights_fitness_function(particle, model, X_train, y_train):
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        num_weights = weights.size
        num_biases = biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = particle[:num_weights]
        sliced_biases = particle[num_weights : num_weights + num_biases]

        # Update the continuous_values array for the next iteration
        particle = particle[num_weights + num_biases :]

        # Set the sliced weights and biases in the layer
        layer.set_weights(
            [sliced_weights.reshape(weights.shape), sliced_biases.reshape(biases.shape)]
        )

    # Evaluate the model and get the evaluation metrics
    evaluation_metrics = model.evaluate(X_train, y_train, verbose=0)
    rmse = np.sqrt(evaluation_metrics[0])

    return rmse


# # Create the particle
# num_nodes_per_layer = [
#     6,
#     6,
#     6,
# ]  # Example: Input layer with 15 nodes, Hidden layer with 3 nodes

# # Create and build the model based on the particle configuration
# model = create_model(num_nodes_per_layer)
# # # Calculate the total number of values required
# total_num_weights = sum(np.prod(w.shape) for w in model.get_weights()[::2])
# total_num_biases = sum(np.prod(b.shape) for b in model.get_weights()[1::2])
# total_number_of_values = total_num_weights + total_num_biases

# # # Initialize a numpy array of continuous values (replace this with your actual data)
# continuous_values = np.random.random((total_number_of_values,))

# X_train, X_test, y_train, y_test = get_fingerprinted_data()

# # # Call the fitness_function to get predictions from the updated model
# loss = ann_weights_fitness_function(continuous_values, model, X_train, y_train)

# predictions = model.predict(X_test)
# print(explained_variance_score(y_test, predictions))

# some_position = [[75, 87, 80, 6920, 17112, 17286]]  # this should produce (1, 0)
# some_position_2 = [[72, 78, 81, 8503, 8420, 8924]]  # this should produce (8,6)

# scaler = MinMaxScaler()
# scaler.fit(X_train)
# transformed_some_position = scaler.transform(some_position_2)

# x_value = model.predict(transformed_some_position)
# print(x_value)
