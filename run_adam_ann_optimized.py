import os
import json
import multiprocessing
import numpy as np
import tensorflow as tf
from functools import partial
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from datetime import datetime
import matplotlib.pyplot as plt

from CostFunctions import get_fingerprinted_data


def run_ann_fitting(_, learning_rate, beta_1, beta_2):
    model = Sequential()
    model.add(Dense(6, activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(2))

    adam_optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
    )

    X_train, X_test, y_train, y_test, scaler = get_fingerprinted_data()

    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
    )

    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train,
        y_train,
        epochs=500,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0,
    )

    # this should produce (1, 0)
    some_position = [[75, 87, 80, 6920, 17112, 17286]]
    # this should produce (8,6)
    some_position_2 = [[72, 78, 81, 8503, 8420, 8924]]
    transformed_some_position = scaler.transform(some_position)
    transformed_some_position_2 = scaler.transform(some_position_2)

    coord_one = model.predict(transformed_some_position)
    coord_two = model.predict(transformed_some_position_2)

    squared_distance_one = (
        sum((p1 - p2) ** 2 for p1, p2 in zip((1, 0), coord_one[0]))
    ) ** 0.5

    squared_distance_two = (
        sum((p1 - p2) ** 2 for p1, p2 in zip((6, 8), coord_two[0]))
    ) ** 0.5

    sum_of_distances = squared_distance_one + squared_distance_two

    return sum_of_distances, model, squared_distance_one, squared_distance_two


def save_json_results(sub_dir, parameters, losses, best_model_filename):
    # Calculate minimum, mean, and maximum loss values
    min_loss = min(losses)
    mean_loss = np.mean(losses)
    max_loss = max(losses)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Create a dictionary for results with the filename and additional fields
    result_data = {
        "timestamp": timestamp,
        "parameters": parameters,
        "results": {
            "loss": losses,
            "best_model_filename": best_model_filename,
            "min_loss": min_loss,
            "mean_loss": mean_loss,
            "max_loss": max_loss,
        },
    }

    # Save the result_data dictionary to a JSON file in the sub-directory
    json_filename = os.path.join(sub_dir, f"results_{timestamp}.json")
    with open(json_filename, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    print(f"JSON file with results saved as '{json_filename}'.")


def save_model(sub_dir, best_model):
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Save the best model as a Keras .h5 file in the sub-directory
    model_filename = os.path.join(sub_dir, f"optimized_ann_gbest_{timestamp}.h5")
    best_model.save(model_filename)

    print(f"Best model saved as '{model_filename}'.")


def generate_box_plot(losses):
    # Create a box plot of the loss values
    plt.figure(figsize=(8, 6))
    plt.boxplot(losses, vert=False)
    plt.title("Box Plot of Loss Values")
    plt.xlabel("Loss")
    plt.show()


if __name__ == "__main__":
    learning_rate = 0.0613005
    beta_1 = 0.51088201
    beta_2 = 0.72861215

    run_ann = partial(
        run_ann_fitting, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
    )
    total_runs = 2
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = pool.map(run_ann, range(total_runs))

    # Create a sub-directory for saving files
    sub_dir = "gbest_optimized_adam_ann_stats"
    os.makedirs(sub_dir, exist_ok=True)

    # Extract the losses and model weights and biases from the results
    losses = [result[0] for result in results]

    # Find the index of the best model
    index_of_min = np.argmin(losses)
    best_model = results[index_of_min][1]

    # Save the JSON file with results
    save_json_results(
        sub_dir,
        {"learning_rate": learning_rate, "beta_1": beta_1, "beta_2": beta_2},
        losses,
        best_model_filename=f"{sub_dir}/optimized_ann_gbest.h5",
    )

    # Save the best model
    save_model(sub_dir, best_model)

    # Generate a box plot
    generate_box_plot(losses)
