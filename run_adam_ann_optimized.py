import os
import json
import multiprocessing
import numpy as np
from sklearn.metrics import mean_squared_error
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
        patience=200,
    )

    model.compile(optimizer=adam_optimizer, loss="mse")
    model.fit(
        X_train,
        y_train,
        epochs=500,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0,
    )

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    return mse, model


def save_json_results(
    sub_dir, parameters, losses, best_model_filename, box_plot_filename
):
    min_loss = min(losses)
    mean_loss = np.mean(losses)
    max_loss = max(losses)

    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    result_data = {
        "timestamp": timestamp,
        "parameters": parameters,
        "results": {
            "loss": losses,
            "best_model_filename": best_model_filename,
            "min_loss": min_loss,
            "mean_loss": mean_loss,
            "max_loss": max_loss,
            "box_plot_filename": box_plot_filename,
        },
    }

    json_filename = os.path.join(sub_dir, f"results_{timestamp}.json")
    with open(json_filename, "w") as json_file:
        json.dump(result_data, json_file, indent=4)


def save_model(sub_dir, best_model):
    # Generate a timestamp without seconds
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    model_filename = f"optimized_ann_gbest_{timestamp}.h5"
    model_filename_full = os.path.join(sub_dir, model_filename)
    best_model.save(model_filename_full)

    print(f"Best model saved as '{model_filename_full}'.")
    return model_filename  # Return the filename for inclusion in JSON


def generate_box_plot(sub_dir, losses):
    plt.figure(figsize=(10, 4))
    plt.boxplot(losses, vert=False, sym="")
    plt.title("Box Plot of Loss Values")
    plt.xlabel("Loss")
    plt.xlim(0, 10)
    plt.xticks(range(0, 10, 1))

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    box_plot_filename = os.path.join(sub_dir, f"box_plot_{timestamp}.png")

    plt.savefig(box_plot_filename, bbox_inches="tight")

    return box_plot_filename


if __name__ == "__main__":
    learning_rate = 0.08763297274684727
    beta_1 = 0.5351799417510288
    beta_2 = 0.8821722454935323

    run_ann = partial(
        run_ann_fitting, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
    )
    total_runs = 21
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = pool.map(run_ann, range(total_runs))

    # Extract the test RMSE scores and best models from the results
    mse_scores = [result[0] for result in results]
    best_model = results[np.argmin(mse_scores)][1]

    sub_dir = "gbest_optimized_adam_ann_stats"
    os.makedirs(sub_dir, exist_ok=True)

    box_plot_filename = generate_box_plot(sub_dir, mse_scores)

    best_model_filename = save_model(sub_dir, best_model)

    save_json_results(
        sub_dir,
        {"learning_rate": learning_rate, "beta_1": beta_1, "beta_2": beta_2},
        mse_scores,
        best_model_filename=best_model_filename,
        box_plot_filename=box_plot_filename,
    )
