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

from CostFunctions import get_fingerprinted_data, get_fingerprinted_data_noisy, get_fingerprinted_random_points_calm_data, get_fingerprinted_random_points_noisy_data


def run_ann_fitting(id, learning_rate, beta_1, beta_2):
    model = Sequential()
    model.add(Dense(6, activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(2))

    adam_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
    )

    X_train, X_test, y_train, y_test, scaler = get_fingerprinted_data_noisy()

    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=500,
    )

    # Define the ModelCheckpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        # Filepath to save the best model weights
        filepath=f'best_model_{id}.h5',
        # Metric to monitor for improvement (e.g., validation loss)
        monitor='val_loss',
        save_best_only=True,      # Save only the best model
        save_weights_only=False,   # Save only the model's weights, not the entire model
        mode='auto',               # 'min' for loss, 'max' for accuracy, 'auto' to infer
        verbose=0                # Verbosity mode (1 shows progress)
    )

    model.compile(optimizer=adam_optimizer, loss="mse")
    model.fit(
        X_train,
        y_train,
        epochs=500,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1,
    )

    best_model = tf.keras.models.load_model(f'best_model_{id}.h5')

    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    return np.sqrt(mse), best_model


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


def generate_box_plot(sub_dir, losses, learning_rate, beta_1, beta_2):
    plt.figure(figsize=(10, 3))
    plt.boxplot(losses, vert=False, sym="")
    plt.title("Unoptimized Adam ANN optimization")
    plt.xlabel("Loss")
    plt.xlim(0, 5)
    plt.xticks(range(0, 5, 1))

    # Calculate statistics
    min_loss = min(losses)
    mean_loss = np.mean(losses)
    max_loss = max(losses)
    std_dev = np.std(losses)  # Calculate standard deviation

    # Adjust the horizontal positions for the "Best" and "Worst" labels
    offset = 0.1  # Vertical offset for text labels
    best_x = min_loss + 0.03  # Slightly to the right
    worst_x = max_loss - 0.03  # Slightly to the left
    plt.text(best_x, 1 + offset, f'Best: {min_loss:.3f}',
             horizontalalignment='left', verticalalignment='center')
    plt.text(worst_x, 1 + offset, f'Worst: {max_loss:.3f}',
             horizontalalignment='right', verticalalignment='center')

    # # Create a legend with the statistics
    legend_text = f"Min: {min_loss:.2f}\nMean: {mean_loss:.2f}\nMax: {max_loss:.2f}\nStd Dev: {std_dev:.2f}\nLearning Rate: {learning_rate}\nBeta_1: {beta_1}\nBeta_2: {beta_2}"

    # # Add the legend to the plot
    plt.text(0.8, 0.5, legend_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    box_plot_filename = os.path.join(sub_dir, f"box_plot_{timestamp}.png")

    plt.savefig(box_plot_filename, bbox_inches="tight")

    return box_plot_filename


if __name__ == "__main__":
    # learning_rate = 0.05677689792466579
    # beta_1 = 0.6751472722094489
    # beta_2 = 0.848038363244917
  
    learning_rate = 0.03493588115825588
    beta_1 = 0.7284982255951089
    beta_2 = 0.8632245088246844
    rmse_results = []
    for run_id in range(100):
        rmse, best_model = run_ann_fitting(
            run_id, learning_rate, beta_1, beta_2)
        rmse_results.append(rmse)
    print(np.min(rmse_results))
    sub_dir = "optimized_adam_model_for_verification"
    os.makedirs(sub_dir, exist_ok=True)

    box_plot_filename = generate_box_plot(
        sub_dir, rmse_results, learning_rate, beta_1, beta_2)

    best_model_filename = save_model(sub_dir, best_model)

    save_json_results(
        sub_dir,
        {"learning_rate": learning_rate, "beta_1": beta_1, "beta_2": beta_2},
        rmse_results,
        best_model_filename=best_model_filename,
        box_plot_filename=box_plot_filename,
    )
