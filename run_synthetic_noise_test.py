# Run the RPSO to optimize the weights and biases of an ANN
from __future__ import division
import json
import os
import sys
import time
import numpy as np
import multiprocessing
from functools import partial
import pandas as pd
import tensorflow as tf
from CostFunctions import (
    get_fingerprinted_data_noisy_as_verification_set,
    get_fingerprinted_random_points_calm_data,
    get_fingerprinted_random_points_noisy_data,
    get_fingerprinted_synthetic_noise_data,
)
from GBestPSO import GBest_PSO

from RPSO import RPSO
from Statistics import save_opt_ann_rpso_stats
from pso_options import create_model
from run_verification_sets_collection import get_ann_stats


(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
) = get_fingerprinted_synthetic_noise_data(100000)
model, num_dimensions = create_model()


def normal_ann_fitness_function(X_train, y_train, X_test, y_test):
    try:
        model, _ = create_model()
        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train,
            y_train,
            epochs=500,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
        )

        loss = model.evaluate(X_test, y_test, verbose=1)

        if np.isnan(loss):
            loss = float("inf")

        return np.sqrt(loss), model.get_weights()
    except tf.errors.InvalidArgumentError as e:
        # Handle the specific exception here
        print("Caught an InvalidArgumentError:", e)
        # You can choose to return a specific value or take other actions
        return float("inf")  # For example, return infinity in case of an error


def train_ann_fitness_function(particle, model):
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        num_weights = weights.size
        num_biases = biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = particle[:num_weights]
        sliced_biases = particle[num_weights : num_weights + num_biases]
        sliced_weights = np.array(sliced_weights)
        sliced_biases = np.array(sliced_biases)
        # Update the continuous_values array for the next iteration
        particle = particle[num_weights + num_biases :]

        # Set the sliced weights and biases in the layer
        layer.set_weights(
            [sliced_weights.reshape(weights.shape), sliced_biases.reshape(biases.shape)]
        )
    try:
        model.compile(optimizer="adam", loss="mse")

        # Evaluate the model and get the evaluation metrics
        evaluation_metrics = model.evaluate(X_train, y_train, verbose=1)
        rmse = np.sqrt(evaluation_metrics)

        return rmse
    except tf.errors.InvalidArgumentError as e:
        # Handle the specific exception here
        print("Caught an InvalidArgumentError:", e)
        # You can choose to return a specific value or take other actions
        return float("inf")  # For example, return infinity in case of an error
    except tf.errors.OpError as e:
        # Handle TensorFlow-specific errors here
        print(f"TensorFlow error: {e}")
        return float("inf")  # For example, return infinity in case of an error
    except Exception as e:
        # Handle other exceptions here
        print(f"An error occurred: {e}")
        return float("inf")  # For example, return infinity in case of an error


def run_pso(
    _,
    iterations,
    position_bounds,
    velocity_bounds,
    fitness_threshold,
    num_particles,
    c1,
    c2,
    inertia,
):
    swarm = GBest_PSO(
        iterations,
        num_particles,
        num_dimensions,
        position_bounds,
        velocity_bounds,
        inertia,
        c1,
        c2,
        fitness_threshold,
        train_ann_fitness_function,
    )

    swarm.run_pso(model)

    return (
        swarm.swarm_best_fitness,
        swarm.swarm_best_position,
        swarm.swarm_fitness_history,
        swarm.swarm_position_history,
    )


if __name__ == "__main__":
    for _ in range(9):
        fitness_normal_run, adam_training_weights = normal_ann_fitness_function(
            X_train, y_train, X_test, y_test
        )
        adam_training_weights = np.concatenate(
            [w.flatten() for w in adam_training_weights]
        )

        # ---- GBest options ----
        position_bounds = [(-1.0, 1.0)] * num_dimensions
        velocity_bounds = [(-0.2, 0.2)] * num_dimensions
        fitness_threshold = 0.1
        num_particles = 60
        c1 = 2.0
        c2 = 2.0
        w = 0.8
        iterations = 100

        run_pso_partial = partial(
            run_pso,
            iterations=iterations,
            position_bounds=position_bounds,
            velocity_bounds=velocity_bounds,
            fitness_threshold=fitness_threshold,
            num_particles=num_particles,
            c1=c1,
            c2=c2,
            inertia=w,
        )

        pso_runs = 1

        start_time = time.time()
        with multiprocessing.Pool(processes=1) as pool:
            results = pool.map(run_pso_partial, range(pso_runs))
        end_time = time.time()
        elapsed_time = end_time - start_time

        fitness_histories = [result[2] for result in results]
        (
            swarm_best_fitness,
            swarm_best_position,
            swarm_fitness_history,
            swarm_position_history,
        ) = zip(*results)
        mean_best_fitness = np.mean(swarm_best_fitness)
        min_best_fitness = np.min(swarm_best_fitness)
        max_best_fitness = np.max(swarm_best_fitness)
        index_of_best_fitness = swarm_best_fitness.index(min_best_fitness)
        pso_training_weights = swarm_best_position[index_of_best_fitness]

        (
            X_test_calm_random,
            y_test_calm_random,
        ) = get_fingerprinted_random_points_calm_data()
        (
            X_test_noisy_random,
            y_test_noisy_random,
        ) = get_fingerprinted_random_points_noisy_data()
        (
            X_test_fingerprinted_noisy,
            y_test_fingerprinted_noisy,
        ) = get_fingerprinted_data_noisy_as_verification_set()

        (
            mae_adam_training_calm_random,
            mse_adam_training_calm_random,
            rmse_adam_training_calm_random,
            medae_adam_training_calm_random,
            min_error_adam_training_calm_random,
            max_error_adam_training_calm_random,
            predicted_locations_adam_training_calm_random,
            absolute_errors_adam_training_calm_random,
        ) = get_ann_stats(adam_training_weights, X_test_calm_random, y_test_calm_random)

        (
            mae_adam_training_noisy_random,
            mse_adam_training_noisy_random,
            rmse_adam_training_noisy_random,
            medae_adam_training_noisy_random,
            min_error_adam_training_noisy_random,
            max_error_adam_training_noisy_random,
            predicted_locations_adam_training_noisy_random,
            absolute_errors_adam_training_noisy_random,
        ) = get_ann_stats(
            adam_training_weights, X_test_noisy_random, y_test_noisy_random
        )

        (
            mae_adam_training_fingerprinted_noisy,
            mse_adam_training_fingerprinted_noisy,
            rmse_adam_training_fingerprinted_noisy,
            medae_adam_training_fingerprinted_noisy,
            min_error_adam_training_fingerprinted_noisy,
            max_error_adam_training_fingerprinted_noisy,
            predicted_locations_adam_training_fingerprinted_noisy,
            absolute_errors_adam_training_fingerprinted_noisy,
        ) = get_ann_stats(
            adam_training_weights,
            X_test_fingerprinted_noisy,
            y_test_fingerprinted_noisy,
        )

        mea_adam_mean = np.mean(
            [
                mae_adam_training_calm_random,
                mae_adam_training_noisy_random,
                mae_adam_training_fingerprinted_noisy,
            ]
        )
        mse_adam_mean = np.mean(
            [
                mse_adam_training_calm_random,
                mse_adam_training_noisy_random,
                mse_adam_training_fingerprinted_noisy,
            ]
        )
        rmse_adam_mean = np.mean(
            [
                rmse_adam_training_calm_random,
                rmse_adam_training_noisy_random,
                rmse_adam_training_fingerprinted_noisy,
            ]
        )
        medae_adam_mean = np.mean(
            [
                medae_adam_training_calm_random,
                medae_adam_training_noisy_random,
                medae_adam_training_fingerprinted_noisy,
            ]
        )
        min_error_adam_mean = np.mean(
            [
                min_error_adam_training_calm_random,
                min_error_adam_training_noisy_random,
                min_error_adam_training_fingerprinted_noisy,
            ]
        )
        max_error_adam_mean = np.mean(
            [
                max_error_adam_training_calm_random,
                max_error_adam_training_noisy_random,
                max_error_adam_training_fingerprinted_noisy,
            ]
        )

        (
            mae_pso_training_calm_random,
            mse_pso_training_calm_random,
            rmse_pso_training_calm_random,
            medae_pso_training_calm_random,
            min_error_pso_training_calm_random,
            max_error_pso_training_calm_random,
            predicted_locations_pso_training_calm_random,
            absolute_errors_pso_training_calm_random,
        ) = get_ann_stats(pso_training_weights, X_test_calm_random, y_test_calm_random)

        (
            mae_pso_training_noisy_random,
            mse_pso_training_noisy_random,
            rmse_pso_training_noisy_random,
            medae_pso_training_noisy_random,
            min_error_pso_training_noisy_random,
            max_error_pso_training_noisy_random,
            predicted_locations_pso_training_noisy_random,
            absolute_errors_pso_training_noisy_random,
        ) = get_ann_stats(
            pso_training_weights, X_test_noisy_random, y_test_noisy_random
        )

        (
            mae_pso_training_fingerprinted_noisy,
            mse_pso_training_fingerprinted_noisy,
            rmse_pso_training_fingerprinted_noisy,
            medae_pso_training_fingerprinted_noisy,
            min_error_pso_training_fingerprinted_noisy,
            max_error_pso_training_fingerprinted_noisy,
            predicted_locations_pso_training_fingerprinted_noisy,
            absolute_errors_pso_training_fingerprinted_noisy,
        ) = get_ann_stats(
            pso_training_weights, X_test_fingerprinted_noisy, y_test_fingerprinted_noisy
        )

        mea_pso_mean = np.mean(
            [
                mae_pso_training_calm_random,
                mae_pso_training_noisy_random,
                mae_pso_training_fingerprinted_noisy,
            ]
        )
        mse_pso_mean = np.mean(
            [
                mse_pso_training_calm_random,
                mse_pso_training_noisy_random,
                mse_pso_training_fingerprinted_noisy,
            ]
        )
        rmse_pso_mean = np.mean(
            [
                rmse_pso_training_calm_random,
                rmse_pso_training_noisy_random,
                rmse_pso_training_fingerprinted_noisy,
            ]
        )
        medae_pso_mean = np.mean(
            [
                medae_pso_training_calm_random,
                medae_pso_training_noisy_random,
                medae_pso_training_fingerprinted_noisy,
            ]
        )
        min_error_pso_mean = np.mean(
            [
                min_error_pso_training_calm_random,
                min_error_pso_training_noisy_random,
                min_error_pso_training_fingerprinted_noisy,
            ]
        )
        max_error_pso_mean = np.mean(
            [
                max_error_pso_training_calm_random,
                max_error_pso_training_noisy_random,
                max_error_pso_training_fingerprinted_noisy,
            ]
        )
        json_data_unoptimized_adam = {
            "File name": "Synthetic noise test unoptimized Adam",
            "MAE mean": mea_adam_mean,
            "MSE mean": mse_adam_mean,
            "RMSE mean": rmse_adam_mean,
            "Median error mean": medae_adam_mean,
            "Min error mean": min_error_adam_mean,
            "Max error mean": max_error_adam_mean,
            "training_data": X_train.tolist(),
        }

        json_data_rpso = {
            "File name": "Synthetic noise test GBest",
            "MAE mean": mea_pso_mean,
            "MSE mean": mse_pso_mean,
            "RMSE mean": rmse_pso_mean,
            "Median error mean": medae_pso_mean,
            "Min error mean": min_error_pso_mean,
            "Max error mean": max_error_pso_mean,
            "c1": c1,
            "c2": c2,
            "w": w,
            "training_data": X_train.tolist(),
        }
        result_dict = {"Noise test stats": [json_data_unoptimized_adam, json_data_rpso]}
        subfolder = "synthetic_noise_test_results"

        if not os.path.exists(subfolder):
            os.makedirs(subfolder, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M")

        output_json_file_name = os.path.join(subfolder, f"{timestamp}_noise_stats.json")

        with open(output_json_file_name, "w") as json_file:
            json.dump(result_dict, json_file, indent=4)
