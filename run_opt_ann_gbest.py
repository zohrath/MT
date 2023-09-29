# Run the RPSO to optimize the weights and biases of an ANN
from __future__ import division
import sys
import time
import numpy as np
import multiprocessing
from functools import partial
import pandas as pd
import tensorflow as tf
from CostFunctions import get_fingerprinted_data, get_fingerprinted_data_noisy

from GBestPSO import GBest_PSO
from Statistics import save_opt_ann_gbest_stats, save_opt_ann_rpso_stats
from pso_options import create_model


X_train, _, y_train, _, _ = get_fingerprinted_data_noisy()
model, num_dimensions = create_model()


def ann_weights_fitness_function(particle, model):
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        num_weights = weights.size
        num_biases = biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = np.array(particle[:num_weights])
        sliced_biases = np.array(
            particle[num_weights: num_weights + num_biases])

        # Update the continuous_values array for the next iteration
        particle = particle[num_weights + num_biases:]

        # Set the sliced weights and biases in the layer
        layer.set_weights(
            [sliced_weights.reshape(weights.shape),
             sliced_biases.reshape(biases.shape)]
        )
    try:
        model.compile(optimizer="adam", loss="mse")

        # Evaluate the model and get the evaluation metrics
        evaluation_metrics = model.evaluate(X_train, y_train, verbose=0)
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


def run_pso(_, iterations, position_bounds, velocity_bounds, fitness_threshold,
            num_particles, c1, c2, inertia):

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
        ann_weights_fitness_function

    )

    swarm.run_pso(model)

    return (
        swarm.swarm_best_fitness,
        swarm.swarm_best_position,
        swarm.swarm_fitness_history,
        swarm.swarm_position_history,
    )


if __name__ == "__main__":
    # ---- GBest options ----
    position_bounds = [(-1.0, 1.0)] * num_dimensions
    velocity_bounds = [(-0.2, 0.2)] * num_dimensions
    fitness_threshold = 0.1
    num_particles = 60
    c1 = 1.8663
    c2 = 1.94016
    w = 0.8
    iterations = 134

    run_pso_partial = partial(run_pso,
                              iterations=iterations,
                              position_bounds=position_bounds,
                              velocity_bounds=velocity_bounds,
                              fitness_threshold=fitness_threshold,
                              num_particles=num_particles,
                              c1=c1,
                              c2=c2,
                              inertia=w)

    pso_runs = 1

    start_time = time.time()
    with multiprocessing.Pool(
        processes=multiprocessing.cpu_count() - 1
    ) as pool:
        results = pool.map(run_pso_partial, range(pso_runs))
    end_time = time.time()
    elapsed_time = end_time - start_time

    fitness_histories = [result[2] for result in results]
    (
        swarm_best_fitness,
        swarm_best_position,
        swarm_fitness_history,
        swarm_position_history

    ) = zip(*results)
    mean_best_fitness = np.mean(swarm_best_fitness)
    min_best_fitness = np.min(swarm_best_fitness)
    max_best_fitness = np.max(swarm_best_fitness)
    index_of_best_fitness = swarm_best_fitness.index(min_best_fitness)
    best_weights = swarm_best_position[index_of_best_fitness].tolist()

    sys.stdout.write(
        f"Minimum fitness for {pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}"
    )

    save_opt_ann_gbest_stats(fitness_histories, "gbest", pso_runs, position_bounds, velocity_bounds,
                             fitness_threshold, num_particles, c1, c2, w, iterations, elapsed_time,
                             min_best_fitness, mean_best_fitness, max_best_fitness, best_weights)
