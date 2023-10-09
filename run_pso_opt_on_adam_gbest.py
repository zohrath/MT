from functools import partial
import multiprocessing
import sys
import numpy as np
import tensorflow as tf
import time
from CostFunctions import get_fingerprinted_data, get_fingerprinted_data_noisy
from GBestPSO import GBest_PSO
from Statistics import create_pso_run_stats
from pso_options import create_model
from random_search_values import get_vals_pso_opt_adam_params


def fitness_function(particle):
    try:
        learning_rate, beta_1, beta_2 = particle

        adam_optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            # epsilon=particle[3]  # Custom value for epsilon
        )

        model, _ = create_model()
        model.compile(optimizer=adam_optimizer, loss="mse", metrics=["accuracy"])

        X_train, X_test, y_train, y_test, scaler = get_fingerprinted_data_noisy()

        # Define the EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",  # Metric to monitor (usually validation loss)
            patience=50,  # Number of epochs with no improvement after which training will stop
            restore_best_weights=True,
        )

        model.fit(
            X_train,
            y_train,
            epochs=500,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[early_stopping],
        )

        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

        if np.isnan(loss):
            loss = float("inf")

        return np.sqrt(loss)
    except tf.errors.InvalidArgumentError as e:
        # Handle the specific exception here
        print("Caught an InvalidArgumentError:", e)
        # You can choose to return a specific value or take other actions
        return float("inf")  # For example, return infinity in case of an error


model, _ = create_model()
iterations = 20
num_particles = 20
num_dimensions = 3
position_bounds = [(0.001, 0.1), (0.5, 0.999999999), (0.5, 0.99999999)]
velocity_bounds = [
    (-0.03, 0.03),
    (-0.2, 0.2),
    (-0.2, 0.2),
]
c1 = 2.0
c2 = 2.0
inertia = 0.8
threshold = 0.1
function = fitness_function

param_sets = [
    {
        "model": model,
        "iterations": iterations,
        "num_particles": num_particles,
        "num_dimensions": num_dimensions,
        "position_bounds": position_bounds,
        "velocity_bounds": velocity_bounds,
        "inertia": inertia,
        "c1": c1,
        "c2": c2,
        "threshold": threshold,
        "function": function,
    },
    {
        "model": model,
        "iterations": iterations,
        "num_particles": num_particles,
        "num_dimensions": num_dimensions,
        "position_bounds": position_bounds,
        "velocity_bounds": velocity_bounds,
        "inertia": 0.729,
        "c1": 1.49445,
        "c2": 1.49445,
        "threshold": threshold,
        "function": function,
    },
    {
        "model": model,
        "iterations": iterations,
        "num_particles": num_particles,
        "num_dimensions": num_dimensions,
        "position_bounds": position_bounds,
        "velocity_bounds": velocity_bounds,
        "inertia": 0.8,
        "c1": 1.8663,
        "c2": 1.94016,
        "threshold": threshold,
        "function": function,
    },
]


def run_pso(
    thread_id,
    iterations,
    num_particles,
    num_dimensions,
    position_bounds,
    velocity_bounds,
    inertia,
    c1,
    c2,
    threshold,
    function,
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
        threshold,
        function,
    )

    swarm.run_pso(model)

    return (
        swarm.swarm_best_fitness,
        swarm.swarm_best_position,
        swarm.swarm_fitness_history,
        swarm.swarm_position_history,
    )


# Run this for the three GBest params, 500 iterations for the ANN training, 20 PSO iterations and particles
# c1, c2 = 1.49445, w = 0.729
# c1, c2 = 2.0, w = 0.8
# c1 = 1.8663, c2 = 1.94016, w = 0.8


if __name__ == "__main__":
    total_pso_runs = 9
    for params in param_sets:
        (
            model,
            iterations,
            num_particles,
            num_dimensions,
            position_bounds,
            velocity_bounds,
            inertia,
            c1,
            c2,
            threshold,
            function,
        ) = params.values()

        run_fitness_threaded = partial(
            run_pso,
            iterations=iterations,
            num_particles=num_particles,
            num_dimensions=num_dimensions,
            position_bounds=position_bounds,
            velocity_bounds=velocity_bounds,
            inertia=inertia,
            c1=c1,
            c2=c2,
            threshold=threshold,
            function=function,
        )
        start_time = time.time()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            results = pool.map(run_fitness_threaded, range(total_pso_runs))
        end_time = time.time()
        elapsed_time = end_time - start_time

        (
            swarm_best_fitness,
            swarm_best_position,
            swarm_fitness_history,
            swarm_position_history,
        ) = zip(*results)

        mean_best_fitness = np.mean(swarm_best_fitness)
        min_best_fitness = np.min(swarm_best_fitness)
        max_best_fitness = np.max(swarm_best_fitness)
        best_swarm_fitness_index = np.where(swarm_best_fitness == min_best_fitness)
        best_swarm_position = swarm_best_position[best_swarm_fitness_index[0][0]]

        sys.stdout.write(
            f"Minimum fitness for {total_pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}. Best value: {best_swarm_position}\n"
        )

        create_pso_run_stats(
            swarm_position_history,
            swarm_fitness_history,
            {"function_name": "optimize adam parameters"},
            "gbest",
            total_pso_runs,
            best_swarm_position,
            iterations,
            num_particles,
            num_dimensions,
            position_bounds,
            velocity_bounds,
            inertia,
            c1,
            c2,
            threshold,
            elapsed_time,
        )
