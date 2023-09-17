from functools import partial
import multiprocessing
import sys
import numpy as np
import tensorflow as tf
import time
from CostFunctions import get_fingerprinted_data
from GBestPSO import GBest_PSO
from Statistics import create_pso_run_stats
from pso_options import create_model


# STATISTIKEN BEHÖVER FAKTISKT INKLUDERA DE BÄSTA VÄRDENA
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
        model.compile(optimizer=adam_optimizer,
                      loss="mse", metrics=["accuracy"])

        X_train, X_test, y_train, y_test, scaler = get_fingerprinted_data()

        # Define the EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",  # Metric to monitor (usually validation loss)
            patience=100,  # Number of epochs with no improvement after which training will stop
            restore_best_weights=True,
        )

        model.fit(
            X_train,
            y_train,
            epochs=250,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[early_stopping],
        )

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        if np.isnan(loss):
            loss = float("inf")

        return np.sqrt(loss)
    except tf.errors.InvalidArgumentError as e:
        # Handle the specific exception here
        print("Caught an InvalidArgumentError:", e)
        # You can choose to return a specific value or take other actions
        return float("inf")  # For example, return infinity in case of an error


model, _ = create_model()
iterations = 10
num_particles = 10
num_dimensions = 3
position_bounds = [(0.001, 0.1), (0.5, 0.999999999), (0.5, 0.99999999)]
velocity_bounds = [
    (-0.03, 0.03),
    (-0.2, 0.2),
    (-0.2, 0.2),
]
inertia = 0.8
c1 = 2.0
c2 = 2.0
threshold = 1
function = fitness_function


def run_pso(thread_id):
    print("Starting job", thread_id)

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


if __name__ == "__main__":
    run_fitness_threaded = partial(run_pso)
    total_pso_runs = 7
    for iter in [10, 20, 30, 40, 50, 60]:
        iterations = iter
        for _ in range(10):
            start_time = time.time()
            with multiprocessing.Pool(
                processes=multiprocessing.cpu_count() - 1
            ) as pool:
                results = pool.map(run_fitness_threaded, range(total_pso_runs))
            end_time = time.time()
            elapsed_time = end_time - start_time
            # LÄGG TILL OM ADAM ÄR OPTIMERAD ELLER INTE I STATISTIK
            (
                swarm_best_fitness,
                swarm_best_position,
                swarm_fitness_history,
                swarm_position_history,
            ) = zip(*results)

            mean_best_fitness = np.mean(swarm_best_fitness)
            min_best_fitness = np.min(swarm_best_fitness)
            max_best_fitness = np.max(swarm_best_fitness)
            best_swarm_fitness_index = np.where(
                swarm_best_fitness == min_best_fitness)
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
    iterations = 10
    for particles in [10, 20, 30, 40, 50, 60]:
        num_particles = particles
        for _ in range(10):
            start_time = time.time()
            with multiprocessing.Pool(
                processes=multiprocessing.cpu_count() - 1
            ) as pool:
                results = pool.map(run_fitness_threaded, range(total_pso_runs))
            end_time = time.time()
            elapsed_time = end_time - start_time
            # LÄGG TILL OM ADAM ÄR OPTIMERAD ELLER INTE I STATISTIK
            (
                swarm_best_fitness,
                swarm_best_position,
                swarm_fitness_history,
                swarm_position_history,
            ) = zip(*results)

            mean_best_fitness = np.mean(swarm_best_fitness)
            min_best_fitness = np.min(swarm_best_fitness)
            max_best_fitness = np.max(swarm_best_fitness)
            best_swarm_fitness_index = np.where(
                swarm_best_fitness == min_best_fitness)
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
