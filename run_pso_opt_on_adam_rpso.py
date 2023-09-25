# Run RPSO to optimize the params of the adam optimizer here
from functools import partial
import multiprocessing
import sys
import numpy as np
import tensorflow as tf
import time
from CostFunctions import get_fingerprinted_data
from RPSO import RPSO
from Statistics import create_pso_run_stats_rpso
from pso_options import create_model
from random_search_values import get_vals_pso_opt_adam_params

# Do the same type of random search for params with this as done with gbest, but group cp and cg together


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
        model.compile(optimizer=adam_optimizer, loss="mse")

        X_train, X_test, y_train, y_test, scaler = get_fingerprinted_data()

        # Define the EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",  # Metric to monitor (usually validation loss)
            patience=500,  # Number of epochs with no improvement after which training will stop
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

        loss = model.evaluate(X_test, y_test, verbose=1)

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
Cp_min = 0.1
Cp_max = 3.5
Cg_min = 0.9
Cg_max = 3.0
w_min = 0.3
w_max = 1.7
gwn_std_dev = 0.15
threshold = 1
function = fitness_function


def run_pso(thread_id, iterations, num_particles):
    print("Starting job", iterations, num_particles)

    swarm = RPSO(
        iterations,
        num_particles,
        num_dimensions,
        position_bounds,
        velocity_bounds,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        threshold,
        function,
        gwn_std_dev
    )

    swarm.run_pso(model)

    return (
        swarm.swarm_best_fitness,
        swarm.swarm_best_position,
        swarm.swarm_fitness_history,
        swarm.swarm_position_history,
    )

# A test run for some given c1, c2, w parameters of GBest PSO.
# The goal is to do enough total runs of a given param combination to get a feel for the results those are going for, kind of like a limit
# The combination of PSO iterations and particles will be this:
# 10 iterations and 10 particles
# 20 iterations and 10 particles
# 10 iterations and 20 particles
# 20 iterations and 20 particles
# The ANN optimizer runs for 100 epochs, and has an early stopping based on validation fitness, with a patience of 50 epochs

# Run this for the two paper-based PSO params:
# c1, c2 = 1.49445, w = 0.729
# c1, c2 = 2.0, w = 0.8
# Run this for random search values of c1, c2, w


if __name__ == "__main__":
    total_pso_runs = multiprocessing.cpu_count() - 1

    for iterPart in [(20, 20)]:
        iterations, num_particles = iterPart

        run_fitness_threaded = partial(
            run_pso, iterations=iterations, num_particles=num_particles)
        start_time = time.time()
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count() - 1
        ) as pool:
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
        best_swarm_fitness_index = np.where(
            swarm_best_fitness == min_best_fitness)
        best_swarm_position = swarm_best_position[best_swarm_fitness_index[0][0]]

        sys.stdout.write(
            f"Minimum fitness for {total_pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}. Best value: {best_swarm_position}\n"
        )

        create_pso_run_stats_rpso(
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
            Cp_min,
            Cp_max,
            Cg_min,
            Cg_max,
            w_min,
            w_max,
            threshold,
            elapsed_time,
        )
