from functools import partial
import multiprocessing
import sys
import numpy as np
import tensorflow as tf
from CostFunctions import get_fingerprinted_data
from GBestPSO import GBest_PSO
from Statistics import get_swarm_total_particle_distance, plot_all_fitness_histories, plot_average_total_distance, plot_averages_fitness_histories
from pso_options import create_model


def fitness_function(particle):
    learning_rate = particle[0]
    # legacy for Apple M1, M2
    adam_optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate,
        beta_1=particle[1],  # Custom value for beta_1
        beta_2=particle[2],  # Custom value for beta_2
        # epsilon=particle[3]  # Custom value for epsilon
    )

    model, _ = create_model()
    model.compile(optimizer=adam_optimizer,
                  loss='mse', metrics=['accuracy'])

    X_train, X_test, y_train, y_test, scaler = get_fingerprinted_data()

    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor (usually validation loss)
        patience=100,           # Number of epochs with no improvement after which training will stop
        restore_best_weights=True
    )

    model.fit(X_train, y_train, epochs=250, batch_size=32,
              validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    if np.isnan(loss):
        loss = float('inf')

    return loss


def run_pso(_):
    model, _ = create_model()
    iterations = 30
    num_particles = 10
    num_dimensions = 3
    position_bounds = [(0.001, 0.1), (0.5, 0.999999999),
                       (0.5, 0.99999999)]
    velocity_bounds = [(-0.03, 0.03), (-0.2, 0.2), (-0.2, 0.2),
                       ]
    inertia = 0.9
    c1 = 0.9
    c2 = 0.3
    threshold = 1
    function = fitness_function
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
    swarm_best_fitness, swarm_best_position, swarm_fitness_history, swarm_position_history = run_pso(
        0)
    run_fitness_threaded = partial(run_pso)
    total_pso_runs = multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(processes=total_pso_runs) as pool:
        results = pool.map(run_fitness_threaded, range(total_pso_runs))

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
        f"Minimum fitness for {total_pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}. Best value: {best_swarm_position}"
    )
    plot_all_fitness_histories(swarm_fitness_history, {
                               "function_name": "optimize adam"}, "gbest", total_pso_runs)
    plot_averages_fitness_histories(
        swarm_fitness_history, "GBest", total_pso_runs)
    plot_average_total_distance(swarm_position_history, "GBest")
    get_swarm_total_particle_distance(swarm_position_history)
