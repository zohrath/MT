# Run the GBest to optimize the weights and biases of an ANN
from __future__ import division
import sys
import time
import numpy as np
import multiprocessing
from functools import partial
from CostFunctions import sphere, rosenbrock_function, rastrigin_function, schwefel_function, griewank_function, penalized_1_function, step_function
from GBestPSO import GBest_PSO
from Statistics import save_opt_ann_gbest_stats, save_opt_ann_rpso_stats, save_test_func_gbest_stats
from pso_options import create_model

model, _ = create_model()


def run_pso(_, iterations, position_bounds, velocity_bounds, fitness_threshold,
            num_particles, c1, c2, inertia, num_dimensions):

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
        schwefel_function

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
    num_dimensions = 30
    position_bounds = [(-100, 100)] * num_dimensions
    velocity_bounds = [(-40, 40)] * num_dimensions
    fitness_threshold = 0.01
    num_particles = 30
    c1 = 2.0
    c2 = 2.0
    w = 0.8
    iterations = 10000

    run_pso_partial = partial(run_pso,
                              iterations=iterations,
                              position_bounds=position_bounds,
                              velocity_bounds=velocity_bounds,
                              fitness_threshold=fitness_threshold,
                              num_particles=num_particles,
                              c1=c1,
                              c2=c2,
                              inertia=w,
                              num_dimensions=num_dimensions)

    pso_runs = 50

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

    save_test_func_gbest_stats(fitness_histories, "gbest", pso_runs, position_bounds, velocity_bounds,
                               fitness_threshold, num_particles, c1, c2, w, iterations, elapsed_time,
                               min_best_fitness, mean_best_fitness, max_best_fitness, best_weights, "Schwefel")
