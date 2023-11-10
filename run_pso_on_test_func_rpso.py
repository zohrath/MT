# Run the RPSO to optimize the weights and biases of an ANN
from __future__ import division
import sys
import time
import numpy as np
import multiprocessing
from functools import partial
import pandas as pd
import tensorflow as tf
from CostFunctions import (
    sphere,
    rosenbrock_function,
    rastrigin_function,
    schwefel_function,
    griewank_function,
    penalized_1_function,
    step_function,
)

from RPSO import RPSO
from Statistics import save_opt_ann_rpso_stats, save_test_func_rpso_stats
from pso_options import create_model


model, _ = create_model()


def run_pso(
    _,
    iterations,
    position_bounds,
    velocity_bounds,
    fitness_threshold,
    num_particles,
    Cp_min,
    Cp_max,
    Cg_min,
    Cg_max,
    w_min,
    w_max,
    gwn_std_dev,
    num_dimensions,
):
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
        fitness_threshold,
        griewank_function,
        gwn_std_dev,
    )

    swarm.run_pso(model)

    return (
        swarm.swarm_best_fitness,
        swarm.swarm_best_position,
        swarm.swarm_fitness_history,
        swarm.swarm_position_history,
    )


if __name__ == "__main__":
    # ---- RSPO options ----
    num_dimensions = 30
    position_bounds = [(-600, 600)] * num_dimensions
    velocity_bounds = [(-240, 240)] * num_dimensions
    fitness_threshold = 0.01
    num_particles = 30
    Cp_min = 0.5
    Cp_max = 2.5
    Cg_min = 0.5
    Cg_max = 2.5
    w_min = 0.4
    w_max = 0.9
    gwn_std_dev = 0.07
    iterations = 10000

    run_pso_partial = partial(
        run_pso,
        iterations=iterations,
        position_bounds=position_bounds,
        velocity_bounds=velocity_bounds,
        fitness_threshold=fitness_threshold,
        num_particles=num_particles,
        Cp_min=Cp_min,
        Cp_max=Cp_max,
        Cg_min=Cg_min,
        Cg_max=Cg_max,
        w_min=w_min,
        w_max=w_max,
        gwn_std_dev=gwn_std_dev,
        num_dimensions=num_dimensions,
    )

    pso_runs = 50

    start_time = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
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
    best_weights = swarm_best_position[index_of_best_fitness].tolist()

    sys.stdout.write(
        f"Minimum fitness for {pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}"
    )

    save_test_func_rpso_stats(
        fitness_histories,
        "rpso",
        pso_runs,
        position_bounds,
        velocity_bounds,
        fitness_threshold,
        num_particles,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        gwn_std_dev,
        iterations,
        elapsed_time,
        min_best_fitness,
        mean_best_fitness,
        max_best_fitness,
        best_weights,
        "Step",
    )
