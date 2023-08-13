from __future__ import division
import sys
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import multiprocessing
from functools import partial
import argparse
from datetime import datetime
from CostFunctions import (
    ann_node_count_fitness,
    griewank,
    penalized1,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
    step,
)

from GBestPSO import GBest_PSO
from RPSO import RPSO
from Statistics import (
    plot_all_fitness_histories,
    plot_average_total_distance,
    plot_averages_fitness_histories,
)

PSO_TYPE = "rpso"


class Main:
    def __init__(
        self,
        iterations,
        options,
    ):
        self.iterations = iterations
        self.num_particles = options["num_particles"]
        self.num_dimensions = options["num_dimensions"]
        self.position_bounds = options["position_bounds"]
        self.velocity_bounds = options["velocity_bounds"]
        self.threshold = options["threshold"]
        self.function = options["function"]

    def run_pso(self, pso_type):
        if pso_type == "gbest":
            swarm = GBest_PSO(
                self.iterations,
                self.num_particles,
                self.num_dimensions,
                self.position_bounds,
                self.velocity_bounds,
                options["inertia"],
                options["c1"],
                options["c2"],
                self.threshold,
                self.function,
            )
            swarm.run_pso()

            return (
                swarm.swarm_best_fitness,
                swarm.swarm_best_position,
                swarm.swarm_fitness_history,
                swarm.swarm_position_history,
            )
        elif pso_type == "rpso":
            swarm = RPSO(
                self.iterations,
                self.num_particles,
                self.num_dimensions,
                self.position_bounds,
                self.velocity_bounds,
                options["Cp_min"],
                options["Cp_max"],
                options["Cg_min"],
                options["Cg_max"],
                options["w_min"],
                options["w_max"],
                options["threshold"],
                options["function"],
            )
            swarm.run_pso()

            return (
                swarm.swarm_best_fitness,
                swarm.swarm_best_position,
                swarm.swarm_fitness_history,
                swarm.swarm_position_history,
            )


pso_functions = [
    {
        "function": sphere,
        "function_name": "Sphere",
        "position_bounds": (-100, 100),
        "velocity_bounds": (-40, 40),
        "threshold": 0.01,
        "num_particles": 3,
        "num_dimensions": 3,
        "Cp_min": 0.5,
        "Cp_max": 2.5,
        "Cg_min": 0.5,
        "Cg_max": 2.5,
        "w_min": 0.4,
        "w_max": 0.9,
        "inertia": 0.9,
        "c1": 2.0,
        "c2": 2.0,
    },
    {
        "function": rosenbrock,
        "function_name": "Rosenbrock",
        "position_bounds": (-30, 30),
        "velocity_bounds": (-12, 12),
        "threshold": 100,
        "num_particles": 3,
        "num_dimensions": 3,
        "Cp_min": 0.5,
        "Cp_max": 2.5,
        "Cg_min": 0.5,
        "Cg_max": 2.5,
        "w_min": 0.4,
        "w_max": 0.9,
        "inertia": 0.9,
        "c1": 2.0,
        "c2": 2.0,
    },
    {
        "function": rastrigin,
        "function_name": "Rastrigin",
        "position_bounds": (-5.12, 5.12),
        "velocity_bounds": (-2.048, 2.048),
        "threshold": 50,
        "num_particles": 30,
        "num_dimensions": 30,
        "Cp_min": 0.5,
        "Cp_max": 2.5,
        "Cg_min": 0.5,
        "Cg_max": 2.5,
        "w_min": 0.4,
        "w_max": 0.9,
        "inertia": 0.9,
        "c1": 2.0,
        "c2": 2.0,
    },
    {
        "function": schwefel,
        "function_name": "Schwefel",
        "position_bounds": (-100, 100),
        "velocity_bounds": (-40, 40),
        "threshold": 0.01,
        "num_particles": 30,
        "num_dimensions": 30,
        "Cp_min": 0.5,
        "Cp_max": 2.5,
        "Cg_min": 0.5,
        "Cg_max": 2.5,
        "w_min": 0.4,
        "w_max": 0.9,
        "inertia": 0.9,
        "c1": 2,
        "c2": 2,
    },
    {
        "function": griewank,
        "function_name": "Griewank",
        "position_bounds": (-600, 600),
        "velocity_bounds": (-240, 240),
        "threshold": 0.01,
        "num_particles": 30,
        "num_dimensions": 30,
        "Cp_min": 0.5,
        "Cp_max": 2.5,
        "Cg_min": 0.5,
        "Cg_max": 2.5,
        "w_min": 0.4,
        "w_max": 0.9,
        "inertia": 0.9,
        "c1": 2,
        "c2": 2,
    },
    {
        "function": penalized1,
        "function_name": "Penalized1",
        "position_bounds": (-50, 50),
        "velocity_bounds": (-20, 20),
        "threshold": 0.01,
        "num_particles": 3,
        "num_dimensions": 3,
        "Cp_min": 0.5,
        "Cp_max": 2.5,
        "Cg_min": 0.5,
        "Cg_max": 2.5,
        "w_min": 0.4,
        "w_max": 0.9,
        "inertia": 0.9,
        "c1": 1.49445,
        "c2": 1.49445,
    },
    {
        "function": step,
        "function_name": "Step",
        "position_bounds": (-100, 100),
        "velocity_bounds": (-40, 40),
        "threshold": 0.01,
        "num_particles": 3,
        "num_dimensions": 3,
        "Cp_min": 0.5,
        "Cp_max": 2.5,
        "Cg_min": 0.5,
        "Cg_max": 2.5,
        "w_min": 0.4,
        "w_max": 0.9,
        "inertia": 0.9,
        "c1": 1.49445,
        "c2": 1.49445,
    },
    {
        "function": ann_node_count_fitness,
        "function_name": "ANN Node Count RPSO",
        "position_bounds": (1, 100),
        "velocity_bounds": (-40, 40),
        "threshold": 1,
        "num_particles": 3,
        "num_dimensions": 3,
        "Cp_min": 0.5,
        "Cp_max": 2.5,
        "Cg_min": 0.5,
        "Cg_max": 2.5,
        "w_min": 0.4,
        "w_max": 0.9,
    },
    {
        "function": ann_node_count_fitness,
        "function_name": "ANN Node Count Gbest",
        "position_bounds": (1, 100),
        "velocity_bounds": (-40, 40),
        "threshold": 1,
        "num_particles": 3,
        "num_dimensions": 3,
        "inertia": 0.9,
        "c1": 1.49445,
        "c2": 1.49445,
    },
]

# Common options for all PSO runs
iterations = 50
pso_runs = 5
options = pso_functions[0]


def run_pso_threaded(_, pso_type):
    swarm = Main(
        iterations,
        options,
    )
    (
        swarm_best_fitness,
        swarm_best_position,
        swarm_fitness_history,
        swarm.swarm_position_history,
    ) = swarm.run_pso(pso_type)

    return (
        swarm_best_fitness,
        swarm_best_position,
        swarm_fitness_history,
        swarm.swarm_position_history,
    )


def handle_data(fitness_histories, swarm_position_histories):
    plot_average_total_distance(swarm_position_histories, PSO_TYPE)
    plot_averages_fitness_histories(fitness_histories, PSO_TYPE)
    plot_all_fitness_histories(fitness_histories, options, PSO_TYPE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PSO algorithm with single or threaded execution."
    )
    parser.add_argument(
        "mode",
        choices=["single", "threaded"],
        help="Select execution mode: single or threaded",
    )
    args = parser.parse_args()

    main = Main(
        iterations,
        options,
    )

    if args.mode == "single":
        (
            swarm_best_fitness,
            swarm_best_position,
            swarm_fitness_history,
            swarm_position_history,
        ) = main.run_pso(PSO_TYPE)

        sys.stdout.write("Best fitness: {}\n".format(swarm_best_fitness))

        print(swarm_best_position)

    elif args.mode == "threaded":
        num_cores = multiprocessing.cpu_count()
        run_pso_partial = partial(run_pso_threaded, pso_type=PSO_TYPE)

        with multiprocessing.Pool(processes=num_cores - 1) as pool:
            results = pool.map(run_pso_partial, range(pso_runs))

        fitness_histories = [result[2] for result in results]
        (
            swarm_best_fitness,
            swarm_best_position,
            swarm_fitness_history,
            swarm_position_history,
        ) = zip(*results)
        np.save("history_data.npy", swarm_position_history)
        if fitness_histories:
            handle_data(fitness_histories, swarm_position_history)

        mean_best_fitness = np.mean(swarm_best_fitness)
        min_best_fitness = np.min(swarm_best_fitness)
        max_best_fitness = np.max(swarm_best_fitness)

        sys.stdout.write(
            f"Minimum fitness for {pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}"
        )
        # sys.stdout.write(f"The best position was {swarm_best_position}\n\r")
    else:
        print(
            "Invalid mode. Please choose either 'single' or 'threaded' as the execution mode."
        )
