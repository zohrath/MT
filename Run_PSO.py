from __future__ import division
import sys
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing
from functools import partial
import argparse
from datetime import datetime
from CostFunctions import (
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

PSO_TYPE = "rpso"


class Main:
    def __init__(
        self,
        iterations,
        num_particles,
        num_dimensions,
        inertia,
        c1,
        c2,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        options,
    ):
        self.iterations = iterations
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.position_bounds = options["position_bounds"]
        self.velocity_bounds = options["velocity_bounds"]
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.Cp_min = Cp_min
        self.Cp_max = Cp_max
        self.Cg_min = Cg_min
        self.Cg_max = Cg_max
        self.w_min = w_min
        self.w_max = w_max
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
                self.inertia,
                self.c1,
                self.c2,
            )
            swarm.run_pso()

            return (
                swarm.swarm_best_fitness,
                swarm.swarm_best_position,
                swarm.swarm_fitness_history,
            )
        elif pso_type == "rpso":
            swarm = RPSO(
                self.iterations,
                self.num_particles,
                self.num_dimensions,
                self.position_bounds,
                self.velocity_bounds,
                self.Cp_min,
                self.Cp_max,
                self.Cg_min,
                self.Cg_max,
                self.w_min,
                self.w_max,
                self.threshold,
                self.function,
            )
            swarm.run_pso()

            return (
                swarm.swarm_best_fitness,
                swarm.swarm_best_position,
                swarm.swarm_fitness_history,
            )


iterations = 10000
num_particles = 30
num_dimensions = 30
inertia = 0.5
c1 = 0.5
c2 = 0.5
pso_runs = 50
Cp_min = 0.5
Cp_max = 2.5
Cg_min = 0.5
Cg_max = 2.5
w_min = 0.4
w_max = 0.9

pso_functions = [
    {
        "function": sphere,
        "function_name": "Sphere",
        "position_bounds": (-100, 100),
        "velocity_bounds": (-40, 40),
        "threshold": 0.01,
    },
    {
        "function": rosenbrock,
        "function_name": "Rosenbrock",
        "position_bounds": (-30, 30),
        "velocity_bounds": (-12, 12),
        "threshold": 100,
    },
    {
        "function": rastrigin,
        "function_name": "Rastrigin",
        "position_bounds": (-5.12, 5.12),
        "velocity_bounds": (-2.048, 2.048),
        "threshold": 50,
    },
    {
        "function": schwefel,
        "function_name": "Schwefel",
        "position_bounds": (-100, 100),
        "velocity_bounds": (-40, 40),
        "threshold": 0.01,
    },
    {
        "function": griewank,
        "function_name": "Griewank",
        "position_bounds": (-600, 600),
        "velocity_bounds": (-240, 240),
        "threshold": 0.01,
    },
    {
        "function": penalized1,
        "function_name": "Penalized1",
        "position_bounds": (-50, 50),
        "velocity_bounds": (-20, 20),
        "threshold": 0.01,
    },
    {
        "function": step,
        "function_name": "Step",
        "position_bounds": (-100, 100),
        "velocity_bounds": (-40, 40),
        "threshold": 0.01,
    },
]


options = pso_functions[1]


def run_pso_threaded(_, pso_type):
    swarm = Main(
        iterations,
        num_particles,
        num_dimensions,
        inertia,
        c1,
        c2,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        options,
    )
    swarm_best_fitness, swarm_best_position, swarm_fitness_history = swarm.run_pso(
        pso_type
    )

    return (swarm_best_fitness, swarm_best_position, swarm_fitness_history)


def handle_data(fitness_histories):
    plt.figure(figsize=(10, 6))

    for i, fitness_history in enumerate(fitness_histories):
        plt.plot(fitness_history, label=f"PSO Run {i + 1}")

    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.title(f"RPSO {str(options['function_name'])}")

    fitness_values = np.array(
        [fitness_history[-1] for fitness_history in fitness_histories]
    )
    min_fitness = np.min(fitness_values)
    mean_fitness = np.mean(fitness_values)
    max_fitness = np.max(fitness_values)
    std_fitness = np.std(fitness_values)

    statistics_table = (
        "Fitness Statistics:\n"
        "Statistic           | Value\n"
        "--------------------|----------\n"
        "Min                 | {:.6f}\n"
        "Mean                | {:.6f}\n"
        "Max                 | {:.6f}\n"
        "Standard Deviation  | {:.6f}\n"
    ).format(min_fitness, mean_fitness, max_fitness, std_fitness)

    plt.annotate(
        statistics_table,
        xy=(1.02, 0.5),
        xycoords="axes fraction",
        fontsize=10,
        va="center",  # Vertically center the annotation
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f"fitness_histories_plot_{timestamp}.png"
    plt.savefig(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PSO algorithm with single or threaded execution."
    )
    parser.add_argument(
        "mode",
        choices=["single", "threaded", "many_single"],
        help="Select execution mode: single or threaded",
    )
    args = parser.parse_args()

    main = Main(
        iterations,
        num_particles,
        num_dimensions,
        inertia,
        c1,
        c2,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        options,
    )

    if args.mode == "single":
        swarm_best_fitness, swarm_best_position, swarm_fitness_history = main.run_pso(
            PSO_TYPE
        )
        sys.stdout.write("Best fitness: {}\n".format(swarm_best_fitness))

    elif args.mode == "threaded":
        num_cores = multiprocessing.cpu_count()
        run_pso_partial = partial(run_pso_threaded, pso_type=PSO_TYPE)

        with multiprocessing.Pool(processes=num_cores - 1) as pool:
            results = pool.map(run_pso_partial, range(pso_runs))

        fitness_histories = [result[2] for result in results]
        swarm_best_fitness, swarm_best_position, swarm_fitness_history = zip(*results)

        if fitness_histories:
            handle_data(fitness_histories)

        mean_best_fitness = np.mean(swarm_best_fitness)
        min_best_fitness = np.min(swarm_best_fitness)
        max_best_fitness = np.max(swarm_best_fitness)

        sys.stdout.write(
            f"Minimum fitness for {pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}"
        )
    else:
        print(
            "Invalid mode. Please choose either 'single' or 'threaded' as the execution mode."
        )
