from __future__ import division
import sys
import numpy as np
import multiprocessing
from functools import partial
import argparse
import itertools
from keras.layers import Dense
from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
from CostFunctions import (
    ann_node_count_fitness,
    ann_weights_fitness_function,
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
from Statistics import handle_data


PSO_TYPE = "gbest"


def create_model():
    num_nodes_per_layer = [
        6,
        6,
        6,
    ]
    model = Sequential()
    model.add(Dense(num_nodes_per_layer[0], activation="relu", input_shape=(6,)))

    for nodes in num_nodes_per_layer[1:]:
        model.add(Dense(nodes, activation="relu"))

    model.add(Dense(2))  # Output layer

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return model


# Create and build the model based on the particle configuration
model = create_model()
# # Calculate the total number of values required
total_num_weights = sum(np.prod(w.shape) for w in model.get_weights()[::2])
total_num_biases = sum(np.prod(b.shape) for b in model.get_weights()[1::2])
total_number_of_values = total_num_weights + total_num_biases


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
            swarm.run_pso(model)

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
            swarm.run_pso(model)

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
    {
        "function": ann_weights_fitness_function,
        "function_name": "ANN Weights and biases gbest",
        "position_bounds": (-1.0, 1.0),
        "velocity_bounds": (-0.2, 0.2),
        "threshold": 1,
        "num_particles": 3,
        "num_dimensions": total_number_of_values,
        "inertia": 0.9,
        "c1": 1.49445,
        "c2": 1.49445,
    },
    {
        "function": ann_weights_fitness_function,
        "function_name": "ANN Node Count RPSO",
        "position_bounds": (-1.0, 1.0),
        "velocity_bounds": (-0.2, 0.2),
        "threshold": 1,
        "num_particles": 5,
        "num_dimensions": total_number_of_values,
        "Cp_min": 0.5,
        "Cp_max": 2.5,
        "Cg_min": 0.5,
        "Cg_max": 2.5,
        "w_min": 0.4,
        "w_max": 0.9,
    },
]


def run_grid_search(params_range, pso_type, iterations, total_number_of_values):
    best_result = None
    best_fitness = float("inf")

    for params in params_range:
        inertia, c1, c2 = params
        main = Main(
            iterations,
            {
                "function": ann_weights_fitness_function,
                "function_name": "ANN Weights and biases gbest",
                "position_bounds": (-1.0, 1.0),
                "velocity_bounds": (-0.2, 0.2),
                "threshold": 1,
                "num_particles": 3,
                "num_dimensions": total_number_of_values,
                "inertia": inertia,
                "c1": c1,
                "c2": c2,
            },
        )
        (
            swarm_best_fitness,
            swarm_best_position,
            swarm_fitness_history,
            swarm_position_history,
        ) = main.run_pso(pso_type)

        if swarm_best_fitness < best_fitness:
            best_fitness = swarm_best_fitness
            best_result = (swarm_best_fitness, params)

    return best_result


# Common options for all PSO runs
iterations = 100
options = pso_functions[9]


def run_pso(_, pso_type):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PSO algorithm with single or threaded execution."
    )
    parser.add_argument(
        "mode",
        choices=["single", "threaded", "gridSearch"],
        help="Select execution mode: single or threaded",
    )

    args = parser.parse_args()

    if args.mode == "single":
        run_pso(0, PSO_TYPE)

    elif args.mode == "threaded":
        pso_runs = 6
        num_cores = multiprocessing.cpu_count()
        run_pso_partial = partial(run_pso, pso_type=PSO_TYPE)

        with multiprocessing.Pool(processes=num_cores - 1) as pool:
            results = pool.map(run_pso_partial, range(pso_runs))

        fitness_histories = [result[2] for result in results]
        (
            swarm_best_fitness,
            swarm_best_position,
            swarm_fitness_history,
            swarm_position_history,
        ) = zip(*results)

        if fitness_histories:
            handle_data(
                fitness_histories, swarm_position_history, PSO_TYPE, pso_runs, options
            )

        mean_best_fitness = np.mean(swarm_best_fitness)
        min_best_fitness = np.min(swarm_best_fitness)
        max_best_fitness = np.max(swarm_best_fitness)

        sys.stdout.write(
            f"Minimum fitness for {pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}"
        )
    elif args.mode == "gridSearch":
        param_grid = {
            "inertia": [0.5, 0.7, 0.9],
            "c1": [1.5, 2.0, 2.5],
            "c2": [1.5, 2.0, 2.5],
        }

        num_cores = multiprocessing.cpu_count() - 1
        param_ranges = np.array_split(
            list(itertools.product(*param_grid.values())), num_cores
        )

        run_grid_search_partial = partial(
            run_grid_search,
            pso_type=PSO_TYPE,
            iterations=iterations,
            total_number_of_values=total_number_of_values,
        )

        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(run_grid_search_partial, param_ranges)

        # Find the best result across all simulations
        best_result = min(results, key=lambda x: x[0])
        best_fitness = best_result[0]
        best_params = best_result[1]

        # Print the best parameters
        print("Best Parameters:", best_params)
        print("Best Fitness:", best_fitness)
    else:
        print(
            "Invalid mode. Please choose either 'single' or 'threaded' as the execution mode."
        )
