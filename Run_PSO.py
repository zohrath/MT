from __future__ import division
import sys
import numpy as np
import multiprocessing
from functools import partial
import argparse

from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from CostFunctions import (
    ann_node_count_fitness,
    get_fingerprinted_data,
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
    plot_particle_positions,
)
from misc import ann_weights_fitness_function, create_model, get_final_model

PSO_TYPE = "gbest"


# Create the particle
num_nodes_per_layer = [
    6,
    6,
    6,
]  # Example: Input layer with 15 nodes, Hidden layer with 3 nodes

# Create and build the model based on the particle configuration
model = create_model(num_nodes_per_layer)
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
        "position_bounds": (-1, 1),
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


# Common options for all PSO runs
iterations = 100
pso_runs = 50
options = pso_functions[9]


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
    plot_averages_fitness_histories(fitness_histories, PSO_TYPE, pso_runs)
    plot_all_fitness_histories(fitness_histories, options, PSO_TYPE, pso_runs)
    plot_particle_positions(swarm_position_histories, 0, 0, 1, 2)


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
        X_train, X_test, y_train, y_test = get_fingerprinted_data()
        sys.stdout.write("Best fitness: {}\n".format(swarm_best_fitness))
        finalModel = get_final_model(model, swarm_best_position)
        predictions = model.predict(X_test)
        print(explained_variance_score(y_test, predictions))

        some_position = [[75, 87, 80, 6920, 17112, 17286]]  # this should produce (1, 0)
        some_position_2 = [[72, 78, 81, 8503, 8420, 8924]]  # this should produce (8,6)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        transformed_some_position = scaler.transform(some_position_2)

        x_value = model.predict(transformed_some_position)
        print(x_value)

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
        # print(swarm_best_position)
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
