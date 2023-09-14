from __future__ import division
import csv
import datetime
import sys
import numpy as np
import multiprocessing
from functools import partial
import argparse
import itertools
import pandas as pd
from CostFunctions import (
    ann_weights_fitness_function,
)

from GBestPSO import GBest_PSO
from RPSO import RPSO
from Statistics import handle_data
from pso_options import create_model, pso_options

PSO_TYPE = "gbest"


class PSO:
    def __init__(self, iterations, options, num_dimensions, model):
        self.iterations = iterations
        self.num_particles = options["num_particles"]
        self.num_dimensions = num_dimensions
        self.position_bounds = options["position_bounds"]
        self.velocity_bounds = options["velocity_bounds"]
        self.threshold = options["threshold"]
        self.function = options["function"]
        self.options = options
        self.model = model

    def run_pso(self, pso_type):
        if pso_type == "gbest":
            swarm = GBest_PSO(
                self.iterations,
                self.num_particles,
                self.num_dimensions,
                self.position_bounds,
                self.velocity_bounds,
                self.options["inertia"],
                self.options["c1"],
                self.options["c2"],
                self.threshold,
                self.function,
            )
            swarm.run_pso(self.model)

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
                self.options["Cp_min"],
                self.options["Cp_max"],
                self.options["Cg_min"],
                self.options["Cg_max"],
                self.options["w_min"],
                self.options["w_max"],
                self.options["threshold"],
                self.options["function"],
                self.options["gwn_std_dev"],
            )
            swarm.run_pso(self.model)

            return (
                swarm.swarm_best_fitness,
                swarm.swarm_best_position,
                swarm.swarm_fitness_history,
                swarm.swarm_position_history,
            )


def run_grid_search(params, pso_type, iterations):
    model, num_dimensions = create_model()
    Cp_min, Cp_max, Cg_min, Cg_max, w_min, w_max, gwn_std_dev = params

    main = PSO(
        iterations,
        {
            "function": ann_weights_fitness_function,
            "function_name": "ANN Node Count RPSO",
            "position_bounds": (-1.0, 1.0),
            "velocity_bounds": (-0.2, 0.2),
            "threshold": 1,
            "num_particles": 10,
            "num_dimensions": num_dimensions,
            "Cp_min": Cp_min,
            "Cp_max": Cp_max,
            "Cg_min": Cg_min,
            "Cg_max": Cg_max,
            "w_min": w_min,
            "w_max": w_max,
            "gwn_std_dev": gwn_std_dev,
        },
        num_dimensions,
        model,
    )
    (
        swarm_best_fitness,
        swarm_best_position,
        swarm_fitness_history,
        swarm_position_history,
    ) = main.run_pso(pso_type)

    return swarm_best_fitness, params


def run_pso(_, pso_type, iterations, options):
    model, dimensions = create_model()
    swarm = PSO(
        iterations,
        options,
        dimensions,
        model,
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
        options = pso_options[10]
        print(options)
        run_pso(0, PSO_TYPE, iterations=75, options=options)

    elif args.mode == "threaded":
        pso_runs = 75
        options = pso_options[9]

        for _ in range(1):
            run_pso_partial = partial(
                run_pso, pso_type=PSO_TYPE, iterations=75, options=options
            )

            with multiprocessing.Pool(
                processes=multiprocessing.cpu_count() - 1
            ) as pool:
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
                    fitness_histories,
                    swarm_position_history,
                    PSO_TYPE,
                    pso_runs,
                    options,
                )

            mean_best_fitness = np.mean(swarm_best_fitness)
            min_best_fitness = np.min(swarm_best_fitness)
            max_best_fitness = np.max(swarm_best_fitness)

            sys.stdout.write(
                f"Minimum fitness for {pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}"
            )
    elif args.mode == "gridSearch":
        param_grid = {
            "Cp_min": [0.1, 0.3, 0.5, 0.7, 0.9],
            "Cp_max": [3.5, 3.0, 2.5, 2.0, 1.5],
            "Cg_min": [0.1, 0.3, 0.5, 0.7, 0.9],
            "Cg_max": [3.5, 3.0, 2.5, 2.0, 1.5],
            "w_min": [0.05, 0.3, 0.8],
            "w_max": [0.5, 0.9, 1.3, 1.7, 2.0],
            "gwn_std_dev": [0.01, 0.07, 0.15, 0.2],
        }

        combinations = list(itertools.product(*param_grid.values()))
        parameter_permutations_to_test_per_loop = 14

        num_processes = multiprocessing.cpu_count() - 1
        sub_lists = [
            combinations[i : i + parameter_permutations_to_test_per_loop]
            for i in range(0, len(combinations), num_processes)
        ]

        #  Filter out already checked combinations in csv file
        df = pd.read_csv("best_parameter_results.csv", delimiter=",")
        df = df.drop(["Best Fitness"], axis=1)
        list_of_tuples = [tuple(x) for x in df.to_records(index=False)]

        def flatten(lst):
            return [item for sublist in lst for item in sublist]

        flat_sublists = flatten(sub_lists)

        # Check for duplicates in list_of_tuples and remove them from flat_sublists
        filtered_sublists = []
        for tup in flat_sublists:
            if tup not in list_of_tuples:
                filtered_sublists.append(tup)

        # Divide the filtered_sublists into sub-lists again
        filtered_sublists_divided = [
            filtered_sublists[i : i + parameter_permutations_to_test_per_loop]
            for i in range(
                0, len(filtered_sublists), parameter_permutations_to_test_per_loop
            )
        ]

        print("Remaining combinations to check: ", len(filtered_sublists_divided))

        run_grid_search_partial = partial(
            run_grid_search,
            pso_type=PSO_TYPE,
            iterations=75,
        )

        for sublist in filtered_sublists_divided:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(run_grid_search_partial, sublist)

            # Find the best result across all simulations
            best_result, best_params = min(results, key=lambda x: x[0])

            # Append best result to an existing CSV file or create a new one if it doesn't exist
            csv_filename = "best_parameter_results.csv"
            with open(csv_filename, mode="a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                if csv_file.tell() == 0:
                    csv_writer.writerow(
                        ["Best Fitness"]
                        + ["Param" + str(i) for i in range(1, len(best_params) + 1)]
                    )
                csv_writer.writerow([best_result] + list(best_params))

            print("Best result appended at ", datetime.datetime.now())

    else:
        print(
            "Invalid mode. Please choose either 'single' or 'threaded' as the execution mode."
        )
