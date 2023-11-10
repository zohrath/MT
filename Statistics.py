import json
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import re
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def save_test_func_rpso_stats(
    fitness_histories,
    pso_type,
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
    function_name,
):
    # Find the maximum length of the inner lists
    max_length = max(len(seq) for seq in fitness_histories)

    # Pad shorter lists with NaN values to make them the same length
    padded_fitness_histories = [
        seq + [np.nan] * (max_length - len(seq)) for seq in fitness_histories
    ]

    # Convert the list of lists to a NumPy array
    array_fitness_histories = np.array(padded_fitness_histories)

    # Calculate the mean along axis=0, ignoring NaN values
    averages = np.nanmean(array_fitness_histories, axis=0)

    sub_folder = f"test_func_stats"
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save results in a JSON file
    json_data = {
        "pso_type": pso_type,
        "function_name": function_name,
        "pso_runs": pso_runs,
        "min_best_fitness": min_best_fitness,
        "mean_best_fitness": mean_best_fitness,
        "max_best_fitness": max_best_fitness,
        "averages": averages.tolist(),
        "position_bounds": position_bounds,
        "velocity_bounds": velocity_bounds,
        "fitness_threshold": fitness_threshold,
        "num_particles": num_particles,
        "Cp_min": Cp_min,
        "Cp_max": Cp_max,
        "Cg_min": Cg_min,
        "Cg_max": Cg_max,
        "w_min": w_min,
        "w_max": w_max,
        "gwn_std_dev": gwn_std_dev,
        "iterations": iterations,
        "elapsed_time": elapsed_time,
        "best_weights": best_weights,
        "fitness_histories": fitness_histories,
    }
    json_file_name = f"stats_{pso_type}_{function_name}_{timestamp}.json"
    json_file_path = os.path.join(sub_folder, json_file_name)

    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def save_test_func_gbest_stats(
    fitness_histories,
    pso_type,
    pso_runs,
    position_bounds,
    velocity_bounds,
    fitness_threshold,
    num_particles,
    c1,
    c2,
    inertia,
    iterations,
    elapsed_time,
    min_best_fitness,
    mean_best_fitness,
    max_best_fitness,
    best_weights,
    function_name,
):
    # Find the maximum length of the inner lists
    max_length = max(len(seq) for seq in fitness_histories)

    # Pad shorter lists with NaN values to make them the same length
    padded_fitness_histories = [
        seq + [np.nan] * (max_length - len(seq)) for seq in fitness_histories
    ]

    # Convert the list of lists to a NumPy array
    array_fitness_histories = np.array(padded_fitness_histories)

    # Calculate the mean along axis=0, ignoring NaN values
    averages = np.nanmean(array_fitness_histories, axis=0)

    sub_folder = f"test_func_stats"
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save results in a JSON file
    json_data = {
        "pso_type": pso_type,
        "function_name": function_name,
        "pso_runs": pso_runs,
        "min_best_fitness": min_best_fitness,
        "mean_best_fitness": mean_best_fitness,
        "max_best_fitness": max_best_fitness,
        "averages": averages.tolist(),
        "position_bounds": position_bounds,
        "velocity_bounds": velocity_bounds,
        "fitness_threshold": fitness_threshold,
        "num_particles": num_particles,
        "c1": c1,
        "c2": c2,
        "w": inertia,
        "iterations": iterations,
        "elapsed_time": elapsed_time,
        "best_weights": best_weights,
        "fitness_histories": fitness_histories,
    }
    json_file_name = f"stats_{pso_type}_{function_name}_{timestamp}.json"
    json_file_path = os.path.join(sub_folder, json_file_name)

    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def save_opt_ann_rpso_stats(
    fitness_histories,
    pso_type,
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
):
    averages = np.mean(fitness_histories, axis=0)

    sub_folder = f"synthetic_noise_{pso_type}_stats"
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save results in a JSON file
    json_data = {
        "pso_type": pso_type,
        "pso_runs": pso_runs,
        "min_best_fitness": min_best_fitness,
        "mean_best_fitness": mean_best_fitness,
        "max_best_fitness": max_best_fitness,
        "averages": averages.tolist(),
        "position_bounds": position_bounds,
        "velocity_bounds": velocity_bounds,
        "fitness_threshold": fitness_threshold,
        "num_particles": num_particles,
        "Cp_min": Cp_min,
        "Cp_max": Cp_max,
        "Cg_min": Cg_min,
        "Cg_max": Cg_max,
        "w_min": w_min,
        "w_max": w_max,
        "gwn_std_dev": gwn_std_dev,
        "iterations": iterations,
        "elapsed_time": elapsed_time,
        "best_weights": best_weights,
        "fitness_histories": fitness_histories,
    }
    json_file_name = f"stats_{timestamp}.json"
    json_file_path = os.path.join(sub_folder, json_file_name)

    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def save_opt_ann_gbest_stats(
    fitness_histories,
    pso_type,
    pso_runs,
    position_bounds,
    velocity_bounds,
    fitness_threshold,
    num_particles,
    c1,
    c2,
    inertia,
    iterations,
    elapsed_time,
    min_best_fitness,
    mean_best_fitness,
    max_best_fitness,
    best_weights,
):
    # Find the maximum length of the inner lists
    max_length = max(len(seq) for seq in fitness_histories)

    # Pad shorter lists with NaN values to make them the same length
    padded_fitness_histories = [
        seq + [np.nan] * (max_length - len(seq)) for seq in fitness_histories
    ]

    # Convert the list of lists to a NumPy array
    array_fitness_histories = np.array(padded_fitness_histories)

    # Calculate the mean along axis=0, ignoring NaN values
    averages = np.nanmean(array_fitness_histories, axis=0)

    sub_folder = f"opt_ann_{pso_type}_stats"
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save results in a JSON file
    json_data = {
        "pso_type": pso_type,
        "pso_runs": pso_runs,
        "min_best_fitness": min_best_fitness,
        "mean_best_fitness": mean_best_fitness,
        "max_best_fitness": max_best_fitness,
        "averages": averages.tolist(),
        "position_bounds": position_bounds,
        "velocity_bounds": velocity_bounds,
        "fitness_threshold": fitness_threshold,
        "num_particles": num_particles,
        "c1": c1,
        "c2": c2,
        "w": inertia,
        "iterations": iterations,
        "elapsed_time": elapsed_time,
        "best_weights": best_weights,
        "fitness_histories": fitness_histories,
    }
    json_file_name = f"stats_{timestamp}.json"
    json_file_path = os.path.join(sub_folder, json_file_name)

    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def create_pso_run_stats(
    swarm_position_history,
    fitness_histories,
    options,
    pso_type,
    pso_runs,
    best_swarm_position,
    *pso_params_used,
):
    sub_folder = f"opt_sgd_params_with_gbest_stats"

    # Create a new structure for the desired output
    output_data = []

    # Define a run number counter
    run_number = 0

    for run in swarm_position_history:
        run_data = []
        for iteration in run:
            particle_data = {}
            for i, particle in enumerate(iteration):
                particle_key = f"p{i}"
                particle_data[particle_key] = particle.tolist()
            run_data.append(particle_data)
        output_data.append({"run" + str(run_number): run_data})
        run_number += 1

    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Get the final fitness values from each PSO run
    fitness_values = np.array(
        [fitness_history[-1] for fitness_history in fitness_histories]
    )
    min_fitness = np.min(fitness_values)
    mean_fitness = np.mean(fitness_values)
    max_fitness = np.max(fitness_values)
    std_dev_fitness = np.std(fitness_values)

    (
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
        best_swarm_weights,
    ) = pso_params_used
    best_params = []
    for param in best_swarm_position:
        best_params.append(param)
    json_data = {
        "pso_type": pso_type,
        "pso_runs": pso_runs,
        "best_swarm_position": best_params,
        "statistics": {
            "min_fitness": min_fitness,
            "mean_fitness": mean_fitness,
            "max_fitness": max_fitness,
            "std_dev_fitness": std_dev_fitness,
        },
        "function_name": options["function_name"],
        "iterations": iterations,
        "num_particles": num_particles,
        "num_dimensions": num_dimensions,
        "best_swarm_weights": best_swarm_weights,
        "position_bounds": position_bounds,
        "velocity_bounds": velocity_bounds,
        "inertia": inertia,
        "c1": c1,
        "c2": c2,
        "threshold": threshold,
        "fitness_histories": fitness_histories,
        "position_histories": output_data,
        "elapsed_time": elapsed_time,
    }
    json_file_name = f"optimize_ann_optimizer_params_{timestamp}.json"
    json_file_path = os.path.join(sub_folder, json_file_name)

    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def create_pso_run_stats_rpso(
    swarm_position_history,
    fitness_histories,
    options,
    pso_type,
    pso_runs,
    best_swarm_position,
    *pso_params_used,
):
    sub_folder = f"opt_sgd_params_with_rpso_stats"

    # Create a new structure for the desired output
    output_data = []

    # Define a run number counter
    run_number = 0

    for run in swarm_position_history:
        run_data = []
        for iteration in run:
            particle_data = {}
            for i, particle in enumerate(iteration):
                particle_key = f"p{i}"
                particle_data[particle_key] = particle.tolist()
            run_data.append(particle_data)
        output_data.append({"run" + str(run_number): run_data})
        run_number += 1

    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Get the final fitness values from each PSO run
    fitness_values = np.array(
        [fitness_history[-1] for fitness_history in fitness_histories]
    )
    min_fitness = np.min(fitness_values)
    mean_fitness = np.mean(fitness_values)
    max_fitness = np.max(fitness_values)
    std_dev_fitness = np.std(fitness_values)

    (
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
        gwn_std_dev,
        best_swarm_weights,
    ) = pso_params_used
    best_params = []
    for param in best_swarm_position:
        best_params.append(param)
    json_data = {
        "pso_type": pso_type,
        "pso_runs": pso_runs,
        "best_swarm_position": best_params,
        "statistics": {
            "min_fitness": min_fitness,
            "mean_fitness": mean_fitness,
            "max_fitness": max_fitness,
            "std_dev_fitness": std_dev_fitness,
        },
        "function_name": options["function_name"],
        "iterations": iterations,
        "num_particles": num_particles,
        "num_dimensions": num_dimensions,
        "best_swarm_weights": best_swarm_weights,
        "position_bounds": position_bounds,
        "velocity_bounds": velocity_bounds,
        "Cp_min": Cp_min,
        "Cp_max": Cp_max,
        "Cg_min": Cg_min,
        "Cg_max": Cg_max,
        "w_min": w_min,
        "w_max": w_max,
        "gwn_std_dev": gwn_std_dev,
        "threshold": threshold,
        "fitness_histories": fitness_histories,
        "position_histories": output_data,
        "elapsed_time": elapsed_time,
    }
    json_file_name = f"optimize_ann_optimizer_params_{timestamp}.json"
    json_file_path = os.path.join(sub_folder, json_file_name)

    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def plot_pso_averages_on_test_functions(
    file_path_gbest, file_path_rpso, start_iteration=100
):
    # Load the JSON data from the file
    with open(file_path_gbest, "r") as json_file:
        data_gbest = json.load(json_file)
    with open(file_path_rpso, "r") as json_file:
        data_rpso = json.load(json_file)

    # Extract relevant data
    gbest_averages = data_gbest.get("averages", [])
    rpso_averages = data_rpso.get("averages", [])
    function_name = data_gbest.get("function_name")

    if not gbest_averages:
        print("No data for averages found in the GBest JSON file.")
        return

    if not rpso_averages:
        print("No data for averages found in the RPSO JSON file.")
        return

    # Slice the averages array to start from the specified iteration
    sliced_averages_gbest = gbest_averages[start_iteration - 1 :]
    sliced_averages_rpso = rpso_averages[start_iteration - 1 :]

    # Create a figure for the averages plot
    plt.figure(figsize=(10, 6))

    # Plot GBest data with a blue line and label
    (gbest_line,) = plt.plot(
        sliced_averages_gbest, label="GBest Averages", color="blue"
    )

    # Plot RPSO data with a red line and label
    (rpso_line,) = plt.plot(sliced_averages_rpso, label="RPSO Averages", color="red")

    plt.title("Averages Plot for GBest and RPSO Sphere")
    plt.xlabel("Iteration")
    plt.ylabel("Average Value")
    plt.grid()

    # Extract the fitness values from data_gbest and data_rpso
    gbest_min_fitness = data_gbest.get("min_best_fitness")
    gbest_mean_fitness = data_gbest.get("mean_best_fitness")
    gbest_max_fitness = data_gbest.get("max_best_fitness")

    rpso_min_fitness = data_rpso.get("min_best_fitness")
    rpso_mean_fitness = data_rpso.get("mean_best_fitness")
    rpso_max_fitness = data_rpso.get("max_best_fitness")

    # Add fitness values to the legend
    legend_labels = [
        f"GBest \nMin: {gbest_min_fitness:.5f}\nMean: {gbest_mean_fitness:.5f}\nMax: {gbest_max_fitness:.5f}",
        f"RPSO \nMin: {rpso_min_fitness:.5f}\nMean: {rpso_mean_fitness:.5f}\nMax: {rpso_max_fitness:.5f}",
    ]

    # Create a custom legend
    plt.legend(handles=[gbest_line, rpso_line], labels=legend_labels)

    # Save the plot with a fixed filename
    json_folder = os.path.dirname(file_path_gbest)
    plot_filename = f"stats_{function_name}.png"
    plot_filepath = os.path.join(json_folder, plot_filename)

    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_rpso_averages(file_path, start_iteration=0):
    # Load the JSON data from the file
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    # Extract relevant data
    rpso_type = data["pso_type"]
    averages = data.get("averages", [])

    if not averages:
        print("No data for averages found in the JSON file.")
        return

    # Slice the averages array to start from the specified iteration
    sliced_averages = averages[start_iteration - 1 :]

    # Create a figure for the averages plot
    plt.figure(figsize=(10, 6))
    plt.plot(sliced_averages, label="Averages", color="blue")

    plt.title(f"Averages Plot for {rpso_type}")
    plt.xlabel("Iteration")
    plt.ylabel("Average Value")
    plt.grid()
    plt.legend()

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = (
        os.path.splitext(os.path.basename(file_path))[0] + "_rpso_averages_plot.png"
    )
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_rpso_fitness_histories(file_path):
    # Load the JSON data from the file
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    # Extract relevant data
    rpso_type = data["pso_type"]
    fitness_histories = data["fitness_histories"]

    # Create a figure for the combined plot
    plt.figure(figsize=(10, 6))

    # Plot each fitness history with a different color
    for i, history in enumerate(fitness_histories):
        plt.plot(history, label=f"History {i + 1}")

    plt.title(f"All Fitness Histories for {rpso_type}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Value")
    plt.grid()

    # Hardcoded legend labels
    legend_labels = [
        f'Cp_min = {data["Cp_min"]}',
        f'Cp_max = {data["Cp_max"]}',
        f'Cg_min = {data["Cg_min"]}',
        f'Cg_max = {data["Cg_max"]}',
        f'w_min = {data["w_min"]}',
        f'w_max = {data["w_max"]}',
        f'gwn_std_dev = {data["gwn_std_dev"]}',
    ]
    plt.legend(legend_labels, loc="upper right", bbox_to_anchor=(1.0, 0.85))

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = (
        os.path.splitext(os.path.basename(file_path))[0]
        + "_rpso_fitness_histories_plot.png"
    )
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches="tight")

    # Show the plot
    plt.show()


def rpso_box_plot(file_path):
    # Load the JSON data from the file
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    # Extract relevant data
    rpso_type = data["pso_type"]
    fitness_histories = data["fitness_histories"]

    # Extract the final fitness values from each history
    final_fitness_values = [history[-1] for history in fitness_histories]

    # Find the best and worst fitness values among all histories
    best_fitness = min(final_fitness_values)
    worst_fitness = max(final_fitness_values)

    # Calculate mean, standard deviation, and other statistics
    mean_fitness = np.mean(final_fitness_values)
    std_deviation = np.std(final_fitness_values)
    median_fitness = np.median(final_fitness_values)

    # Extract additional data for the legend
    gwn_std_dev = data.get("gwn_std_dev", "")

    # Create a horizontal box plot with increased width
    plt.figure(figsize=(10, 6))  # Adjust the figsize for a wider image
    boxplot = plt.boxplot(final_fitness_values, vert=False)
    plt.title(f"Final Fitness Histories for {rpso_type}")
    plt.xlabel("Final Fitness Value")

    # Adjust the horizontal positions for the "Best" and "Worst" labels
    offset = 0.1  # Vertical offset for text labels
    best_x = best_fitness - 0.5  # Slightly to the right
    worst_x = worst_fitness + 0.5  # Slightly to the left
    plt.text(
        best_x,
        1 + offset,
        f"Best: {best_fitness:.3f}",
        horizontalalignment="left",
        verticalalignment="center",
    )
    plt.text(
        worst_x,
        1 + offset,
        f"Worst: {worst_fitness:.3f}",
        horizontalalignment="right",
        verticalalignment="center",
    )

    # Set the x-axis range to 1 to 5 with 0.5 increments
    plt.xticks(np.arange(1, 5.5, 0.5))

    # Remove y-axis tick labels
    plt.yticks([])

    # Add a legend on the right-hand side with custom labels
    legend_labels = [
        f'Cp_min = {data["Cp_min"]}',
        f'Cp_max = {data["Cp_max"]}',
        f'Cg_min = {data["Cg_min"]}',
        f'Cg_max = {data["Cg_max"]}',
        f'w_min = {data["w_min"]}',
        f'w_max = {data["w_max"]}',
        f"gwn_std_dev = {gwn_std_dev}",
    ]
    plt.legend(legend_labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = (
        os.path.splitext(os.path.basename(file_path))[0] + "_rpso_box_plot.png"
    )
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches="tight")

    # Show the plot
    plt.show()

    # Display the statistics
    print(f"Best Final Fitness: {best_fitness:.3f}")
    print(f"Worst Final Fitness: {worst_fitness:.3f}")
    print(f"Mean Final Fitness: {mean_fitness:.3f}")
    print(f"Standard Deviation: {std_deviation:.3f}")
    print(f"Median Final Fitness: {median_fitness:.3f}")


def plot_averages(file_path):
    # Load the JSON data from the file
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    # Extract relevant data
    pso_type = data["pso_type"]
    averages = data.get("averages", [])

    if not averages:
        print("No data for averages found in the JSON file.")
        return

    # Create a figure for the averages plot
    plt.figure(figsize=(10, 6))
    plt.plot(averages, label="Averages", color="blue")

    plt.title(f"Averages Plot for {pso_type}")
    plt.xlabel("Iteration")
    plt.ylabel("Average Value")
    plt.grid()
    plt.legend()

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = (
        os.path.splitext(os.path.basename(file_path))[0] + "_averages_plot.png"
    )
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_fitness_histories(file_path):
    # Load the JSON data from the file
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    # Extract relevant data
    pso_type = data["pso_type"]
    fitness_histories = data["fitness_histories"]

    # Create a figure for the combined plot
    plt.figure(figsize=(10, 6))

    # Plot each fitness history with a different color
    for i, history in enumerate(fitness_histories):
        plt.plot(history, label=f"History {i + 1}")

    plt.title(f"All Fitness Histories for {pso_type}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Value (meters)")
    plt.grid()

    # Hardcoded legend labels
    legend_labels = [
        f'c1 = {data["c1"]}',
        f'c2 = {data["c2"]}',
        f'w = {data["inertia"]}',
        f'Particles = {data.get("num_particles", "")}',
        f'Iterations = {data.get("iterations", "")}',
    ]
    plt.legend(legend_labels, loc="upper right", bbox_to_anchor=(1.0, 0.85))

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = (
        os.path.splitext(os.path.basename(file_path))[0] + "_fitness_histories_plot.png"
    )
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches="tight")

    # Show the plot
    plt.show()


def gbest_box_plot():
    file_path = (
        "opt_ann_gbest_stats/calm_data_set/c120w08/stats_2023-09-24_16-23-32.json"
    )
    # Load the JSON data from the file
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    # Extract relevant data
    pso_type = data["pso_type"]
    fitness_histories = data["fitness_histories"]

    # Extract the final fitness values from each history
    final_fitness_values = [history[-1] for history in fitness_histories]

    # Find the best and worst fitness values among all histories
    best_fitness = min(final_fitness_values)
    worst_fitness = max(final_fitness_values)

    # Calculate mean, standard deviation, and other statistics
    mean_fitness = np.mean(final_fitness_values)
    std_deviation = np.std(final_fitness_values)
    median_fitness = np.median(final_fitness_values)

    # Extract additional data for the legend
    num_particles = data.get("num_particles", "")
    iterations = data.get("iterations", "")

    # Create a horizontal box plot of the final fitness values with the x-axis range set to 1 to 5 with 0.5 increments
    plt.figure(figsize=(8, 4))  # Adjust the figsize for a shorter y-axis
    plt.boxplot(final_fitness_values, vert=False)
    plt.title(f"Fitness values for GBest training on noisy fingerprinted data set")
    plt.xlabel("Fitness Value (meters)")

    # Adjust the horizontal positions for the "Best" and "Worst" labels
    offset = 0.1  # Vertical offset for text labels
    best_x = best_fitness + 0.03  # Slightly to the right
    worst_x = worst_fitness + 1.3  # Slightly to the left
    plt.text(
        best_x,
        1 + offset,
        f"Best: {best_fitness:.3f}",
        horizontalalignment="left",
        verticalalignment="center",
    )
    plt.text(
        worst_x,
        1 + offset,
        f"Worst: {worst_fitness:.3f}",
        horizontalalignment="right",
        verticalalignment="center",
    )

    # Set the x-axis range to 1 to 5 with 0.5 increments
    plt.xticks(np.arange(1, 5.5, 0.5))

    # Remove y-axis tick labels
    plt.yticks([])

    # Add a legend on the right-hand side with custom labels
    legend_labels = [
        f'c1 = {data["c1"]}',
        f'c2 = {data["c2"]}',
        f'w = {data["w"]}',
        f"Particles = {num_particles}",
        f"Iterations = {iterations}",
    ]
    plt.legend(legend_labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = os.path.splitext(os.path.basename(file_path))[0] + "_box_plot.png"
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches="tight")

    # Show the plot
    plt.show()

    # Display the statistics
    print(f"Best Final Fitness: {best_fitness:.3f}")
    print(f"Worst Final Fitness: {worst_fitness:.3f}")
    print(f"Mean Final Fitness: {mean_fitness:.3f}")
    print(f"Standard Deviation: {std_deviation:.3f}")
    print(f"Median Final Fitness: {median_fitness:.3f}")


def gbest_multiple_box_plot():
    json_file_path1 = "opt_sgd_params_with_gbest_stats/noisy_data_set/optimize_ann_optimizer_params_2023-10-13_23-41-29.json"
    with open(json_file_path1, "r") as file1:
        data_set1 = json.load(file1)

    json_file_path2 = "opt_sgd_params_with_gbest_stats/noisy_data_set/optimize_ann_optimizer_params_2023-10-14_01-14-12.json"
    with open(json_file_path2, "r") as file2:
        data_set2 = json.load(file2)

    json_file_path3 = "opt_sgd_params_with_gbest_stats/noisy_data_set/optimize_ann_optimizer_params_2023-10-14_02-55-20.json"
    with open(json_file_path3, "r") as file3:
        data_set3 = json.load(file3)

    # Extracting c1, c2, and w for each dataset
    c1_1, c2_1, w_1 = data_set1["c1"], data_set1["c2"], data_set1["inertia"]
    c1_2, c2_2, w_2 = data_set2["c1"], data_set2["c2"], data_set2["inertia"]
    c1_3, c2_3, w_3 = data_set3["c1"], data_set3["c2"], data_set3["inertia"]

    # Process data for Data Set 1
    fitness_histories_1 = data_set1["fitness_histories"]
    final_fitness_values_1 = [history[-1] for history in fitness_histories_1]

    # Process data for Data Set 2
    fitness_histories_2 = data_set2["fitness_histories"]
    final_fitness_values_2 = [history[-1] for history in fitness_histories_2]

    # Process data for Data Set 3
    fitness_histories_3 = data_set3["fitness_histories"]
    final_fitness_values_3 = [history[-1] for history in fitness_histories_3]

    # Create a smaller image with horizontal box plots
    plt.figure(figsize=(8, 5))  # Smaller figure size

    # Box plots with no different colors
    plt.boxplot(
        [final_fitness_values_1, final_fitness_values_2, final_fitness_values_3],
        vert=False,
        patch_artist=True,
    )

    # Set y-axis labels to "Variant 1", "Variant 2", and "Variant 3"
    plt.yticks([1, 2, 3], ["Variant 1", "Variant 2", "Variant 3"])

    plt.title(
        "Fitness values for GBest variants optimizing SGD parameters on noisy fingerprinted data set"
    )
    plt.xlabel("Fitness Value (meters)")

    plt.figtext(
        0.1,
        0.03,
        (
            f"Variant 1: c1: {c1_1}, c2: {c2_1}, w: {w_1}\n"
            f"Variant 2: c1: {c1_2}, c2: {c2_2}, w: {w_2}\n"
            f"Variant 3: c1: {c1_3}, c2: {c2_3}, w: {w_3}\n"
        ),
        fontsize=10,
        ha="left",
        va="bottom",
    )

    # Set padding at the bottom of the plot
    plt.subplots_adjust(bottom=0.3)
    image_path = os.path.join(
        os.path.dirname("opt_sgd_params_with_gbest_stats/noisy_data_set/"),
        f"Combined_Box_Plots.png",
    )
    plt.savefig(image_path, bbox_inches="tight")
    # Display the plot
    plt.show()


def rpso_multiple_box_plot():
    json_file_path1 = "opt_ann_rpso_stats/noisy_data_set/default_rpso_params/stats_2023-09-29_16-04-52.json"
    with open(json_file_path1, "r") as file1:
        data_set1 = json.load(file1)

    json_file_path2 = "opt_ann_rpso_stats/noisy_data_set/grid_search_params/stats_2023-09-30_03-22-36.json"
    with open(json_file_path2, "r") as file2:
        data_set2 = json.load(file2)

    json_file_path3 = "opt_ann_rpso_stats/noisy_data_set/gwn_random_search/stats_2023-09-29_15-49-39.json"
    with open(json_file_path3, "r") as file3:
        data_set3 = json.load(file3)

    json_file_path4 = (
        "opt_ann_rpso_stats/noisy_data_set/no_gwn_val/stats_2023-09-29_16-34-34.json"
    )
    with open(json_file_path4, "r") as file4:
        data_set4 = json.load(file4)

    Cp_min_1, Cp_max_1, Cg_min_1, Cg_max_1, w_min_1, w_max_1, gwn_std_dev_1 = (
        data_set1["Cp_min"],
        data_set1["Cp_max"],
        data_set1["Cg_min"],
        data_set1["Cg_max"],
        data_set1["w_min"],
        data_set1["w_max"],
        data_set1["gwn_std_dev"],
    )

    Cp_min_2, Cp_max_2, Cg_min_2, Cg_max_2, w_min_2, w_max_2, gwn_std_dev_2 = (
        data_set2["Cp_min"],
        data_set2["Cp_max"],
        data_set2["Cg_min"],
        data_set2["Cg_max"],
        data_set2["w_min"],
        data_set2["w_max"],
        data_set2["gwn_std_dev"],
    )

    Cp_min_3, Cp_max_3, Cg_min_3, Cg_max_3, w_min_3, w_max_3, gwn_std_dev_3 = (
        data_set3["Cp_min"],
        data_set3["Cp_max"],
        data_set3["Cg_min"],
        data_set3["Cg_max"],
        data_set3["w_min"],
        data_set3["w_max"],
        data_set3["gwn_std_dev"],
    )

    Cp_min_4, Cp_max_4, Cg_min_4, Cg_max_4, w_min_4, w_max_4, gwn_std_dev_4 = (
        data_set4["Cp_min"],
        data_set4["Cp_max"],
        data_set4["Cg_min"],
        data_set4["Cg_max"],
        data_set4["w_min"],
        data_set4["w_max"],
        data_set4["gwn_std_dev"],
    )

    # Process data for Data Set 1
    fitness_histories_1 = data_set1["fitness_histories"]
    final_fitness_values_1 = [history[-1] for history in fitness_histories_1]

    # Process data for Data Set 2
    fitness_histories_2 = data_set2["fitness_histories"]
    final_fitness_values_2 = [history[-1] for history in fitness_histories_2]

    # Process data for Data Set 3
    fitness_histories_3 = data_set3["fitness_histories"]
    final_fitness_values_3 = [history[-1] for history in fitness_histories_3]

    # Process data for Data Set 4
    fitness_histories_4 = data_set4["fitness_histories"]
    final_fitness_values_4 = [history[-1] for history in fitness_histories_4]

    # Create a smaller image with horizontal box plots
    plt.figure(figsize=(8, 5))  # Smaller figure size

    # Box plots with no different colors
    plt.boxplot(
        [
            final_fitness_values_1,
            final_fitness_values_2,
            final_fitness_values_3,
            final_fitness_values_4,
        ],
        vert=False,
        patch_artist=True,
    )

    # Set y-axis labels to "Variant 1", "Variant 2", and "Variant 3"
    plt.yticks([1, 2, 3, 4], ["Variant 1", "Variant 2", "Variant 3", "Variant 4"])

    plt.title(
        "Fitness values for RPSO variants training ANN on noisy fingerprinted data set"
    )
    plt.xlabel("Fitness Value (meters)")

    plt.figtext(
        0.1,
        0.03,
        (
            f"Variant 1: Cp_min: {Cp_min_1}, Cp_max: {Cp_max_1}, Cg_min: {Cg_min_1}, Cg_max: {Cg_max_1}, w_min: {w_min_1}, w_max: {w_max_1}, gwn_std_dev: {gwn_std_dev_1}\n"
            f"Variant 2: Cp_min: {Cp_min_2}, Cp_max: {Cp_max_2}, Cg_min: {Cg_min_2}, Cg_max: {Cg_max_2}, w_min: {w_min_2}, w_max: {w_max_2}, gwn_std_dev: {gwn_std_dev_2}\n"
            f"Variant 3: Cp_min: {Cp_min_3}, Cp_max: {Cp_max_3}, Cg_min: {Cg_min_3}, Cg_max: {Cg_max_3}, w_min: {w_min_3}, w_max: {w_max_3}, gwn_std_dev: {gwn_std_dev_3}\n"
            f"Variant 4: Cp_min: {Cp_min_4}, Cp_max: {Cp_max_4}, Cg_min: {Cg_min_4}, Cg_max: {Cg_max_4}, w_min: {w_min_4}, w_max: {w_max_4}, gwn_std_dev: {gwn_std_dev_4}\n"
        ),
        fontsize=10,
        ha="left",
        va="bottom",
    )

    # Set padding at the bottom of the plot
    plt.subplots_adjust(bottom=0.3)
    image_path = os.path.join(
        os.path.dirname("opt_ann_rpso_stats/noisy_data_set/"),
        f"Combined_Box_Plots.png",
    )
    plt.savefig(image_path, bbox_inches="tight")
    # Display the plot
    plt.show()


def display_random_uniform_distribution_search_results():
    # Specify the path to the sub-folder containing the JSON files
    subfolder_path = "opt_ann_gbest_uniform_distribution_search/calm_data_set/100_runs"

    # Initialize an empty list to store the fitness values for each run
    fitness_values = []

    # Iterate through the JSON files in the sub-folder
    for filename in os.listdir(subfolder_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(subfolder_path, filename)
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                fitness_histories = data["fitness_histories"]

                # Extract the final values from each sub-array
                final_values = [subarray[-1] for subarray in fitness_histories]

                # Append the final values for this run
                fitness_values.append(final_values)

    # Generate 'runs' based on the number of JSON files processed
    runs = [f"Run {i+1}" for i in range(len(fitness_values))]

    # Create a single vertical box plot for all runs
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot a box plot for all runs
    ax.boxplot(
        fitness_values, vert=True, widths=0.7, showfliers=False, patch_artist=True
    )

    # Set labels and title
    ax.set_xticklabels(runs, rotation=45)
    ax.set_xlabel("Runs")
    ax.set_ylabel("Fitness Values")
    ax.set_title("Box Plot of Fitness Values for All Runs")

    # Remove borders around the box plots
    for box in ax.artists:
        box.set_edgecolor("black")

    # Show the plot
    plt.tight_layout()
    plt.show()


def create_verification_result_plot():
    file_path = "./verification_stats/manually_created_mean_value_summary/verification_stats.json"

    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    verification_stats = data.get("verification_stats", [])
    sorted_stats = sorted(
        verification_stats, key=lambda x: x.get("Median error mean", float("inf"))
    )
    sorted_stats = sorted_stats[:10]
    if not sorted_stats:
        print("No data for verification stats found in the JSON file.")
        return

    # Define a list of colors to use for the bars
    colors = ["b", "g", "r", "c", "m"]
    # Words to remove

    titles = [
        sorted_stats[0]["File name"].replace("_", " ").capitalize(),
        sorted_stats[1]["File name"].replace("_", " ").capitalize(),
        sorted_stats[2]["File name"].replace("_", " ").capitalize(),
        sorted_stats[3]["File name"].replace("_", " ").capitalize(),
        sorted_stats[4]["File name"].replace("_", " ").capitalize(),
        sorted_stats[5]["File name"].replace("_", " ").capitalize(),
        sorted_stats[6]["File name"].replace("_", " ").capitalize(),
        sorted_stats[7]["File name"].replace("_", " ").capitalize(),
        sorted_stats[8]["File name"].replace("_", " ").capitalize(),
        sorted_stats[9]["File name"].replace("_", " ").capitalize(),
    ]

    words_to_remove = [
        "500",
        "iterations",
        "stats",
        "during",
        "training",
        "c149445w0729",
        "c20w08",
    ]
    pattern = r"\b(?:" + "|".join(re.escape(word) for word in words_to_remove) + r")\b"
    # Loop through the titles list and apply the transformations
    formatted_titles = []
    for title in titles:
        # Remove specified words
        formatted_title = re.sub(pattern, "", title)
        # Replace multiple spaces with a single space
        formatted_title = " ".join(formatted_title.split())
        formatted_titles.append(formatted_title)

    for i, stats in enumerate(sorted_stats):
        metric_names = [
            "MAE mean",
            "MSE mean",
            "RMSE mean",
            "Median error mean",
            "Min error mean",
            "Max error mean",
        ]
        metric_values = [stats.get(metric_name, 0) for metric_name in metric_names]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.xlabel("Metrics")
        plt.ylabel("Meters")

        keys_to_process = [
            "c1",
            "c2",
            "w",
            "Cp_min",
            "Cp_max",
            "Cg_min",
            "Cg_max",
            "w_min",
            "w_max",
            "gwn_std_dev",
        ]
        stats_dict = {}

        for key in keys_to_process:
            try:
                value = sorted_stats[i][key]
                if isinstance(value, (int, float)):
                    rounded_value = round(value, 2)
                    stats_dict[key] = rounded_value
            except KeyError:
                pass

        plt.title(f"{formatted_titles[i]}\n {stats_dict}")
        plt.ylim(0, 9)

        for bar, value in zip(bars, metric_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2 - 0.15,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Save the plot image in the same folder as the JSON file
        output_folder = os.path.dirname(file_path)
        plot_file_name = f"plot_{i+1}.png"
        plot_file_path = os.path.join(output_folder, plot_file_name)
        plt.savefig(plot_file_path)


def generate_error_heatmap():
    absolute_errors = [
        [3.9676778316497803, 2.03533935546875],
        [1.1805458068847656, 0.2350759506225586],
        [0.17825984954833984, 0.7333431243896484],
        [2.368678092956543, 0.30632781982421875],
        [0.00995635986328125, 0.1730508804321289],
        [0.6688542366027832, 0.005642890930175781],
        [0.0089111328125, 0.9026412963867188],
        [0.9024326801300049, 0.4384307861328125],
        [1.3951104879379272, 1.1938929557800293],
        [1.290392279624939, 1.8698885440826416],
        [2.1015533208847046, 0.3818645477294922],
        [1.5778822898864746, 1.2639590501785278],
        [1.251884937286377, 0.662724494934082],
        [0.7745270729064941, 1.6599738597869873],
        [0.6408061981201172, 2.4364837408065796],
        [0.6427230834960938, 0.6819539070129395],
        [0.5847091674804688, 0.2506542205810547],
        [0.060021400451660156, 0.46468257904052734],
        [1.2250356674194336, 0.9845290184020996],
        [0.6247453689575195, 1.957648754119873],
        [3.8669509887695312, 0.14014476537704468],
        [0.40807628631591797, 0.3310115337371826],
        [3.1493606567382812, 0.003936469554901123],
        [3.72092342376709, 0.9871511459350586],
        [2.5006179809570312, 0.4046769142150879],
        [2.3701276779174805, 0.14478635787963867],
        [1.0284576416015625, 0.4296894073486328],
        [1.8324546813964844, 0.012808799743652344],
        [0.7442073822021484, 1.1269989013671875],
        [0.07163333892822266, 0.039267539978027344],
        [1.1738014221191406, 0.9628429412841797],
        [1.5815649032592773, 0.13577556610107422],
        [0.9039592742919922, 1.210442066192627],
        [0.6139020919799805, 0.7075347900390625],
        [2.641911506652832, 1.1032581329345703],
        [0.9405050277709961, 0.26720428466796875],
        [0.35647010803222656, 0.503333568572998],
        [0.5652360916137695, 0.5725975036621094],
        [0.42625904083251953, 0.16082191467285156],
        [0.2786836624145508, 0.6542654037475586],
        [0.20084190368652344, 0.3694143295288086],
        [0.8952836990356445, 0.22771263122558594],
        [1.953896164894104, 1.1189956665039062],
        [1.8928205966949463, 0.5102806091308594],
        [0.0409390926361084, 1.1562528610229492],
    ]
    df = pd.read_csv("fingerprints-random-points-calm.csv")

    # Calculate the sum of elements in each sub-array and use them as colors
    colors = [sum(sub_array) for sub_array in absolute_errors]

    # Extract X and Y coordinates from the DataFrame
    x = df["X"]
    y = df["Y"]

    # Create a custom colormap from green to red
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["lime", "red"], N=256)

    plt.figure(figsize=(10, 6))

    new_df = pd.read_csv("fingerprints-calm.csv")
    new_x = new_df["X"]
    new_y = new_df["Y"]

    # Filter out the points from the new CSV that exist in the old CSV
    unique_x = []
    unique_y = []
    for i in range(len(new_x)):
        if (new_x[i], new_y[i]) not in zip(x, y):
            unique_x.append(new_x[i])
            unique_y.append(new_y[i])

    # Filter out error dots where the absolute error is less than 1
    filtered_colors = []
    filtered_x = []
    filtered_y = []
    for i in range(len(colors)):
        if colors[i] >= 1:
            filtered_colors.append(colors[i])
            filtered_x.append(x[i])
            filtered_y.append(y[i])

    # Plot the unique points from the new data as bright blue dots
    plt.scatter(
        unique_x,
        unique_y,
        s=20,
        c="black",
        alpha=0.1,
        label="Training data",
        marker="o",
    )

    # Overlay the previous dots based on absolute_errors with the filter applied
    sc = plt.scatter(
        filtered_x,
        filtered_y,
        s=20,
        c=filtered_colors,
        cmap=cmap,
        vmin=min(filtered_colors),
        vmax=max(filtered_colors),
        label="Verification points with > 1m error",
    )

    # plt.title("Error map best model on calm verification set")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    # plt.legend()

    # Add a colorbar for reference
    colorbar = plt.colorbar(sc)
    colorbar.set_label("Absolute errors scale")
    # colorbar.remove()

    # Save the plot with a transparent background
    plt.savefig("dot_plot.png", transparent=True)

    plt.show()


def generate_verification_set_absolute_error_histogram():
    data_errors = [
        [3.5743465423583984, 0.2107219696044922],
        [3.449167490005493, 0.45858192443847656],
        [2.409945011138916, 0.55859375],
        [4.079694747924805, 0.3928394317626953],
        [0.38111114501953125, 0.17768383026123047],
        [1.2815914154052734, 1.471817970275879],
        [1.7282557487487793, 0.5521974563598633],
        [0.7924472093582153, 1.651357650756836],
        [0.08137130737304688, 0.4992713928222656],
        [1.1834864616394043, 1.2972314357757568],
        [0.6844336986541748, 0.02749776840209961],
        [1.0368618965148926, 0.5309829711914062],
        [1.551931381225586, 0.37918615341186523],
        [1.6687297821044922, 2.1513200998306274],
        [1.9856910705566406, 1.195833146572113],
        [1.6521825790405273, 0.005169987678527832],
        [0.5066413879394531, 0.2833542823791504],
        [1.3473749160766602, 0.060303688049316406],
        [2.1676416397094727, 0.059762001037597656],
        [0.8070535659790039, 1.5664339065551758],
        [3.3412656784057617, 0.3856205940246582],
        [0.4689455032348633, 0.5975174903869629],
        [2.9067726135253906, 0.5545867681503296],
        [3.0228700637817383, 0.6832327842712402],
        [2.1122589111328125, 0.012772083282470703],
        [3.0077056884765625, 0.09112691879272461],
        [1.1556453704833984, 0.10997676849365234],
        [1.6269245147705078, 1.5461845397949219],
        [1.9969062805175781, 1.5220909118652344],
        [0.3511810302734375, 0.5826959609985352],
        [1.9786567687988281, 2.0618391036987305],
        [1.522726058959961, 0.3058891296386719],
        [0.178802490234375, 0.716526985168457],
        [1.019521713256836, 0.7005538940429688],
        [2.2980871200561523, 1.3431310653686523],
        [0.4175844192504883, 0.8997149467468262],
        [0.5885772705078125, 0.17059659957885742],
        [0.7370052337646484, 0.45836734771728516],
        [0.3396415710449219, 0.2434673309326172],
        [0.1753406524658203, 1.3635759353637695],
        [0.31200408935546875, 1.347219467163086],
        [1.2463407516479492, 1.2928047180175781],
        [1.570070743560791, 0.7096672058105469],
        [1.5350384712219238, 2.0574073791503906],
        [0.039583444595336914, 0.30926990509033203],
    ]
    # Calculate the combined errors
    combined_errors = np.sort([sum(error) for error in data_errors])
    y_errors = np.sort([error[1] for error in data_errors])
    x_errors = np.sort([error[0] for error in data_errors])

    # Create a line chart to visualize the errors
    plt.figure(figsize=(10, 6))
    plt.plot(combined_errors, label="Combined Errors", linestyle="-")
    plt.plot(x_errors, label="X-coordinate Errors", linestyle="--")
    plt.plot(y_errors, label="Y-coordinate Errors", linestyle="-.")
    plt.xlabel("Coordinate Error Index")
    plt.ylabel("Error (m)")
    plt.title("Error Distribution Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_fitness_histories_from_json(json_file_path):
    # Get the full path to the JSON file, assuming it's in a subfolder named "data"

    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)

        fitness_histories = data.get("fitness_histories", [])

        if not fitness_histories:
            print("No fitness histories found in the JSON file.")
            return

        for history in fitness_histories:
            plt.plot(history, label="Fitness History")

        plt.xlabel("Generation")
        plt.ylabel("Fitness Value")
        plt.title("Fitness History Plot")
        plt.legend()
        plt.show()

    except FileNotFoundError:
        print(f"File '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from '{json_file_path}'.")


def create_latex_table_from_verification_stats(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    verification_stats = data.get("verification_stats", [])

    if not verification_stats:
        return "No verification stats found in the JSON file."

    # Create the LaTeX table header
    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += "MAE & MSE & RMSE & MedAE & Min Error & Max Error \\\\\n"
    latex_table += "\\hline\n"

    # Iterate over the verification_stats and populate the table
    for stat in verification_stats:
        latex_table += f"{stat['MAE']} & {stat['MSE']} & {stat['RMSE']} & {stat['MedAE']} & {stat['Min Error']} & {stat['Max Error']} \\\\\n"

    # Complete the LaTeX table
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"

    print(latex_table)


def generate_random_uniform_distribution_search_data_table():
    # Define the path to the subfolder containing JSON files
    subfolder_path = "opt_ann_gbest_uniform_distribution_search/with_velocity_bounds"

    # Define the output filename
    output_filename = "combined_data_table.png"

    # Initialize an empty list to store data from all JSON files
    all_data = []

    # Get a list of JSON files in the subfolder
    json_files = [f for f in os.listdir(subfolder_path) if f.endswith(".json")]

    for json_file in json_files:
        # Construct the full path to each JSON file
        json_file_path = os.path.join(subfolder_path, json_file)

        # Load data from the JSON file
        with open(json_file_path, "r") as file:
            data = json.load(file)

        # Extract the desired fields from the JSON data
        row_data = {
            "File Name": os.path.splitext(json_file)[0],
            "min_best_fitness": data["min_best_fitness"],
            "mean_best_fitness": data["mean_best_fitness"],
            "max_best_fitness": data["max_best_fitness"],
            "position_bounds[0]": data["position_bounds"][0],
            "velocity_bounds[0]": data["velocity_bounds"][0],
            "c1": data["c1"],
            "c2": data["c2"],
            "w": data["w"],
        }

        all_data.append(row_data)

    # Create a Pandas DataFrame from the combined data
    df = pd.DataFrame(all_data)

    # Convert the DataFrame to a LaTeX table
    latex_table = df.to_latex(index=False, escape=False)

    # Remove the default table environment and center the table
    latex_table = latex_table.replace(
        "\\begin{tabular}", "\\begin{center}\n\\begin{tabular}"
    ).replace("\\end{tabular}", "\\end{tabular}\n\\end{center}")

    # Print or return the LaTeX table
    print(latex_table)


def plot_synthetic_noise_data(json_file):
    # Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract data
    means_data = {}
    for item in data["Noise test stats"]:
        file_name = "Unoptimized Adam" if "Adam" in item["File name"] else "RPSO"
        means_data[file_name] = {
            "MAE mean": item["MAE mean"],
            "MSE mean": item["MSE mean"],
            "RMSE mean": item["RMSE mean"],
            "Median error mean": item["Median error mean"],
            "Min error mean": item["Min error mean"],
            "Max error mean": item["Max error mean"],
        }

    # Plotting
    labels = ["Unoptimized Adam", "RPSO"]
    bar_width = 0.15
    r = np.arange(len(labels))
    colors = ["b", "g", "r", "c", "m", "b"]

    plt.figure(figsize=(10, 6))
    for idx, (mean_type, color) in enumerate(zip(means_data[labels[0]].keys(), colors)):
        plt.bar(
            r + idx * bar_width,
            [means_data[label][mean_type] for label in labels],
            width=bar_width,
            color=color,
            edgecolor="gray",
            label=mean_type,
        )

    plt.xlabel("Method", fontweight="bold")
    plt.xticks(
        [r + bar_width for r in range(len(labels))], labels, rotation=45, ha="right"
    )
    plt.ylabel("Values")
    plt.title("Bar Chart of Mean Fields")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mae_means(folder_path):
    """
    Reads all JSON files in the specified folder and plots the 'MAE mean' for each object,
    using the last word of the 'File name' as the x-tick label, separating bar pairs for different files,
    and giving each pair a unique color.

    Parameters:
    - folder_path: The path to the folder containing JSON files.
    """
    mae_means = []  # List to store the 'MAE mean' values
    file_labels = []  # List to store the last word of the 'File name' for x-ticks
    bar_width = 0.4  # Width of the bars
    gap_width = 0.2  # Gap width between pairs

    for file_name in sorted(
        os.listdir(folder_path)
    ):  # Sort the files for consistent order
        if file_name.endswith(".json"):  # Check if the file is a JSON file
            file_path = os.path.join(folder_path, file_name)  # Get the full file path
            with open(file_path, "r") as json_file:  # Open the file for reading
                data = json.load(json_file)  # Read the JSON file
                for obj in data["Noise test stats"]:
                    mae_means.append(obj["Median error mean"])
                    # Extract the last word from the 'File name'
                    last_word = obj["File name"].rsplit(None, 1)[-1]
                    file_labels.append(last_word)

    # Generate a list of colors from the matplotlib colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(mae_means) // 2))

    # Calculate the positions for each bar
    bar_positions = np.arange(len(mae_means)) * (bar_width + gap_width) + gap_width / 2

    # Now plot the 'MAE mean' values
    plt.figure(figsize=(15, 7))  # Set the figure size

    # Plot bars with unique colors for each pair
    for i in range(0, len(mae_means), 2):
        plt.bar(
            bar_positions[i : i + 2],
            mae_means[i : i + 2],
            width=bar_width,
            color=colors[i // 2],
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel("File Name")
    plt.ylabel("MAE mean")
    plt.title("MAE mean values per file name")
    plt.xticks(
        bar_positions, file_labels, rotation=45
    )  # Set x-ticks to the last word of the 'File name'

    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.show()


def plot_noise_vs_rmse(json_file_path):
    """
    Reads a JSON file containing 'noise level values' and 'RMSE values' and plots them
    with 'noise level values' on the x-axis and 'RMSE values' on the y-axis.

    Parameters:
    - json_file_path: The path to the JSON file.
    """
    with open(json_file_path, "r") as json_file:  # Open the file for reading
        data = json.load(json_file)  # Read the JSON file

    # Extract noise level and RMSE values
    noise_levels = data["noise level values"]
    rmse_values = data["RMSE values"]
    print(noise_levels)
    print(rmse_values)
    plt.figure(figsize=(15, 7))  # Set the figure size
    plt.plot(noise_levels, rmse_values, marker="o", linestyle="-", color="skyblue")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel("Noise Level")
    plt.ylabel("RMSE")
    plt.title(data["name"])
    plt.xticks(noise_levels, [str(noise) for noise in noise_levels], rotation=45)

    # Adding the grid
    plt.grid(True)

    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.show()


# Replace with the name of your JSON file
# json_file = "opt_adam_params_with_gbest_stats/noisy_data_set/500_iterations_during_training/random_search_params/optimize_ann_optimizer_params_2023-10-12_13-57-08.json"
# plot_fitness_histories(json_file)
# gbest_box_plot()
# rpso_multiple_box_plot()
# generate_verification_set_absolute_error_histogram()
# create_verification_result_plot()

# generate_verification_set_absolute_error_histogram()
# generate_error_heatmap()
# display_random_uniform_distribution_search_results()
# generate_random_uniform_distribution_search_data_table()
# plot_pso_averages_on_test_functions(
#     "test_func_stats/stats_gbest_Sphere.json",
#     "test_func_stats/stats_rpso_Sphere.json",
# )
# plot_synthetic_noise_data(
#     "synthetic_noise_test_results/2023-11-05_18-38_noise_stats.json"
# )

# path = "synthetic_noise_test_results/rpso"
# plot_mae_means(path)
plot_noise_vs_rmse("synthetic_noise_test_results/noise_level_check_adam.json")
