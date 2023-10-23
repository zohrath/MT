import json
import os
import time
from matplotlib import pyplot as plt
import numpy as np


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
    function_name
):
    # Find the maximum length of the inner lists
    max_length = max(len(seq) for seq in fitness_histories)

    # Pad shorter lists with NaN values to make them the same length
    padded_fitness_histories = [
        seq + [np.nan] * (max_length - len(seq)) for seq in fitness_histories]

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
    function_name
):
    # Find the maximum length of the inner lists
    max_length = max(len(seq) for seq in fitness_histories)

    # Pad shorter lists with NaN values to make them the same length
    padded_fitness_histories = [
        seq + [np.nan] * (max_length - len(seq)) for seq in fitness_histories]

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
        seq + [np.nan] * (max_length - len(seq)) for seq in fitness_histories]

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
        best_swarm_weights
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
        best_swarm_weights
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


def plot_pso_averages_on_test_functions(file_path_gbest, file_path_rpso, start_iteration=100):
    # Load the JSON data from the file
    with open(file_path_gbest, 'r') as json_file:
        data_gbest = json.load(json_file)
    with open(file_path_rpso, 'r') as json_file:
        data_rpso = json.load(json_file)

    # Extract relevant data
    gbest_averages = data_gbest.get('averages', [])
    rpso_averages = data_rpso.get('averages', [])
    function_name = data_gbest.get('function_name')

    if not gbest_averages:
        print("No data for averages found in the GBest JSON file.")
        return

    if not rpso_averages:
        print("No data for averages found in the RPSO JSON file.")
        return

    # Slice the averages array to start from the specified iteration
    sliced_averages_gbest = gbest_averages[start_iteration - 1:]
    sliced_averages_rpso = rpso_averages[start_iteration - 1:]

    # Create a figure for the averages plot
    plt.figure(figsize=(10, 6))

    # Plot GBest data with a blue line and label
    gbest_line, = plt.plot(sliced_averages_gbest,
                           label='GBest Averages', color='blue')

    # Plot RPSO data with a red line and label
    rpso_line, = plt.plot(sliced_averages_rpso,
                          label='RPSO Averages', color='red')

    plt.title('Averages Plot for GBest and RPSO')
    plt.xlabel('Iteration')
    plt.ylabel('Average Value')
    plt.grid()

    # Extract the fitness values from data_gbest and data_rpso
    gbest_min_fitness = data_gbest.get('min_best_fitness')
    gbest_mean_fitness = data_gbest.get('mean_best_fitness')
    gbest_max_fitness = data_gbest.get('max_best_fitness')

    rpso_min_fitness = data_rpso.get('min_best_fitness')
    rpso_mean_fitness = data_rpso.get('mean_best_fitness')
    rpso_max_fitness = data_rpso.get('max_best_fitness')

    # Add fitness values to the legend
    legend_labels = [
        f'GBest \nMin: {gbest_min_fitness:.5f}\nMean: {gbest_mean_fitness:.5f}\nMax: {gbest_max_fitness:.5f}',
        f'RPSO \nMin: {rpso_min_fitness:.5f}\nMean: {rpso_mean_fitness:.5f}\nMax: {rpso_max_fitness:.5f}'
    ]

    # Create a custom legend
    plt.legend(handles=[gbest_line, rpso_line], labels=legend_labels)

    # Save the plot with a fixed filename
    json_folder = os.path.dirname(file_path_gbest)
    plot_filename = f'stats_{function_name}.png'
    plot_filepath = os.path.join(json_folder, plot_filename)

    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_rpso_averages(file_path, start_iteration=0):
    # Load the JSON data from the file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract relevant data
    rpso_type = data['pso_type']
    averages = data.get('averages', [])

    if not averages:
        print("No data for averages found in the JSON file.")
        return

    # Slice the averages array to start from the specified iteration
    sliced_averages = averages[start_iteration - 1:]

    # Create a figure for the averages plot
    plt.figure(figsize=(10, 6))
    plt.plot(sliced_averages, label='Averages', color='blue')

    plt.title(f'Averages Plot for {rpso_type}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Value')
    plt.grid()
    plt.legend()

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = os.path.splitext(os.path.basename(file_path))[
        0] + '_rpso_averages_plot.png'
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_rpso_fitness_histories(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract relevant data
    rpso_type = data['pso_type']
    fitness_histories = data['fitness_histories']

    # Create a figure for the combined plot
    plt.figure(figsize=(10, 6))

    # Plot each fitness history with a different color
    for i, history in enumerate(fitness_histories):
        plt.plot(history, label=f'History {i + 1}')

    plt.title(f'All Fitness Histories for {rpso_type}')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.grid()

    # Hardcoded legend labels
    legend_labels = [f'Cp_min = {data["Cp_min"]}', f'Cp_max = {data["Cp_max"]}', f'Cg_min = {data["Cg_min"]}',
                     f'Cg_max = {data["Cg_max"]}', f'w_min = {data["w_min"]}', f'w_max = {data["w_max"]}',
                     f'gwn_std_dev = {data["gwn_std_dev"]}']
    plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.0, 0.85))

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = os.path.splitext(os.path.basename(file_path))[
        0] + '_rpso_fitness_histories_plot.png'
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches='tight')

    # Show the plot
    plt.show()


def rpso_box_plot(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract relevant data
    rpso_type = data['pso_type']
    fitness_histories = data['fitness_histories']

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
    gwn_std_dev = data.get('gwn_std_dev', '')

    # Create a horizontal box plot with increased width
    plt.figure(figsize=(10, 6))  # Adjust the figsize for a wider image
    boxplot = plt.boxplot(final_fitness_values, vert=False)
    plt.title(f'Final Fitness Histories for {rpso_type}')
    plt.xlabel('Final Fitness Value')

    # Adjust the horizontal positions for the "Best" and "Worst" labels
    offset = 0.1  # Vertical offset for text labels
    best_x = best_fitness - 0.5  # Slightly to the right
    worst_x = worst_fitness + 0.5  # Slightly to the left
    plt.text(best_x, 1 + offset, f'Best: {best_fitness:.3f}',
             horizontalalignment='left', verticalalignment='center')
    plt.text(worst_x, 1 + offset, f'Worst: {worst_fitness:.3f}',
             horizontalalignment='right', verticalalignment='center')

    # Set the x-axis range to 1 to 5 with 0.5 increments
    plt.xticks(np.arange(1, 5.5, 0.5))

    # Remove y-axis tick labels
    plt.yticks([])

    # Add a legend on the right-hand side with custom labels
    legend_labels = [f'Cp_min = {data["Cp_min"]}', f'Cp_max = {data["Cp_max"]}', f'Cg_min = {data["Cg_min"]}',
                     f'Cg_max = {data["Cg_max"]}', f'w_min = {data["w_min"]}', f'w_max = {data["w_max"]}',
                     f'gwn_std_dev = {gwn_std_dev}']
    plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = os.path.splitext(os.path.basename(file_path))[
        0] + '_rpso_box_plot.png'
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches='tight')

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
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract relevant data
    pso_type = data['pso_type']
    averages = data.get('averages', [])

    if not averages:
        print("No data for averages found in the JSON file.")
        return

    # Create a figure for the averages plot
    plt.figure(figsize=(10, 6))
    plt.plot(averages, label='Averages', color='blue')

    plt.title(f'Averages Plot for {pso_type}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Value')
    plt.grid()
    plt.legend()

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = os.path.splitext(os.path.basename(file_path))[
        0] + '_averages_plot.png'
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_fitness_histories(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract relevant data
    pso_type = data['pso_type']
    fitness_histories = data['fitness_histories']

    # Create a figure for the combined plot
    plt.figure(figsize=(10, 6))

    # Plot each fitness history with a different color
    for i, history in enumerate(fitness_histories):
        plt.plot(history, label=f'History {i + 1}')

    plt.title(f'All Fitness Histories for {pso_type}')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.grid()

    # Hardcoded legend labels
    legend_labels = [f'c1 = {data["c1"]}', f'c2 = {data["c2"]}', f'w = {data["inertia"]}',
                     f'Particles = {data.get("num_particles", "")}', f'Iterations = {data.get("iterations", "")}']
    plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.0, 0.85))

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = os.path.splitext(os.path.basename(file_path))[
        0] + '_fitness_histories_plot.png'
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches='tight')

    # Show the plot
    plt.show()


def gbest_box_plot(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract relevant data
    pso_type = data['pso_type']
    fitness_histories = data['fitness_histories']

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
    num_particles = data.get('num_particles', '')
    iterations = data.get('iterations', '')

    # Create a horizontal box plot of the final fitness values with the x-axis range set to 1 to 5 with 0.5 increments
    plt.figure(figsize=(8, 4))  # Adjust the figsize for a shorter y-axis
    boxplot = plt.boxplot(final_fitness_values, vert=False)
    plt.title(f'Final Fitness Histories for {pso_type}')
    plt.xlabel('Final Fitness Value')

    # Adjust the horizontal positions for the "Best" and "Worst" labels
    offset = 0.1  # Vertical offset for text labels
    best_x = best_fitness + 0.03  # Slightly to the right
    worst_x = worst_fitness + 1.3  # Slightly to the left
    plt.text(best_x, 1 + offset, f'Best: {best_fitness:.3f}',
             horizontalalignment='left', verticalalignment='center')
    plt.text(worst_x, 1 + offset, f'Worst: {worst_fitness:.3f}',
             horizontalalignment='right', verticalalignment='center')

    # Set the x-axis range to 1 to 5 with 0.5 increments
    plt.xticks(np.arange(1, 5.5, 0.5))

    # Remove y-axis tick labels
    plt.yticks([])

    # Add a legend on the right-hand side with custom labels
    legend_labels = [f'c1 = {data["c1"]}', f'c2 = {data["c2"]}', f'w = {data["inertia"]}',
                     f'Particles = {num_particles}', f'Iterations = {iterations}']
    plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.0, 0.85))

    # Save the plot as an image in the same folder as the JSON file
    json_folder = os.path.dirname(file_path)
    plot_filename = os.path.splitext(os.path.basename(file_path))[
        0] + '_box_plot.png'
    plot_filepath = os.path.join(json_folder, plot_filename)
    # bbox_inches='tight' ensures that the legend is not cut off
    plt.savefig(plot_filepath, bbox_inches='tight')

    # Show the plot
    plt.show()

    # Display the statistics
    print(f"Best Final Fitness: {best_fitness:.3f}")
    print(f"Worst Final Fitness: {worst_fitness:.3f}")
    print(f"Mean Final Fitness: {mean_fitness:.3f}")
    print(f"Standard Deviation: {std_deviation:.3f}")
    print(f"Median Final Fitness: {median_fitness:.3f}")


def display_random_uniform_distribution_search_results():
    # Specify the path to the sub-folder containing the JSON files
    subfolder_path = 'opt_ann_gbest_uniform_distribution_search/calm_data_set/100_runs'

    # Initialize an empty list to store the fitness values for each run
    fitness_values = []

    # Iterate through the JSON files in the sub-folder
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.json'):
            json_file_path = os.path.join(subfolder_path, filename)
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                fitness_histories = data['fitness_histories']

                # Extract the final values from each sub-array
                final_values = [subarray[-1] for subarray in fitness_histories]

                # Append the final values for this run
                fitness_values.append(final_values)

    # Generate 'runs' based on the number of JSON files processed
    runs = [f'Run {i+1}' for i in range(len(fitness_values))]

    # Create a single vertical box plot for all runs
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot a box plot for all runs
    ax.boxplot(fitness_values, vert=True, widths=0.7,
               showfliers=False, patch_artist=True)

    # Set labels and title
    ax.set_xticklabels(runs, rotation=45)
    ax.set_xlabel('Runs')
    ax.set_ylabel('Fitness Values')
    ax.set_title('Box Plot of Fitness Values for All Runs')

    # Remove borders around the box plots
    for box in ax.artists:
        box.set_edgecolor('black')

    # Show the plot
    plt.tight_layout()
    plt.show()


def create_verification_result_plot():
    file_path = "./verification_stats/calm_random_set/verification_stats.json"

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    verification_stats = data.get('verification_stats', [])
    sorted_stats = sorted(verification_stats,
                          key=lambda x: x.get('MedAE', float('inf')))
    sorted_stats = sorted_stats[:5]
    if not sorted_stats:
        print("No data for verification stats found in the JSON file.")
        return
    print(sorted_stats)
    # Define a list of colors to use for the bars
    colors = ['b', 'g', 'r', 'c', 'm']

    titles = [
        sorted_stats[0]["File Name"],
        sorted_stats[1]["File Name"],
        sorted_stats[2]["File Name"],
        sorted_stats[3]["File Name"],
        sorted_stats[4]["File Name"],
    ]

    for i, stats in enumerate(sorted_stats):
        metric_names = ['MAE', 'MSE', 'RMSE',
                        'MedAE', 'Min Error', 'Max Error']
        metric_values = [stats.get(metric_name, 0)
                         for metric_name in metric_names]

        # Set the title for the current plot

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.xlabel('Metrics')
        plt.ylabel('Meters')
        plt.title(titles[i])
        plt.ylim(0, 9)

        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.15,
                     value, f'{value:.2f}', ha='center', va='bottom')

        # Save the plot image in the same folder as the JSON file
        output_folder = os.path.dirname(file_path)
        plot_file_name = f'plot_{i+1}.png'
        plot_file_path = os.path.join(output_folder, plot_file_name)
        plt.savefig(plot_file_path)

        plt.show()


def generate_verification_set_absolute_error_histogram():
    data_predicted = [
        [0.9676792621612549, 17.357040405273438],
        [2.6409964561462402, 15.294694900512695],
        [2.1081995964050293, 10.434110641479492],
        [6.703535556793213, 10.339791297912598],
        [8.539361953735352, 9.527189254760742],
        [7.79000997543335, 12.284538269042969],
        [2.5002849102020264, 8.09228229522705],
        [2.2756917476654053, 8.484781265258789],
        [-0.7392514944076538, 4.867146968841553],
        [0.11716413497924805, 2.299685478210449],
        [0.1702350378036499, 2.249837875366211],
        [2.6698415279388428, 3.3576321601867676],
        [4.570627689361572, 4.157527446746826],
        [5.197019577026367, 3.5862908363342285],
        [7.182644844055176, 1.285169005393982],
        [9.003217697143555, 2.6410484313964844],
        [12.317537307739258, 2.792997360229492],
        [10.52513599395752, 4.923949241638184],
        [10.362205505371094, 5.984879493713379],
        [10.361896514892578, 2.081681251525879],
        [9.13701343536377, 1.4089242219924927],
        [12.287692070007324, 0.5656107664108276],
        [13.53325366973877, 1.7982851266860962],
        [12.179170608520508, 1.6569218635559082],
        [12.856273651123047, 3.781865119934082],
        [15.094114303588867, 5.990777969360352],
        [15.07192325592041, 7.206153869628906],
        [14.618402481079102, 11.231653213500977],
        [15.692530632019043, 11.11452865600586],
        [15.49174976348877, 11.286646842956543],
        [13.831721305847168, 10.840497970581055],
        [13.490255355834961, 9.95637035369873],
        [9.806777954101562, 8.223896026611328],
        [8.808314323425293, 6.606976509094238],
        [7.799278259277344, 4.539035797119141],
        [8.771146774291992, 5.2889275550842285],
        [8.795844078063965, 8.306037902832031],
        [11.381338119506836, 10.778519630432129],
        [11.32949161529541, 11.74087905883789],
        [8.807378768920898, 12.117585182189941],
        [7.27183198928833, 13.179265975952148],
        [5.405222415924072, 13.913012504577637],
        [1.0529276132583618, 16.83958625793457],
        [1.1863726377487183, 16.02958106994629],
        [1.0113542079925537, 10.98906135559082]
    ]

    data_actual = [
        [5,  19],
        [5,  16],
        [5, 12],
        [7,  10],
        [9,  10],
        [8,  11],
        [5,   9],
        [2,   8],
        [1,   5],
        [2,   3],
        [4,   2],
        [5,   3],
        [6.5,  4],
        [7, 4],
        [9, 2],
        [11, 2],
        [12, 3],
        [12, 4],
        [13, 5],
        [13, 1],
        [14, 1],
        [15, 1],
        [16, 1],
        [17, 1],
        [17, 3],
        [17, 6],
        [17.5,  7],
        [18, 9],
        [18, 11],
        [16, 10.5],
        [16, 11],
        [12, 9],
        [11, 8],
        [10, 7],
        [9, 6],
        [10, 5],
        [9, 8],
        [11, 10],
        [12, 12],
        [10.5, 13],
        [9, 13.5],
        [7, 13],
        [3, 18],
        [3, 16],
        [1, 11],
    ]

    data_errors = [
        [4.032320737838745, 1.6429595947265625],
        [2.3590035438537598, 0.7053050994873047],
        [2.8918004035949707, 1.5658893585205078],
        [0.2964644432067871, 0.33979129791259766],
        [0.46063804626464844, 0.4728107452392578],
        [0.2099900245666504, 1.2845382690429688],
        [2.4997150897979736, 0.9077177047729492],
        [0.2756917476654053, 0.48478126525878906],
        [1.7392514944076538, 0.13285303115844727],
        [1.882835865020752, 0.7003145217895508],
        [3.82976496219635, 0.24983787536621094],
        [2.3301584720611572, 0.3576321601867676],
        [1.9293723106384277, 0.15752744674682617],
        [1.8029804229736328, 0.4137091636657715],
        [1.8173551559448242, 0.7148309946060181],
        [1.9967823028564453, 0.6410484313964844],
        [0.3175373077392578, 0.2070026397705078],
        [1.4748640060424805, 0.9239492416381836],
        [2.6377944946289062, 0.9848794937133789],
        [2.638103485107422, 1.081681251525879],
        [4.8629865646362305, 0.4089242219924927],
        [2.712307929992676, 0.43438923358917236],
        [2.4667463302612305, 0.7982851266860962],
        [4.820829391479492, 0.6569218635559082],
        [4.143726348876953, 0.781865119934082],
        [1.9058856964111328, 0.009222030639648438],
        [2.42807674407959, 0.20615386962890625],
        [3.3815975189208984, 2.2316532135009766],
        [2.307469367980957, 0.11452865600585938],
        [0.5082502365112305, 0.786646842956543],
        [2.168278694152832, 0.1595020294189453],
        [1.490255355834961, 0.9563703536987305],
        [1.1932220458984375, 0.22389602661132812],
        [1.191685676574707, 0.3930234909057617],
        [1.2007217407226562, 1.4609642028808594],
        [1.2288532257080078, 0.2889275550842285],
        [0.20415592193603516, 0.30603790283203125],
        [0.38133811950683594, 0.7785196304321289],
        [0.6705083847045898, 0.2591209411621094],
        [1.6926212310791016, 0.8824148178100586],
        [1.72816801071167, 0.32073402404785156],
        [1.5947775840759277, 0.9130125045776367],
        [1.9470723867416382, 1.1604137420654297],
        [1.8136273622512817, 0.029581069946289062],
        [0.011354207992553711, 0.010938644409179688]
    ]

    # Filter data_predicted based on data_errors
    filtered_data_predicted = [data_predicted[i] for i, point in enumerate(
        data_errors) if any(val >= 1 for val in point)]

    # Split the filtered data into x and y coordinates
    x1 = [item[0] for item in filtered_data_predicted]
    y1 = [item[1] for item in filtered_data_predicted]

    x2 = [item[0] for item in data_actual]
    y2 = [item[1] for item in data_actual]

    # Create a scatter plot for the first set of data (blue)
    plt.scatter(x1, y1, color='blue', label='Predicted')

    # Create a scatter plot for the second set of data (red)
    plt.scatter(x2, y2, color='red', label='Actual')

    # Add labels and a title
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Scatter Plot of 2D Data')

    # Add a legend to distinguish between the two data sets
    plt.legend()

    # Show the plot
    plt.show()


def plot_fitness_histories_from_json(json_file_path):
    # Get the full path to the JSON file, assuming it's in a subfolder named "data"

    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        fitness_histories = data.get("fitness_histories", [])

        if not fitness_histories:
            print("No fitness histories found in the JSON file.")
            return

        for history in fitness_histories:
            plt.plot(history, label="Fitness History")

        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Fitness History Plot')
        plt.legend()
        plt.show()

    except FileNotFoundError:
        print(f"File '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from '{json_file_path}'.")


# Replace with the name of your JSON file
json_file = "opt_ann_rpso_stats/stats_2023-10-23_12-12-21.json"
# plot_fitness_histories_from_json(json_file)
