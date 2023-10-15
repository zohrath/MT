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
        "best_swarm_weights":best_swarm_weights,
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
    ax.boxplot(fitness_values, vert=True, widths=0.7, showfliers=False, patch_artist=True)

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

# Example usage:
# file_path = './opt_ann_gbest_stats/stats_2023-09-24_17-06-41.json'
# opt_ann_gbest_box_plot(file_path)
# plot_fitness_histories(file_path)
# plot_averages(file_path)
# RPSO usage
# Example usage:
file_path_rpso = './opt_ann_rpso_stats/noisy_data_set/no_gwn_val/stats_2023-09-29_17-05-42.json'
# plot_rpso_averages(file_path_rpso, 1)
# rpso_box_plot(file_path_rpso)
# plot_rpso_fitness_histories(file_path_rpso)

# file_path_gbest = './opt_adam_params_with_gbest_stats/noisy_data_set/500_iterations_during_training/random_search_params/optimize_ann_optimizer_params_2023-10-12_13-57-08.json'
# plot_fitness_histories(file_path_gbest)
# gbest_box_plot(file_path_gbest)
# plot_averages(file_path_gbest)
