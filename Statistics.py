import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler

from CostFunctions import get_fingerprinted_data


def get_final_model(model, particle):
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        num_weights = weights.size
        num_biases = biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = particle[:num_weights]
        sliced_biases = particle[num_weights: num_weights + num_biases]

        # Update the continuous_values array for the next iteration
        particle = particle[num_weights + num_biases:]

        # Set the sliced weights and biases in the layer
        layer.set_weights(
            [sliced_weights.reshape(weights.shape),
             sliced_biases.reshape(biases.shape)]
        )
    return model


def make_coordinate_prediction_with_ann_model(model, swarm_best_position):
    X_train, X_test, y_train, y_test = get_fingerprinted_data()
    finalModel = get_final_model(model, swarm_best_position)
    predictions = finalModel.predict(X_test)
    print(explained_variance_score(y_test, predictions))

    # this should produce (1, 0)
    some_position = [[75, 87, 80, 6920, 17112, 17286]]
    # this should produce (8,6)
    some_position_2 = [[72, 78, 81, 8503, 8420, 8924]]

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    transformed_some_position = scaler.transform(some_position_2)

    x_value = finalModel.predict(transformed_some_position)
    print(x_value)


def get_ann_node_count(ann_nodes):
    """
    Calculate the sum of elements in each sub-array and return individual counts per array,
    as well as the total count of all elements across all sub-arrays.

    Parameters:
    ann_nodes (list of np.ndarray): A list containing numpy arrays, each representing a sub-array.

    Returns:
    tuple: A tuple containing two elements:
        - individual_ann_node_count (list of int): A list of sums of elements for each sub-array.
        - total_nodes_count (int): Total count of all elements across all sub-arrays.
    """

    total_nodes_count = 0
    individual_ann_node_count = []

    for arr in ann_nodes:
        count = len(arr)
        total_nodes_count += count
        arr_sum = np.sum(arr)
        individual_ann_node_count.append(arr_sum)

    return individual_ann_node_count, total_nodes_count


# def get_swarm_total_particle_distance():
#     data = np.load("history_data.npy")
#     print("Total: ", data)
#     print("----------------")
#     for run_num, pso_run in enumerate(data):
#         print("Run: ", run_num, pso_run)
#         for iteration, swarm_positions in enumerate(pso_run):
#             print(f"Positions for iteration {iteration + 1}: ", swarm_positions)
#             for particle_num, particle_position in enumerate(swarm_positions):
#                 print(
#                     f"Particle position for particle {particle_num}: ",
#                     particle_position,
#                 )


def calculate_distance(position1, position2):
    return np.linalg.norm(position1 - position2)


def get_swarm_total_particle_distance(data):
    """
    Calculate the total distance between particles for each iteration of each PSO run.

    This function computes the total distance between all pairs of particles in each iteration
    of every PSO run and returns a nested structure representing these distances.

    Parameters:
        data (list of lists of lists): A 3D list containing the position histories
            of particles for each iteration within each PSO run.
            Format: [ [ [run1_iter1_positions], [run1_iter2_positions], ... ],
                      [ [run2_iter1_positions], [run2_iter2_positions], ... ],
                      ... ]

    Returns:
        list of lists: A nested structure representing the total distances between particles.
            The outermost list corresponds to different PSO runs, the inner lists correspond
            to iterations within each run, and each element of the inner lists represents
            the total distance for that iteration.
            Format: [ [run1_iter1_distance, run1_iter2_distance, ... ],
                      [run2_iter1_distance, run2_iter2_distance, ... ],
                      ... ]

    Example:
        >>> swarm_positions = [ [ [iter1_particle1, iter1_particle2, ...], [iter2_particle1, iter2_particle2, ...], ... ],
        >>>                     [ [iter1_particle1, iter1_particle2, ...], [iter2_particle1, iter2_particle2, ...], ... ],
        >>>                     ... ]
        >>> distances = get_swarm_total_particle_distance(swarm_positions)

    Note:
        The input data structure is expected to be organized as a list of PSO runs,
        where each run contains lists of particle positions for each iteration.
    """
    total_distances = []

    for run_num, pso_run in enumerate(data):
        run_distances = []

        for iteration, swarm_positions in enumerate(pso_run):
            iteration_distance = 0

            for particle_num, particle_position in enumerate(swarm_positions):
                for other_particle_num, other_particle_position in enumerate(
                    swarm_positions
                ):
                    if particle_num != other_particle_num:
                        distance = calculate_distance(
                            particle_position, other_particle_position
                        )
                        iteration_distance += distance

            run_distances.append(iteration_distance)

        total_distances.append(run_distances)

    return total_distances


def plot_average_total_distance(swarm_position_histories, pso_type, save_image=True):
    """
    Plot and visualize the average total particle distances over iterations for multiple PSO runs.

    This function calculates the average total particle distance for each iteration
    across multiple PSO runs and generates a plot to visualize the trend.

    Parameters:
        swarm_position_histories (list of lists of lists): A 3D list containing the position histories
            of particles for each iteration within each PSO run.
            Format: [
                    [
                        [run1_iter1_positions], [run1_iter2_positions], ... ],
                    [
                        [run2_iter1_positions], [run2_iter2_positions], ... ],
                    ... ]

        pso_type (str): Type of the PSO algorithm for labeling the plot and filename.

        save_image (bool, optional): Whether to save the plot as an image. Default is True.

    Returns:
        None. Generates a plot showing the average total particle distances over iterations.

    Example:
        >>> swarm_positions = [ [ [iter1_particle1, iter1_particle2, ...], [iter2_particle1, iter2_particle2, ...], ... ],
        >>>                     [ [iter1_particle1, iter1_particle2, ...], [iter2_particle1, iter2_particle2, ...], ... ],
        >>>                     ... ]
        >>> plot_average_total_distance(swarm_positions, "PSO_Type_A")

    Note:
        This function assumes that the input data structure is organized as a list of PSO runs,
        where each run is represented as a list of iterations, and each iteration is a list
        containing the particle positions for that specific iteration.
    """
    total_distances = get_swarm_total_particle_distance(
        swarm_position_histories)
    average_distances = np.mean(total_distances, axis=0)
    plt.plot(average_distances, label="Average Total Distance")

    if save_image:
        sub_folder = f"{pso_type}_stats"
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        plt.xlabel("Iteration")
        plt.ylabel("Total Particle Distance")
        plt.title("Average Total Particle Distance over Iterations")
        plt.legend()
        file_name = os.path.join(
            sub_folder, f"average_distance_plot_{timestamp}.png")
        plt.savefig(file_name)

        # Save results in a JSON file
        json_data = {
            "pso_type": pso_type,
            "total_distances": total_distances,
            "average_distances": average_distances.tolist(),
        }
        json_file_name = f"average_distance_{timestamp}.json"
        json_file_path = os.path.join(sub_folder, json_file_name)

        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)


def save_opt_ann_rpso_stats(fitness_histories, pso_type, pso_runs, position_bounds,
                            velocity_bounds, fitness_threshold, num_particles,
                            Cp_min, Cp_max, Cg_min, Cg_max, w_min, w_max,
                            gwn_std_dev, iterations, elapsed_time,
                            min_best_fitness, mean_best_fitness, max_best_fitness, best_weights):
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


def plot_averages_fitness_histories(fitness_histories, pso_type, pso_runs):
    """
    Saves an image with a history of the average fitness values of multiple PSO runs

    Parameters:
        fitness_histories (Array of arrays): Numbers of the fitness value of the swarm for each iteration

    Returns:
        Nothing, saves an image locally

    Example:
        >>> plot_averages_fitness_histories([[1,2,3], [2,4,6], [7,8,9]])
    """

    # Calculate the averages for each position
    averages = np.mean(fitness_histories, axis=0)

    # Generate the x-axis positions for the points
    x_positions = np.arange(1, len(averages) + 1)

    plt.figure()
    # Plot the points
    plt.scatter(x_positions, averages, label="Averages", color="blue")

    # Plot the lines between the points
    plt.plot(x_positions, averages, linestyle="-", color="blue")

    # Add labels and title
    plt.xlabel("Element Position")
    plt.ylabel("Average Value")
    plt.title(f"Average fitness histories of {pso_runs} PSO runs")
    f"{str(pso_type)}"

    # Show the plot
    plt.legend()
    plt.grid(True)

    sub_folder = f"{pso_type}_stats"
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.join(
        sub_folder, f"average_fitness_histories_{timestamp}.png")
    plt.savefig(file_name)

    # Save results in a JSON file
    json_data = {
        "pso_type": pso_type,
        "pso_runs": pso_runs,
        "fitness_histories": fitness_histories,
        "averages": averages.tolist(),
    }
    json_file_name = f"average_fitness_histories_{timestamp}.json"
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
    sub_folder = f"{pso_type}_stats"

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


def get_particle_dimension_history(
    swarm_position_histories, particle_index, *dimension_indices
):
    """
    Collects historical particle positions for specified dimensions from a series of runs.

    This function takes a series of runs, each containing iterations with particle positions,
    and extracts historical positions for a specific particle and dimensions of interest.

    Args:
        swarm_position_histories (list): A list of runs, each containing iterations with particle positions.
        particle_index (int): The index of the particle whose positions are being tracked.
        *dimension_indices (int): Variable number of dimension indices to track positions for.

    Returns:
        list: A list of arrays containing historical positions for each specified dimension.

    Example:
        swarm_positions = [
            [
                [(1, 2, 3), (2, 3, 4), (3, 4, 5)],
                [(4, 5, 6), (5, 6, 7), (6, 7, 8)]
            ],
            [
                [(7, 8, 9), (8, 9, 10), (9, 10, 11)],
                [(10, 11, 12), (11, 12, 13), (12, 13, 14)]
            ]
        ]

        history = plot_particle_dimension_history(swarm_positions, 0, 1, 2)
        # Returns: [[2, 5, 8], [3, 6, 9], [4, 7, 10]]
    """
    particle_position_history = []

    for run in swarm_position_histories:
        run_positions = [[] for _ in range(len(dimension_indices))]

        for iteration in run:
            dimensional_positions = iteration[particle_index]

            for i, dim_idx in enumerate(dimension_indices):
                if dim_idx < len(dimensional_positions):
                    position = dimensional_positions[dim_idx]
                    run_positions[i].append(position)

        particle_position_history.append(run_positions)

    # Save results in a JSON file
    json_data = {"particle_dimension_history": particle_position_history}
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    json_file_name = f"particle_dimension_history_{timestamp}.json"

    with open(json_file_name, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    return particle_position_history


def plot_particle_positions(swarm_position_histories, particle_index, *dimensions):
    particle_position_history = get_particle_dimension_history(
        swarm_position_histories, particle_index, *dimensions
    )
    """
    Plots historical particle positions of one particle

    Args:
        particle_position_history (list): A list of lists of arrays containing historical positions
        for each specified dimension, separated by runs.
    """
    for run_idx, run_positions in enumerate(particle_position_history):
        # Create a new figure
        plt.figure()

        # Plot lines for each data sublist
        for i, line_data in enumerate(run_positions):
            plt.plot(line_data, marker="o", label=f"Dimension {i}")

        # Add labels and title
        plt.xlabel("Iteration number")
        plt.ylabel("Position value")
        plt.title(
            f"Particle X's position in {len(run_positions)} different dimensions")
        plt.legend()
        # Show the plot
        plt.show()


def handle_data(
    fitness_histories, swarm_position_histories, PSO_TYPE, pso_runs, options
):
    plot_average_total_distance(swarm_position_histories, PSO_TYPE)
    plot_averages_fitness_histories(fitness_histories, PSO_TYPE, pso_runs)
    plot_all_fitness_histories(fitness_histories, options, PSO_TYPE, pso_runs)
    # plot_particle_positions(swarm_position_histories, 0, 0, 1, 2)
