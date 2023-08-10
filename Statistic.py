import time
import numpy as np
import matplotlib.pyplot as plt


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
    total_distances = get_swarm_total_particle_distance(swarm_position_histories)
    average_distances = np.mean(total_distances, axis=0)
    plt.plot(average_distances, label="Average Total Distance")

    if save_image:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        plt.xlabel("Iteration")
        plt.ylabel("Total Particle Distance")
        plt.title("Average Total Particle Distance over Iterations")
        plt.legend()
        plt.savefig(f"average_distance_plot_{pso_type}_{timestamp}.png")

    plt.xlabel("Iteration")
    plt.ylabel("Total Particle Distance")
    plt.title("Average Total Particle Distance over Iterations")
    plt.legend()
    plt.show()


def plot_averages_fitness_histories(fitness_histories, pso_type):
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

    # Plot the points
    plt.scatter(x_positions, averages, label="Averages", color="blue")

    # Plot the lines between the points
    plt.plot(x_positions, averages, linestyle="-", color="blue")

    # Add labels and title
    plt.xlabel("Element Position")
    plt.ylabel("Average Value")
    f"{str(pso_type)}"

    # Show the plot
    plt.legend()
    plt.grid(True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save the plot to an image file
    file_name = f"average_fitness_histories_{timestamp}.png"
    plt.savefig(file_name)


def plot_all_fitness_histories(fitness_histories, options, pso_type):
    plt.figure(figsize=(10, 6))

    for i, fitness_history in enumerate(fitness_histories):
        plt.plot(fitness_history, label=f"PSO Run {i + 1}")

    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.title(f"{str(options['function_name'])} with {pso_type}")

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
        xy=(0.66, 0.8),
        xycoords="axes fraction",
        fontsize=10,
        va="center",  # Vertically center the annotation
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"fitness_histories_plot_{timestamp}.png"
    plt.savefig(file_name)
