import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_rpso_averages(file_path):
    # Load the JSON data from the file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract relevant data
    rpso_type = data['pso_type']
    averages = data.get('averages', [])

    if not averages:
        print("No data for averages found in the JSON file.")
        return

    # Create a figure for the averages plot
    plt.figure(figsize=(10, 6))
    plt.plot(averages, label='Averages', color='blue')

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
    legend_labels = [f'c1 = {data["c1"]}', f'c2 = {data["c2"]}', f'w = {data["w"]}',
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
    worst_x = worst_fitness - 0.03  # Slightly to the left
    plt.text(best_x, 1 + offset, f'Best: {best_fitness:.3f}',
             horizontalalignment='left', verticalalignment='center')
    plt.text(worst_x, 1 + offset, f'Worst: {worst_fitness:.3f}',
             horizontalalignment='right', verticalalignment='center')

    # Set the x-axis range to 1 to 5 with 0.5 increments
    plt.xticks(np.arange(1, 5.5, 0.5))

    # Remove y-axis tick labels
    plt.yticks([])

    # Add a legend on the right-hand side with custom labels
    legend_labels = [f'c1 = {data["c1"]}', f'c2 = {data["c2"]}', f'w = {data["w"]}',
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


# Example usage:
# file_path = './opt_ann_gbest_stats/stats_2023-09-24_17-06-41.json'
# opt_ann_gbest_box_plot(file_path)
# plot_fitness_histories(file_path)
# plot_averages(file_path)

# RPSO usage
# Example usage:
file_path = './opt_adam_params_with_rpso_stats/500_iterations_during_training/optimize_ann_optimizer_params_2023-09-24_23-27-33.json'
plot_rpso_averages(file_path)
plot_rpso_fitness_histories(file_path)
rpso_box_plot(file_path)
