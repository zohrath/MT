import json
import matplotlib.pyplot as plt
import numpy as np


def get_vals_from_stats(data):
    all_best_positions = []
    last_fitness_values_dict = {}

    for index, pso_run_data in enumerate(data["calm"]["stats"]):
        last_fitness_values = []
        if pso_run_data:
            all_best_positions.append(pso_run_data["best_swarm_position"])
            for fitness_history in pso_run_data["fitness_histories"]:
                last_fitness_values.append(fitness_history[-1])
        last_fitness_values_dict[index] = last_fitness_values

    return all_best_positions, last_fitness_values_dict


def open_json_file(json_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            return get_vals_from_stats(data)

    except FileNotFoundError:
        print(f"The file '{json_file_path}' does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


def create_combined_box_plots(data_dict1, data_dict2=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create a list to hold the data for all box plots
    all_data1, all_data2 = [], []

    # Create and add box plots for each key (index) in the first dictionary
    for index, values in data_dict1.items():
        box_plot = axes[0].boxplot(
            values, vert=False, positions=[index], widths=0.6)
        all_data1.append(box_plot)

    # Customize the plot labels and appearance for the first subplot
    axes[0].set_title("Calm data set")
    axes[0].set_xlabel("Fitness Values")
    axes[0].set_ylabel("Index")
    axes[0].set_yticks(list(data_dict1.keys()))
    axes[0].set_yticklabels([f"Index {index}" for index in data_dict1.keys()])

    if data_dict2 is not None:
        # Create and add box plots for each key (index) in the second dictionary
        for index, values in data_dict2.items():
            box_plot = axes[1].boxplot(
                values, vert=False, positions=[index], widths=0.6)
            all_data2.append(box_plot)

        # Customize the plot labels and appearance for the second subplot
        axes[1].set_title("Noisy data set")
        axes[1].set_xlabel("Fitness Values")
        axes[1].set_ylabel("Index")
        axes[1].set_yticks(list(data_dict2.keys()))
        axes[1].set_yticklabels(
            [f"Index {index}" for index in data_dict2.keys()])
    fig.suptitle("Optimizing Adam parameters using GBest")
    # Show all the box plots in the same figure
    plt.tight_layout()
    plt.show()


def main():
    # Specify the path to your JSON file
    json_file_path_gbest = 'opt_adam_params_with_gbest_stats/combined_stats.json'
    json_file_path_rpso = 'opt_adam_params_with_rpso_stats/combined_stats.json'

    best_swarm_positios, last_fitness_values_gbest = open_json_file(
        json_file_path_gbest)

    best_swarm_positios, last_fitness_values_rpso = open_json_file(
        json_file_path_rpso)

    # print("Last fitness values:", last_fitness_values)
    # Example data for the two horizontal box plots
    data1 = np.random.normal(0, 1, 50)
    data2 = np.random.normal(1, 1, 50)

    # Call the function to create and display the double horizontal box plot
    create_combined_box_plots(
        last_fitness_values_gbest, last_fitness_values_rpso)


if __name__ == "__main__":
    main()
