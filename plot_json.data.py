import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the JSON files
dir_path = 'gbest_stats/c-20-w08'

# Get a list of all JSON files in the directory
json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]

# Create a subplot grid based on the number of JSON files
num_files = len(json_files)
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_files + num_cols - 1) // num_cols

# Create a figure and subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
axes = axes.flatten()

# Loop through JSON files and plot fitness histories
for i, json_file in enumerate(json_files):
    json_path = os.path.join(dir_path, json_file)

    with open(json_path, 'r') as file:
        data = json.load(file)

    fitness_histories = data['fitness_histories']
    num_runs = len(fitness_histories)
    iterations = data['iterations']
    num_particles = data['num_particles']

    # Plot fitness histories
    for run_index, run_fitness in enumerate(fitness_histories):
        ax = axes[i]
        ax.plot(range(len(run_fitness)), run_fitness, label=f'Run {run_index}')
        ax.set_title(
            f'Iterations: {iterations}, Num Particles: {num_particles}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness')
        ax.legend()

    # Extract statistics
    statistics = data['statistics']
    min_fitness = statistics['min_fitness']
    mean_fitness = statistics['mean_fitness']
    max_fitness = statistics['max_fitness']
    std_dev_fitness = statistics['std_dev_fitness']

    # Create a separate sub-plot for the box plots
    ax2 = axes[i + num_files]

    # Create box plots for statistics
    box_plot_data = [min_fitness, mean_fitness, max_fitness, std_dev_fitness]
    labels = ['Min', 'Mean', 'Max', 'Std Dev']
    ax2.boxplot([box_plot_data], vert=False, labels=labels)
    ax2.set_title(
        f'Statistics (Iterations: {iterations}, Num Particles: {num_particles})')

# Remove any empty subplots
for i in range(len(json_files), num_cols * num_rows):
    fig.delaxes(axes[i])

# Adjust layout
plt.tight_layout()
plt.show()
