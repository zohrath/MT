import os
import json
import matplotlib.pyplot as plt
import pandas as pd

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
