import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data from the file
file_path = './opt_ann_gbest_stats/stats_2023-09-24_16-23-32.json'
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# Accessing specific data from the JSON
pso_type = data['pso_type']
pso_runs = data['pso_runs']
min_best_fitness = data['min_best_fitness']
mean_best_fitness = data['mean_best_fitness']
max_best_fitness = data['max_best_fitness']
averages = data['averages']
position_bounds = data['position_bounds']
velocity_bounds = data['velocity_bounds']
fitness_threshold = data['fitness_threshold']
num_particles = data['num_particles']
c1 = data['c1']
c2 = data['c2']
w = data['w']
iterations = data['iterations']
elapsed_time = data['elapsed_time']
best_weights = data['best_weights']
fitness_histories = data['fitness_histories']

# Now you can work with the extracted data as needed
# For example, printing some of the values:
print(f"PSO Type: {pso_type}")
print(f"Number of PSO Runs: {pso_runs}")
print(f"Minimum Best Fitness: {min_best_fitness}")
print(f"Mean Best Fitness: {mean_best_fitness}")
print(f"Maximum Best Fitness: {max_best_fitness}")

# You can iterate through the arrays and inner arrays as well, e.g.:
# for avg in averages:
#     print(f"Average: {avg}")

# for lower_bound, upper_bound in zip(position_bounds, velocity_bounds):
#     print(f"Position Bounds: {lower_bound} to {upper_bound}")

# for weight in best_weights:
#     print(f"Best Weight: {weight}")

# for history in fitness_histories:
#     print("Fitness History:")
#     for fitness in history:
#         print(f"  {fitness}")


# Extract the final fitness values from each history
final_fitness_values = [history[-1] for history in fitness_histories]

# Find the best and worst fitness values among all histories
best_fitness = min(final_fitness_values)
worst_fitness = max(final_fitness_values)

# Calculate mean, standard deviation, and other statistics
mean_fitness = np.mean(final_fitness_values)
std_deviation = np.std(final_fitness_values)
median_fitness = np.median(final_fitness_values)


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

# Add a legend on the right-hand side with c1, c2, and w values
legend_labels = [f'c1 = {c1}', f'c2 = {c2}', f'w = {w}',
                 f'Particles = {num_particles}', f'Iterations = {iterations}']
plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.0, 1.0))

# Show the plot
plt.show()
