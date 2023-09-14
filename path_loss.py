import numpy as np
import pyswarms as ps

# Actual distance (known value)
actual_distance = 3.0  # Adjust this to the actual distance in your scenario


def objective_function(particles):
    n_particles = particles.shape[0]
    distances = np.zeros(n_particles)

    for i in range(n_particles):
        A = particles[i, 0]
        n = particles[i, 1]
        RSSI = -77  # The measured RSSI value at the time of calculation

        # Calculate distance (d) based on RSSI using scientific notation
        calculated_distance = 10 ** ((RSSI - A) / (10 * n))

        # Calculate the squared difference between actual and calculated distances
        squared_difference = (actual_distance - calculated_distance) ** 2

        distances[i] = np.sqrt(squared_difference)

    return distances


bounds = (np.array([0, 0]), np.array([1000, 1000]))
options = {'c1': 0.145555, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(
    n_particles=10, dimensions=2, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(objective_function, iters=1000)

print(f"Optimized A: {pos[0]}")
print(f"Optimized n: {pos[1]}")
print(f"Calculated Distances: {10 ** ((-77 - pos[0]) / (10 * pos[1]))}")
