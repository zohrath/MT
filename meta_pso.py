# Import modules
import numpy as np

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

from CostFunctions import get_fingerprinted_data
from misc import create_ann_model, get_ann_dimensions


X_train, X_test, y_train, y_test, _ = get_fingerprinted_data()
model = create_ann_model(X_train.shape[1])


def ann_weights_fitness_function(particle):
    for layer in model.layers:
        weights, biases = layer.get_weights()
        num_weights = weights.size
        num_biases = biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = particle[:num_weights]
        sliced_biases = particle[num_weights : num_weights + num_biases]

        # Update the continuous_values array for the next iteration
        particle = particle[num_weights + num_biases :]

        # Set the sliced weights and biases in the layer
        new_weights = [
            sliced_weights.reshape(weights.shape),
            sliced_biases.reshape(biases.shape),
        ]
        layer.set_weights(new_weights)

    # Evaluate the model and get the evaluation metrics
    evaluation_metrics = model.evaluate(X_train, y_train, verbose=0)
    # rmse = np.sqrt(evaluation_metrics)

    return evaluation_metrics


# Define a custom objective function
def custom_objective_function(particles):
    best_cost = float("inf")

    for particle in particles:
        mse = ann_weights_fitness_function(particle)
        if mse < best_cost:
            best_cost = mse
    return best_cost


# Set-up hyperparameters for the meta-optimizer PSO
meta_options = {"c1": 0.5, "c2": 0.3, "w": 0.9}


# Create a function to optimize the PSO parameters
def optimize_pso_params(params):
    best_cost = float("inf")
    max_bound = 5.0 * np.ones(3)
    min_bound = np.zeros(3)
    bounds = (min_bound, max_bound)
    for particle in params:
        c1, c2, w = particle
        pso_options = {"c1": c1, "c2": c2, "w": w}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=10,
            dimensions=get_ann_dimensions(model),
            options=pso_options,
        )
        cost, _ = optimizer.optimize(custom_objective_function, iters=20)
        if cost < best_cost:
            best_cost = cost

    return best_cost


# The max and min values the optimized parameters can take,
# the dimensions are the same the number of params
max_bound = 5.0 * np.ones(3)
min_bound = np.zeros(3)
bounds = (min_bound, max_bound)

# Create a second PSO to optimize the parameters of the first PSO
meta_optimizer = ps.single.GlobalBestPSO(
    n_particles=30, dimensions=3, options=meta_options, bounds=bounds
)

# Perform optimization of the PSO parameters
best_cost, best_params = meta_optimizer.optimize(optimize_pso_params, iters=20)

print("Best cost:", best_cost)
print("Best parameters:", best_params)
