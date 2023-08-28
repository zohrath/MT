import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate some synthetic data for demonstration
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define a function to create a Keras model
def create_model(optimizer_idx, activation_idx, hidden_units):
    optimizers = [Adam(), SGD()]
    activations = ["relu", "sigmoid"]

    model = Sequential()
    model.add(
        Dense(
            hidden_units,
            input_dim=X_train.shape[1],
            activation=activations[int(activation_idx)],
        )
    )
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers[int(optimizer_idx)],
        metrics=["accuracy"],
    )
    return model


# Define the fitness function for PSO
def fitness_function(params):
    optimizer_idx, activation_idx, hidden_units = params

    model = create_model(optimizer_idx, activation_idx, int(hidden_units))
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(
        int
    )  # Convert probabilities to binary classes
    accuracy = accuracy_score(y_test, y_pred_classes)
    return -accuracy  # Negative sign because PSO minimizes the fitness function


# Define PSO parameters
n_particles = 10
n_iterations = 20
n_dimensions = 3
bounds = [(0, 1), (0, 1), (8, 32)]  # optimizer, activation, hidden_units

# Initialize particle positions and velocities randomly within bounds
positions = np.random.rand(n_particles, n_dimensions)
velocities = np.random.rand(n_particles, n_dimensions)

# Initialize the best particle position and fitness value
best_positions = positions.copy()
best_fitness = np.array([fitness_function(params) for params in best_positions])

# Initialize the global best position and fitness value
global_best_position = best_positions[np.argmin(best_fitness)]
global_best_fitness = np.min(best_fitness)

# PSO main loop
for iteration in range(n_iterations):
    for i in range(n_particles):
        # Update particle velocities and positions using the PSO formula
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = (
            velocities[i]
            + r1 * (best_positions[i] - positions[i])
            + r2 * (global_best_position - positions[i])
        )
        positions[i] = positions[i] + velocities[i]

        # Apply bounds to ensure parameters stay within limits
        for j in range(n_dimensions):
            positions[i, j] = np.clip(positions[i, j], bounds[j][0], bounds[j][1])
        positions[i, -1] = int(positions[i, -1])  # Convert hidden_units to integer

        # Evaluate fitness of the updated position
        fitness = fitness_function(positions[i])

        # Update the best position and fitness for the particle
        if fitness < best_fitness[i]:
            best_positions[i] = positions[i]
            best_fitness[i] = fitness

            # Update the global best position and fitness if necessary
            if fitness < global_best_fitness:
                global_best_position = positions[i]
                global_best_fitness = fitness


# Print the best parameters found by PSO
best_optimizer, best_activation, best_hidden_units = global_best_position
print(
    "Best parameters: Optimizer={}, Activation={}, Hidden Units={}".format(
        int(best_optimizer), int(best_activation), int(best_hidden_units)
    )
)
