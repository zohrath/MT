from functools import partial
import multiprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import argparse
from CostFunctions import ann_weights_fitness_function, get_fingerprinted_data


def create_ann_model(input_shape):
    model = Sequential()
    model.add(Dense(units=6, activation="relu", input_dim=input_shape))
    model.add(Dense(units=6, activation="relu"))
    model.add(Dense(units=6, activation="relu"))
    model.add(Dense(units=2))
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def get_ann_dimensions(model):
    total_num_weights = sum(np.prod(w.shape) for w in model.get_weights()[::2])
    total_num_biases = sum(np.prod(b.shape) for b in model.get_weights()[1::2])
    dimensions = total_num_weights + total_num_biases
    return dimensions


# Define the Global Best PSO optimizer
class GlobalBestPSO:
    def __init__(
        self,
        num_particles,
        num_dimensions,
        max_iters,
        c1,
        c2,
        w,
        lower_bound,
        upper_bound,
        model,
        X_train,
        y_train,
    ):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_iters = max_iters
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.particles = np.random.uniform(
            low=lower_bound, high=upper_bound, size=(num_particles, num_dimensions)
        )
        self.velocities = np.zeros((num_particles, num_dimensions))
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.best_positions = self.particles.copy()
        self.best_fitness = np.array(
            [
                ann_weights_fitness_function(p, self.model, self.X_train, self.y_train)
                for p in self.best_positions
            ]
        )
        self.global_best_position = self.best_positions[np.argmin(self.best_fitness)]
        self.global_best_fitness = np.min(self.best_fitness)

    def optimize(self):
        for _ in range(self.max_iters):
            for i in range(self.num_particles):
                # Update particle velocity and position
                r1 = np.random.rand(self.num_dimensions)
                r2 = np.random.rand(self.num_dimensions)
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.best_positions[i] - self.particles[i])
                    + self.c2 * r2 * (self.global_best_position - self.particles[i])
                )
                self.particles[i] += self.velocities[i]

                # Clip particle position within bounds
                self.particles[i] = np.clip(
                    self.particles[i], self.lower_bound, self.upper_bound
                )

                # Update particle's best position and fitness
                fitness = ann_weights_fitness_function(
                    self.particles[i], self.model, self.X_train, self.y_train
                )
                if fitness < self.best_fitness[i]:
                    self.best_positions[i] = self.particles[i]
                    self.best_fitness[i] = fitness

                    # Update global best if needed
                    if fitness < self.global_best_fitness:
                        self.global_best_position = self.particles[i]
                        self.global_best_fitness = fitness


class RPSO:
    def __init__(
        self,
        num_particles,
        num_dimensions,
        max_iters,
        c2,
        w_min,
        w_max,
        cognitive_noise_stddev,
        social_noise_stddev,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        X_train,
        y_train,
        model,
        lower_bound,
        upper_bound,
    ):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_iters = max_iters
        self.c2 = c2
        self.w_min = w_min
        self.w_max = w_max
        self.cognitive_noise_stddev = cognitive_noise_stddev
        self.social_noise_stddev = social_noise_stddev
        self.Cp_min = Cp_min
        self.Cp_max = Cp_max
        self.Cg_min = Cg_min
        self.Cg_max = Cg_max
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.particles = np.random.uniform(
            low=lower_bound, high=upper_bound, size=(num_particles, num_dimensions)
        )
        self.velocities = np.zeros((num_particles, num_dimensions))
        self.best_positions = self.particles.copy()
        self.best_fitness = np.array(
            [
                ann_weights_fitness_function(p, model, X_train, y_train)
                for p in self.best_positions
            ]
        )
        self.global_best_position = self.best_positions[np.argmin(self.best_fitness)]
        self.global_best_fitness = np.min(self.best_fitness)

    def optimize(self):
        for iteration in range(self.max_iters):
            # Calculate inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iters)
            # Calculate dynamic cognitive parameter
            Cp = (self.Cp_max - self.Cp_min) * (
                (self.max_iters - iteration) / self.max_iters
            ) + self.Cp_min
            # Calculate dynamic social parameter
            Cg = (self.Cg_max - self.Cg_min) * (
                (self.max_iters - iteration) / self.max_iters
            ) + self.Cg_min
            # Add Gaussian white noise to Cp and Cg
            Cp_noise = np.random.normal(0, self.cognitive_noise_stddev)
            Cg_noise = np.random.normal(0, self.social_noise_stddev)
            for i in range(self.num_particles):
                # Update particle velocity and position
                r1 = np.random.rand(self.num_dimensions)
                r2 = np.random.rand(self.num_dimensions)
                self.velocities[i] = (
                    w * self.velocities[i]
                    + (Cp + Cp_noise)
                    * r1
                    * (self.best_positions[i] - self.particles[i])
                    + (Cg + Cg_noise)
                    * r2
                    * (self.global_best_position - self.particles[i])
                )
                self.particles[i] += self.velocities[i]

                # Clip particle position within bounds
                self.particles[i] = np.clip(
                    self.particles[i], self.lower_bound, self.upper_bound
                )

                # Update particle's best position and fitness
                fitness = ann_weights_fitness_function(
                    self.particles[i], self.model, self.X_train, self.y_train
                )
                if fitness < self.best_fitness[i]:
                    self.best_positions[i] = self.particles[i]
                    self.best_fitness[i] = fitness

                    # Update global best if needed
                    if fitness < self.global_best_fitness:
                        self.global_best_position = self.particles[i]
                        self.global_best_fitness = fitness


# X_train, X_test, y_train, y_test, _ = get_fingerprinted_data()

# # Create your Keras model
# model = create_ann_model(X_train.shape[1])
# total_number_of_values = get_ann_dimensions(model)

# # Parameters
# num_particles = 20
# num_dimensions = total_number_of_values
# max_iters = 100
# c1 = 0.5
# c2 = 0.3
# w = 0.9
# lower_bound = -1.0  # Adjust as needed based on your problem
# upper_bound = 1.0  # Adjust as needed based on your problem

# # Create and run the optimizer
# optimizer = GlobalBestPSO(
#     num_particles, num_dimensions, max_iters, c1, c2, w, lower_bound, upper_bound, model, X_train, y_train
# )
# optimizer.optimize()

# print("Best position found:", optimizer.global_best_position)
# print("Best fitness value:", optimizer.global_best_fitness)

# ------------------ THREADED ---------------#


def run_parameter_variation(
    upper_bound_values, lower_bound_values, optimizer_params, num_processes, num_runs
):
    # Create an empty list to store the results
    all_results = []

    # Loop through upper_bound and lower_bound values
    for upper_bound in upper_bound_values:
        for lower_bound in lower_bound_values:
            optimizer_params["upper_bound"] = upper_bound
            optimizer_params["lower_bound"] = lower_bound

            # Create and run RPSO optimizer
            optimizer = RPSO(**optimizer_params)

            with multiprocessing.Pool(processes=num_processes) as pool:
                optimizer_partial = partial(
                    run_optimizer,
                    optimizer=optimizer,
                )
                results_list = pool.map(optimizer_partial, range(num_runs))

            # Find the best result among all processes
            best_position, best_fitness = min(results_list, key=lambda x: x[1])

            # Store the results
            all_results.append(
                {
                    "upper_bound": upper_bound,
                    "lower_bound": lower_bound,
                    "best_position": best_position,
                    "best_fitness": best_fitness,
                }
            )

    return all_results


def run_optimizer(process_id, optimizer):
    print(f"Process {process_id} started.")

    optimizer.optimize()

    print(f"Process {process_id} completed.")
    return optimizer.global_best_position, optimizer.global_best_fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Particle Swarm Optimization")
    parser.add_argument(
        "--pso_type",
        choices=["gbest", "rpso"],
        default="gbest",
        help="Type of PSO algorithm to use",
    )

    args = parser.parse_args()

    X_train, X_test, y_train, y_test, _ = get_fingerprinted_data()

    # Create your Keras model
    model = create_ann_model(X_train.shape[1])
    total_number_of_values = get_ann_dimensions(model)

    # Parameters for optimization
    num_particles = 200
    num_dimensions = total_number_of_values
    max_iters = 100
    c1 = 0.5
    c2 = 0.3
    w = 0.9
    lower_bound = -1.0  # Adjust as needed based on your problem
    upper_bound = 1.0  # Adjust as needed based on your problem
    cognitive_noise_stddev = 0.07
    social_noise_stddev = 0.07
    Cp_min = 0.1
    Cp_max = 2.0
    Cg_min = 0.1
    Cg_max = 2.0
    w_min = 0.1
    w_max = 1.0

    if args.pso_type == "gbest":
        optimizer_args = (
            num_particles,
            num_dimensions,
            max_iters,
            c1,
            c2,
            w,
            lower_bound,
            upper_bound,
            model,
            X_train,
            y_train,
        )
        # Create and run GlobalBestPSO optimizer
        optimizer = GlobalBestPSO(
            num_particles=num_particles,
            num_dimensions=num_dimensions,
            max_iters=max_iters,
            c1=c1,
            c2=c2,
            w=w,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model=model,
            X_train=X_train,
            y_train=y_train,
        )
    elif args.pso_type == "rpso":
        optimizer_params = {
            "num_particles": num_particles,
            "num_dimensions": num_dimensions,
            "max_iters": max_iters,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model": model,
            "X_train": X_train,
            "y_train": y_train,
            "cognitive_noise_stddev": cognitive_noise_stddev,
            "social_noise_stddev": social_noise_stddev,
            "Cp_min": Cp_min,
            "Cp_max": Cp_max,
            "Cg_min": Cg_min,
            "Cg_max": Cg_max,
            "w_min": w_min,
            "w_max": w_max,
        }

        # Create and run RPSO optimizer
        optimizer = RPSO(**optimizer_params)

    else:
        raise ValueError("Invalid PSO type specified")

    num_processes = 6  # Change this to the desired number of parallel processes
    num_runs = 6
    with multiprocessing.Pool(processes=num_processes) as pool:
        optimizer_partial = partial(
            run_optimizer,
            optimizer=optimizer,
        )
        results_list = pool.map(optimizer_partial, range(num_runs))

    # Find the best result among all processes
    best_position, best_fitness = min(results_list, key=lambda x: x[1])

    print("Best position found:", best_position)
    print("Best fitness value:", best_fitness)

# --------------- PARAMETER OPT --------------

# upper_bound_values = [3.0, 2.75, 2.5, 2.25, 2.0]
# lower_bound_values = [-1.0, -0.75, -0.5, -0.25, 0]

# # Call the function
# all_results = run_parameter_variation(
#     upper_bound_values, lower_bound_values, optimizer_params, num_processes, num_runs
# )

# # Print or analyze the results
# for result in all_results:
#     print("Upper bound:", result["upper_bound"])
#     print("Lower bound:", result["lower_bound"])
#     print("Best position found:", result["best_position"])
#     print("Best fitness value:", result["best_fitness"])
#     print("-------------------------")
