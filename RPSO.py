import numpy as np
from CostFunctions import (
    ann_node_count_fitness,
    get_fingerprinted_data,
    get_regression_data,
    linear_regression,
    penalized1,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
)

from Particle import Particle


def get_GWN(mean=0, std_dev=0.07):
    return np.random.normal(loc=mean, scale=std_dev, size=1)


class RPSOParticle:
    def __init__(
        self,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        max_iterations,
        num_dimensions,
        position_bounds,
        velocity_bounds,
        gwn_std_dev,
    ):
        self.num_dimensions = num_dimensions
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds

        self.position = self.initialize_particle_position()
        self.velocity = self.initialize_particle_velocity()

        self.best_position = self.position
        self.best_fitness = float("inf")

        self.position_history = np.empty((num_dimensions, max_iterations))

        self.Cg_min = Cg_min
        self.Cg_max = Cg_max
        self.Cp_min = Cp_min
        self.Cp_max = Cp_max
        self.w_min = w_min
        self.w_max = w_max
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.d1 = get_GWN()
        self.d2 = get_GWN()
        self.Cp = self.get_cognitive_parameter()
        self.Cg = self.get_social_parameter()
        self.inertia_weight = self.get_inertia_weight_parameter()
        self.gwn_std_dev = gwn_std_dev

    def initialize_particle_position(self):
        return np.random.uniform(
            self.position_bounds[0],
            self.position_bounds[1],
            size=self.num_dimensions,
        )

    def initialize_particle_velocity(self):
        return np.random.uniform(
            self.velocity_bounds[0], self.velocity_bounds[1], size=self.num_dimensions
        )

    def get_inertia_weight_parameter(self):
        return self.w_max - (self.w_max - self.w_min) * (
            self.current_iteration / self.max_iterations
        )

    def get_cognitive_parameter(self):
        return (self.Cp_max - self.Cp_min) * (
            (self.max_iterations - self.current_iteration) / self.max_iterations
        ) + self.Cp_min

    def get_social_parameter(self):
        return (self.Cg_min - self.Cg_max) * (
            (self.max_iterations - self.current_iteration) / self.max_iterations
        ) + self.Cg_max

    def get_update_velocity_inertia_part(self):
        return self.get_inertia_weight_parameter() * self.velocity

    def get_cognitive_velocity_part(self):
        updated_Cp = self.get_cognitive_parameter()
        r1 = np.random.uniform(0, 1)
        d1 = get_GWN(self.gwn_std_dev)
        return r1 * (updated_Cp + d1) * (self.best_position - self.position)

    def get_social_velocity_part(self, swarm_best_position):
        updated_Cg = self.get_social_parameter()
        r2 = np.random.uniform(0, 1)
        d2 = get_GWN(self.gwn_std_dev)
        return r2 * (updated_Cg + d2) * (swarm_best_position - self.position)

    def update_particle_velocity(
        self,
        swarm_best_position,
    ):
        inertia_param = self.get_update_velocity_inertia_part()
        cognitive_param = self.get_cognitive_velocity_part()
        social_param = self.get_social_velocity_part(swarm_best_position)

        updated_velocity = inertia_param + cognitive_param + social_param

        updated_velocity = np.clip(
            updated_velocity, self.velocity_bounds[0], self.velocity_bounds[1]
        )

        self.velocity = updated_velocity

    def update_position(self):
        new_position = np.clip(
            self.position + self.velocity,
            self.position_bounds[0],
            self.position_bounds[1],
        )
        np.append(self.position_history, new_position)
        self.position = new_position

    def update_current_iteration(self):
        self.current_iteration += 1


class RPSO:
    def __init__(
        self,
        iterations,
        num_particles,
        num_dimensions,
        position_bounds,
        velocity_bounds,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        threshold,
        function,
        gwn_std_dev,
    ):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds
        self.iterations = iterations
        self.Cp_min = Cp_min
        self.Cp_max = Cp_max
        self.Cg_min = Cg_min
        self.Cg_max = Cg_max
        self.w_min = w_min
        self.w_max = w_max
        self.threshold = threshold
        self.function = function
        self.gwn_std_dev = gwn_std_dev
        self.swarm_best_fitness = float("inf")
        self.swarm_best_position = None

        self.particles = self.initialize_particles()

        self.swarm_position_history = []
        self.swarm_fitness_history = []

    def initialize_particles(self):
        particles = []

        for _ in range(self.num_particles):
            particles.append(
                RPSOParticle(
                    self.Cp_min,
                    self.Cp_max,
                    self.Cg_min,
                    self.Cg_max,
                    self.w_min,
                    self.w_max,
                    self.iterations,
                    self.num_dimensions,
                    self.position_bounds,
                    self.velocity_bounds,
                    self.gwn_std_dev,
                )
            )

        return particles

    def run_linear_regression(self, particle, X_test, y_test):
        return linear_regression(particle.position, X_test, y_test)

    def run_pso(self, model):
        X_train, X_test, y_train, y_test, scaler = get_fingerprinted_data()
        for iter in range(self.iterations):
            for particle in self.particles:
                fitness = self.function(particle.position, model, X_train, y_train)
                # fitness = ann_node_count_fitness(particle.position)
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position
                if fitness < self.swarm_best_fitness:
                    self.swarm_best_fitness = fitness
                    self.swarm_best_position = particle.position
            # Check if fitness threshold has been reached
            if fitness <= self.threshold:
                break
            for particle in self.particles:
                particle.update_particle_velocity(
                    self.swarm_best_position,
                )
                particle.update_position()
                particle.update_current_iteration()
            self.swarm_fitness_history.append(self.swarm_best_fitness)
            self.swarm_position_history.append(
                [particle.position for particle in self.particles]
            )
