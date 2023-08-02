import numpy as np
from CostFunctions import (
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


class RPSOParticle(Particle):
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
    ):
        super().__init__(num_dimensions, position_bounds, velocity_bounds)

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

    def get_social_velocity_part(
        self, Cg, swarm_best_position, particle_current_position
    ):
        r2 = np.random.rand()
        d2 = get_GWN()
        return r2 * (self.Cg + d2) * (swarm_best_position - particle_current_position)

    def get_cognitive_velocity_part(
        self, Cp, particle_best_position, particle_current_position
    ):
        r1 = np.random.rand()
        d1 = get_GWN()
        return (
            r1 * (self.Cp + d1) * (particle_best_position - particle_current_position)
        )

    def update_particle_velocity(
        self,
        particle_best_position,
        particle_current_position,
        swarm_best_position,
    ):
        inertia_param = self.get_inertia_velocity_part(
            self.get_inertia_weight_parameter(), particle_current_position
        )
        cognitive_param = self.get_cognitive_velocity_part(
            self.get_cognitive_parameter(),
            particle_best_position,
            particle_current_position,
        )
        social_param = self.get_social_velocity_part(
            self.get_social_parameter(), swarm_best_position, particle_current_position
        )

        updated_velocity = inertia_param + cognitive_param + social_param

        # updated_velocity = np.clip(
        #     updated_velocity, self.velocity_bounds[0], self.velocity_bounds[1]
        # )

        return updated_velocity

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
                )
            )

        return particles

    def run_linear_regression(self, particle, X_test, y_test):
        return linear_regression(particle.position, X_test, y_test)

    def run_pso(self):
        # _, X_test, _, y_test = get_regression_data()
        for _ in range(self.iterations):
            for particle in self.particles:
                fitness = sphere(particle.position)
                # fitness = self.run_linear_regression(particle, X_test, y_test)
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
                particle.velocity = particle.update_particle_velocity(
                    particle.best_position,
                    particle.position,
                    self.swarm_best_position,
                )
                particle.position = particle.update_position(
                    particle.position, particle.velocity
                )
                particle.update_current_iteration()
            self.swarm_fitness_history.append(self.swarm_best_fitness)
            swarm_positions = [particle.position for particle in self.particles]
            self.swarm_position_history.append(swarm_positions)
