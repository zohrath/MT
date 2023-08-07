from CostFunctions import (
    ann_cost_function,
    ann_node_count_fitness,
    get_regression_data,
    linear_regression,
)
from Particle import Particle


class GBest_PSO:
    def __init__(
        self,
        iterations,
        num_particles,
        num_dimensions,
        position_bounds,
        velocity_bounds,
        inertia,
        c1,
        c2,
        threshold,
        function,
    ):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds
        self.iterations = iterations
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.threshold = threshold
        self.function = function
        self.swarm_best_fitness = float("inf")
        self.swarm_best_position = None

        self.particles = self.initialize_particles()

        self.swarm_position_history = []
        self.swarm_fitness_history = []

    def run_linear_regression(self, particle, X_test, y_test):
        return linear_regression(particle.position, X_test, y_test)

    def run_ann_regression(self, particle):
        return ann_cost_function(particle)

    def initialize_particles(self):
        particles = []

        for _ in range(self.num_particles):
            particles.append(
                Particle(
                    self.num_dimensions, self.position_bounds, self.velocity_bounds
                )
            )

        return particles

    def run_pso(self):
        for _ in range(self.iterations):
            for particle in self.particles:
                fitness = ann_node_count_fitness(particle.position)
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position
                if fitness < self.swarm_best_fitness:
                    self.swarm_best_fitness = fitness
                    self.swarm_best_position = particle.position
            if fitness <= self.threshold:
                break
            for particle in self.particles:
                particle.velocity = particle.update_particle_velocity(
                    self.inertia,
                    self.c1,
                    particle.best_position,
                    particle.position,
                    particle.velocity,
                    self.c2,
                    self.swarm_best_position,
                )
                particle.position = particle.update_position(
                    particle.position, particle.velocity
                )
            self.swarm_fitness_history.append(self.swarm_best_fitness)
            swarm_positions = [particle.position for particle in self.particles]
            self.swarm_position_history.append(swarm_positions)
