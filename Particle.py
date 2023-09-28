import numpy as np


class Particle:
    def __init__(self, num_dimensions, position_bounds, velocity_bounds):
        self.num_dimensions = num_dimensions
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds

        self.position = self.initialize_particle_position()
        self.velocity = self.initialize_particle_velocity()
        self.best_position = self.position
        self.best_fitness = float("inf")

        self.position_history = []

    def initialize_particle_position(self):
        return [np.random.uniform(self.position_bounds[i][0], self.position_bounds[i][1]) for i in range(self.num_dimensions)]

    def initialize_particle_velocity(self):
        return [np.random.uniform(self.velocity_bounds[i][0], self.velocity_bounds[i][1]) for i in range(self.num_dimensions)]

    def get_social_parameter(self, c2, swarm_best_position, particle_current_position):
        r2 = np.random.rand()
        return c2 * r2 * (np.array(swarm_best_position) - np.array(particle_current_position))

    def get_cognitive_parameter(self, c1, particle_best_position, particle_current_position):
        r1 = np.random.rand()
        return c1 * r1 * (np.array(particle_best_position) - np.array(particle_current_position))

    def get_inertia_velocity_part(self, inertia, particle_current_velocity):
        inertia_param = [inertia * v for v in particle_current_velocity]
        return inertia_param

    def update_particle_velocity(
        self,
        inertia,
        c1,
        particle_best_position,
        particle_current_position,
        particle_current_velocity,
        c2,
        swarm_best_position,
    ):
        inertia_param = self.get_inertia_velocity_part(
            inertia, particle_current_velocity
        )
        cognitive_param = self.get_cognitive_parameter(
            c1, particle_best_position, particle_current_position
        )
        social_param = self.get_social_parameter(
            c2, swarm_best_position, particle_current_position
        )

        updated_velocity = inertia_param + cognitive_param + social_param

        for i in range(self.num_dimensions):
            updated_velocity[i] = np.clip(
                updated_velocity[i], self.velocity_bounds[i][0], self.velocity_bounds[i][1])

        return updated_velocity

    def update_position(self, current_position, updated_particle_velocity):
        new_position = current_position + updated_particle_velocity

        for i in range(self.num_dimensions):
            new_position[i] = np.clip(
                new_position[i], self.position_bounds[i][0], self.position_bounds[i][1])

        self.position_history.append(new_position)

        return new_position
