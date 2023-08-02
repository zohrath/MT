from __future__ import division
import sys
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing
from functools import partial
import argparse
from GBestPSO import GBest_PSO
from RPSO import RPSO

PSO_TYPE = "rpso"


class Main:
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
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        threshold,
    ):
        self.iterations = iterations
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.Cp_min = Cp_min
        self.Cp_max = Cp_max
        self.Cg_min = Cg_min
        self.Cg_max = Cg_max
        self.w_min = w_min
        self.w_max = w_max
        self.threshold = threshold

    def run_pso(self, pso_type):
        if pso_type == "gbest":
            swarm = GBest_PSO(
                self.iterations,
                num_particles,
                num_dimensions,
                position_bounds,
                velocity_bounds,
                inertia,
                c1,
                c2,
            )
            swarm.run_pso()

            return (
                swarm.swarm_best_fitness,
                swarm.swarm_best_position,
                swarm.swarm_fitness_history,
            )
        elif pso_type == "rpso":
            swarm = RPSO(
                iterations,
                num_particles,
                num_dimensions,
                position_bounds,
                velocity_bounds,
                self.Cp_min,
                self.Cp_max,
                self.Cg_min,
                self.Cg_max,
                self.w_min,
                self.w_max,
                self.threshold,
            )
            swarm.run_pso()

            return (
                swarm.swarm_best_fitness,
                swarm.swarm_best_position,
                swarm.swarm_fitness_history,
            )


iterations = 10000
num_particles = 30
num_dimensions = 30
position_bounds = (-100.0, 100.0)
velocity_bounds = (-40.0, 40.0)
inertia = 0.5
c1 = 0.5
c2 = 0.5
num_runs = 50
fitness_history = []
Cp_min = 0.5
Cp_max = 2.5
Cg_min = 0.5
Cg_max = 2.5
w_min = 0.4
w_max = 0.9
threshold = 0.01
fifty_runs_res = []


def run_pso_threaded(_, pso_type):
    swarm = Main(
        iterations,
        num_particles,
        num_dimensions,
        position_bounds,
        velocity_bounds,
        inertia,
        c1,
        c2,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        threshold,
    )
    swarm_best_fitness, swarm_best_position, swarm_fitness_history = swarm.run_pso(
        pso_type
    )

    return (swarm_best_fitness, swarm_best_position, swarm_fitness_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PSO algorithm with single or threaded execution."
    )
    parser.add_argument(
        "mode",
        choices=["single", "threaded", "many_single"],
        help="Select execution mode: single or threaded",
    )
    args = parser.parse_args()

    main = Main(
        iterations,
        num_particles,
        num_dimensions,
        position_bounds,
        velocity_bounds,
        inertia,
        c1,
        c2,
        Cp_min,
        Cp_max,
        Cg_min,
        Cg_max,
        w_min,
        w_max,
        threshold,
    )

    if args.mode == "single":
        swarm_best_fitness, swarm_best_position, swarm_fitness_history = main.run_pso(
            PSO_TYPE
        )
        sys.stdout.write("Best fitness: {}\n".format(swarm_best_fitness))
        fitness_history = swarm_fitness_history

        iterations = list(range(1, len(fitness_history) + 1))

        # Plot the fitness history
        plt.plot(iterations, fitness_history, marker="o", linestyle="-", color="b")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.title("Fitness History of PSO")
        plt.grid(True)
        plt.show()

    elif args.mode == "threaded":
        num_cores = multiprocessing.cpu_count()
        run_pso_partial = partial(run_pso_threaded, pso_type=PSO_TYPE)

        with multiprocessing.Pool(processes=num_cores - 1) as pool:
            results = pool.map(run_pso_partial, range(num_cores - 1))

        swarm_best_fitness, swarm_best_position, swarm_fitness_history = zip(*results)
        sys.stdout.write(
            "Fitness values and mean: {} {}".format(
                swarm_best_fitness, np.mean(swarm_best_fitness)
            )
        )
    elif args.mode == "many_single":
        for _ in range(num_runs):
            (
                swarm_best_fitness,
                swarm_best_position,
                swarm_fitness_history,
            ) = main.run_pso(PSO_TYPE)
            fifty_runs_res.append(swarm_best_fitness)
            sys.stdout.write(
                "Fitness values and mean: {} {}".format(
                    swarm_best_fitness, np.mean(swarm_best_fitness)
                )
            )
    else:
        print(
            "Invalid mode. Please choose either 'single' or 'threaded' as the execution mode."
        )
