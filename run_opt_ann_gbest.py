# Run the GBest to optimize the weights and biases of an ANN
from __future__ import division
import sys
import time
import numpy as np
import multiprocessing
from functools import partial
import pandas as pd
import tensorflow as tf
from CostFunctions import get_fingerprinted_data, get_fingerprinted_data_noisy

from GBestPSO import GBest_PSO
from Statistics import save_opt_ann_gbest_stats, save_opt_ann_rpso_stats
from pso_options import create_model


X_train, _, y_train, _, _ = get_fingerprinted_data()
model, num_dimensions = create_model()


def ann_weights_fitness_function(particle, model):
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        num_weights = weights.size
        num_biases = biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = np.array(particle[:num_weights])
        sliced_biases = np.array(
            particle[num_weights: num_weights + num_biases])

        # Update the continuous_values array for the next iteration
        particle = particle[num_weights + num_biases:]

        # Set the sliced weights and biases in the layer
        layer.set_weights(
            [sliced_weights.reshape(weights.shape),
             sliced_biases.reshape(biases.shape)]
        )
    try:
        model.compile(optimizer="adam", loss="mse")

        # Evaluate the model and get the evaluation metrics
        evaluation_metrics = model.evaluate(X_train, y_train, verbose=1)
        rmse = np.sqrt(evaluation_metrics)

        return rmse
    except tf.errors.InvalidArgumentError as e:
        # Handle the specific exception here
        print("Caught an InvalidArgumentError:", e)
        # You can choose to return a specific value or take other actions
        return float("inf")  # For example, return infinity in case of an error
    except tf.errors.OpError as e:
        # Handle TensorFlow-specific errors here
        print(f"TensorFlow error: {e}")
        return float("inf")  # For example, return infinity in case of an error
    except Exception as e:
        # Handle other exceptions here
        print(f"An error occurred: {e}")
        return float("inf")  # For example, return infinity in case of an error


def run_pso(_, iterations, position_bounds, velocity_bounds, fitness_threshold,
            num_particles, c1, c2, inertia):

    swarm = GBest_PSO(
        iterations,
        num_particles,
        num_dimensions,
        position_bounds,
        velocity_bounds,
        inertia,
        c1,
        c2,
        fitness_threshold,
        ann_weights_fitness_function

    )

    swarm.run_pso(model)

    return (
        swarm.swarm_best_fitness,
        swarm.swarm_best_position,
        swarm.swarm_fitness_history,
        swarm.swarm_position_history,
    )


if __name__ == "__main__":
    # ---- GBest options ----
    position_bounds = [(-4.0, 4.0)] * num_dimensions
    velocity_bounds = [(-0.2, 0.2)] * num_dimensions
    fitness_threshold = 0.1
    num_particles = 30
    # c1 = 1.8663
    # c2 = 1.94016
    # w = 0.8
    iterations = 200

    velocity_vals = [0.25075366125782467, 0.6164545330679831, 0.2610572949023807, 0.43649972014578836, 0.5081047551068012, 0.07424351354149411, 0.5006944819730034, 0.732732633737257, 0.041396298498333026,
                     0.1729377281516451, 0.5669769898875903, 0.49305341516053525, 0.1856638700653086, 0.6224193478649852, 0.274297540723357, 0.05956926693752715, 0.07952712854296815, 0.3771574773872123, 0.29493499477033314, 0.272718360996686, 0.6406467824612986, 0.7592455059645575, 0.6894299055029663, 0.4555071194724097, 0.6345490629278445, 0.5544357465223577, 0.7428206631023622]
    velocity_vals = [[(-x, x)] * num_dimensions for x in velocity_vals]

    c1_vals = [1.734045256050708, 1.6079799885994959, 1.3256315421906024, 1.5117542127638601, 1.877473873187585, 1.6754234822702827, 1.7216203586996048, 1.3616323956440397, 1.7613058856019097,
               1.466514367402177, 1.5105965239143222, 1.6296385364612984, 1.8807163428922178, 1.5129867435424915, 2.0294183770646574, 1.6245724425534993, 1.9831108813824951, 1.636133828972696, 1.838157282203278, 1.6315151315639669, 1.991963104937526, 1.390713429003613, 1.6866504321197457, 2.0529055193944528, 1.6621239509358927, 1.5198558606634704, 1.949166868037568]

    c2_vals = [1.8504718566095906, 1.5409169390452369, 1.4262694391334971, 1.7682266552684311, 1.3053866307839765, 1.702605569563214, 1.9834509058367615, 1.3909940058739236, 2.068234609923147, 1.5286122569544012,
               1.5701442562464514, 1.3713643283251733, 1.8701589476946519, 1.595721678155064, 1.5075641539349662, 1.6443657519866492, 1.4770729650324375, 1.3578071839621406, 1.9902389736496646, 1.5504334338133428, 1.971153803550881, 1.7519695245111768, 1.395563736132512, 1.9136518926601318, 1.8757981232291019, 1.4889156092598235, 1.3471798100598997, 1.6822940574168537]

    w_vals = [0.8350333869730729, 0.9163769619943374, 0.7603190852160345, 0.9865080743334982, 0.6155985163793721, 0.8499100991796464, 0.8576214262284811, 0.7118820808136387, 0.7220059119255825,
              0.6175914726081042, 0.9982235819756665, 0.8575383607692455, 0.7891131592713108, 0.848606477841146, 0.7941257602645067, 0.8128767777675963, 0.8905413220986471, 0.6491614619110713, 0.9986677202850651, 0.8148193473345878, 0.7521713890114037, 0.8169604141344282, 0.8061807417222006, 0.6087598110371877, 0.7066381636883836, 0.6464097810571167, 0.613171074716452]

    parameter_dicts = {}

    for i in range(len(c1_vals)):
        parameter_dicts[i] = {
            'c1': c1_vals[i],
            'c2': c2_vals[i],
            'w': w_vals[i],
            'velocity_bounds': velocity_vals[i],
        }

        run_pso_partial = partial(run_pso,
                                  iterations=iterations,
                                  position_bounds=position_bounds,
                                  velocity_bounds=parameter_dicts[i]["velocity_bounds"],
                                  fitness_threshold=fitness_threshold,
                                  num_particles=num_particles,
                                  c1=parameter_dicts[i]["c1"],
                                  c2=parameter_dicts[i]["c2"],
                                  inertia=parameter_dicts[i]["w"])

        pso_runs = multiprocessing.cpu_count() - 1

        start_time = time.time()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            results = pool.map(run_pso_partial, range(pso_runs))
        end_time = time.time()
        elapsed_time = end_time - start_time

        fitness_histories = [result[2] for result in results]
        (
            swarm_best_fitness,
            swarm_best_position,
            swarm_fitness_history,
            swarm_position_history

        ) = zip(*results)
        mean_best_fitness = np.mean(swarm_best_fitness)
        min_best_fitness = np.min(swarm_best_fitness)
        max_best_fitness = np.max(swarm_best_fitness)
        index_of_best_fitness = swarm_best_fitness.index(min_best_fitness)
        best_weights = swarm_best_position[index_of_best_fitness].tolist()

        sys.stdout.write(
            f"Minimum fitness for {pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}"
        )

        save_opt_ann_gbest_stats(fitness_histories, "gbest", pso_runs, position_bounds, parameter_dicts[i]["velocity_bounds"],
                                 fitness_threshold, num_particles, parameter_dicts[i][
            "c1"], parameter_dicts[i]["c2"], parameter_dicts[i]["w"], iterations,
            elapsed_time, min_best_fitness, mean_best_fitness,
            max_best_fitness, best_weights)
