# Run the RPSO to optimize the weights and biases of an ANN
from __future__ import division
import sys
import time
import numpy as np
import multiprocessing
from functools import partial
import pandas as pd
import tensorflow as tf
from CostFunctions import get_fingerprinted_data, get_fingerprinted_data_noisy

from RPSO import RPSO
from Statistics import save_opt_ann_rpso_stats
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
        sliced_weights = particle[:num_weights]
        sliced_biases = particle[num_weights: num_weights + num_biases]

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
        evaluation_metrics = model.evaluate(X_train, y_train, verbose=0)
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


def run_pso(
    _,
    iterations,
    position_bounds,
    velocity_bounds,
    fitness_threshold,
    num_particles,
    Cp_min,
    Cp_max,
    Cg_min,
    Cg_max,
    w_min,
    w_max,
    gwn_std_dev,
):
    swarm = RPSO(
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
        fitness_threshold,
        ann_weights_fitness_function,
        gwn_std_dev,
    )

    swarm.run_pso(model)

    return (
        swarm.swarm_best_fitness,
        swarm.swarm_best_position,
        swarm.swarm_fitness_history,
        swarm.swarm_position_history,
    )


if __name__ == "__main__":
    # ---- RSPO options ----
    position_bounds = [(-1.0, 1.0)] * num_dimensions
    velocity_bounds = [(-0.2, 0.2)] * num_dimensions
    fitness_threshold = 0.1
    num_particles = 60
    # Cp_min = 0.1
    # Cp_max = 3.5
    # Cg_min = 0.9
    # Cg_max = 3.0
    # w_min = 0.3
    # w_max = 1.7
    # gwn_std_dev = 0.15
    iterations = 100

    # Cp_min = [0.9075920273051346, 0.9307676347929889, 1.3615354778835882, 0.9738435382203143, 1.0330947698993713,
    #           1.7264053277427331, 1.8326633642093986, 1.0915875624258558, 0.9542380152637335, 1.7406233639998683]
    # Cp_max = [4.045912359854258, 3.9251262216281733, 4.770097880137207, 3.676635555917675, 4.865388843567808,
    #           4.00803616585983, 4.324617317799125, 3.6593851406380655, 4.363830992468895, 3.8414370147162007]
    # Cg_min = [1.9833746303128135, 1.560081401321889, 0.9949619527325266, 1.2389110846509612, 1.7183858758496315,
    #           1.886869077185884, 1.902583751217683, 1.8689953088579818, 1.6456264349481828, 1.4804686986513107]
    # Cg_max = [4.9473127856158765, 4.186135921035518, 4.981467044074979, 4.809778548615008, 4.66394388781309,
    #           4.967962251338514, 4.3009493606669835, 4.958097094055031, 3.9186291977382606, 4.406715239704273]
    # w_min = [0.9893630020731844, 1.2591571573471751, 1.130971796613338, 1.5875034855511618, 1.7027829195187936,
    #          1.8136156161109598, 1.5891814924897618, 0.844147903361651, 1.888401672900143, 1.9775420066956115]
    # w_max = [2.450984505185616, 2.7575309856239345, 2.7667295573752804, 2.8663164133146886, 2.326084536204405,
    #          2.6034288014254634, 2.3956330763407445, 2.515286972667997, 2.7720105251061655, 2.8681090602319212]
    # gwn_std_dev = [0.4266606441351424, 0.752351668245, 0.3282391129563975, 0.5770333808344066, 0.37917721641918667,
    #                0.6250913291480527, 0.6783262683093851, 0.32202459810792683, 0.22372449786219423, 0.651209554152169]

    Cp_min = [
        0.5255807110471932,
        0.3429381734058268,
        0.5473953596260367,
        0.5892357456424455,
        0.6079222888523057,
        0.6344813277971112,
        0.19014413636290764,
        0.2117808640159182,
        0.8455214965782043,
        0.33172269877085303
    ]
    Cp_max = [
        1.6561572229917194,
        1.6942349478931538,
        1.8117721144510008,
        3.111438073361451,
        2.488047379454802,
        3.1738084595383205,
        3.249682100682639,
        2.2498771078645934,
        3.026638675276517,
        2.791080482283482
    ]
    Cg_min = [
        0.24866719100172174,
        0.1063992296063363,
        0.6399466405615443,
        0.7789907590032242,
        0.7714898739594362,
        0.11531231543022723,
        0.35601705530224337,
        0.2657472414099825,
        0.5060031362324593,
        0.17612180788459356
    ]
    Cg_max = [
        3.401729764116632,
        2.7656770286923624,
        1.7226611789385775,
        1.9783997678963696,
        1.563962743441281,
        2.8485602766725373,
        2.6324175260968463,
        1.884195354440522,
        3.0420218358982307,
        1.845946114117474
    ]
    w_min = [
        0.3277489076256416,
        0.10323507627270086,
        0.6903954593094627,
        0.2512192466357657,
        0.31828822621491576,
        0.23769350357392033,
        0.3592722860920172,
        0.13808108256010332,
        0.5815967759592018,
        0.5758491828008468
    ]
    w_max = [
        1.090799879141887,
        1.2011811660684617,
        1.0519962824603226,
        1.4027237055762312,
        1.3946578910060003,
        1.90350611469437,
        1.245797074484059,
        0.5383041243044694,
        0.6896131233011519,
        1.8905367686610268
    ]
    gwn_std_dev = [
        0.06427594732038355,
        0.09711193898395054,
        0.18030623089238926,
        0.18926183363254914,
        0.17275962871453357,
        0.026820486300555134,
        0.08580276947147242,
        0.19679651652191496,
        0.07150807341695263,
        0.1315206414296331
    ]

    parameter_dicts = {}

    for i in range(len(Cp_min)):
        parameter_dicts[i] = {
            'Cp_min': Cp_min[i],
            'Cp_max': Cp_max[i],
            'Cg_min': Cg_min[i],
            'Cg_max': Cg_max[i],
            'w_min': w_min[i],
            'w_max': w_max[i],
            'gwn_std_dev': gwn_std_dev[i]
        }

        run_pso_partial = partial(
            run_pso,
            iterations=iterations,
            position_bounds=position_bounds,
            velocity_bounds=velocity_bounds,
            fitness_threshold=fitness_threshold,
            num_particles=num_particles,
            Cp_min=parameter_dicts[i]["Cp_min"],
            Cp_max=parameter_dicts[i]["Cp_max"],
            Cg_min=parameter_dicts[i]["Cg_min"],
            Cg_max=parameter_dicts[i]["Cg_max"],
            w_min=parameter_dicts[i]["w_min"],
            w_max=parameter_dicts[i]["w_max"],
            gwn_std_dev=parameter_dicts[i]["gwn_std_dev"],
        )

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
            swarm_position_history,
        ) = zip(*results)
        mean_best_fitness = np.mean(swarm_best_fitness)
        min_best_fitness = np.min(swarm_best_fitness)
        max_best_fitness = np.max(swarm_best_fitness)

        index_of_best_fitness = swarm_best_fitness.index(min_best_fitness)
        best_weights = swarm_best_position[index_of_best_fitness].tolist()

        sys.stdout.write(
            f"Minimum fitness for {pso_runs} runs: {min_best_fitness}. Mean: {mean_best_fitness}. Max: {max_best_fitness}"
        )

        save_opt_ann_rpso_stats(
            fitness_histories,
            "rpso",
            pso_runs,
            position_bounds,
            velocity_bounds,
            fitness_threshold,
            num_particles,
            parameter_dicts[i]["Cp_min"],
            parameter_dicts[i]["Cp_max"],
            parameter_dicts[i]["Cg_min"],
            parameter_dicts[i]["Cg_max"],
            parameter_dicts[i]["w_min"],
            parameter_dicts[i]["w_max"],
            parameter_dicts[i]["gwn_std_dev"],
            iterations,
            elapsed_time,
            min_best_fitness,
            mean_best_fitness,
            max_best_fitness,
            best_weights,
        )
