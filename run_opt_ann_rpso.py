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

    Cp_min = [0.125858687249717, 0.29241178651541544, 0.12514599174683078, 0.4607639669349878, 0.35044728202058395, 0.37835253496510224, 0.12139058694943015, 0.333114018853061, 0.38881746756271773, 0.344857267216814, 0.31787443754570444,
              0.17119537861166811, 0.18594913651009498, 0.13272276452153917, 0.332488628222665, 0.46802177746494844, 0.11608683816365609, 0.4026130862119216, 0.2760541557854286, 0.45553018130129297, 0.1567308385493953, 0.12878599498858076]
    Cp_max = [3.41268817702453, 3.4982243402969315, 3.045691730178825, 2.9139539348165964, 2.684241482811011, 2.9753117412983006, 2.5431585519770494, 3.4745064468232414, 3.436697081255425, 2.8365695366743995, 3.2897594879237824,
              2.578446022412579, 2.5576281205262754, 2.6243708820312235, 2.7290287476474187, 2.8948759164400215, 2.5595515202809724, 3.128839235361424, 3.1965179949172713, 2.678443309985643, 3.103197321983843, 2.9201982274798315]
    Cg_min = [0.8026644473731985, 0.7055534767731695, 0.8566040560130685, 0.7072097970962644, 0.5250595481030034, 0.7980640358918378, 0.7532147117198675, 0.8473235512670539, 0.5585447499608684, 0.6842115473997475, 0.6377781805658262,
              0.7656272713163512, 0.5797778475140417, 0.6494166558557466, 0.5162435875509918, 0.5245598900248964, 0.5045471366653452, 0.6599833005054556, 0.6803491218119077, 0.6427952063367574, 0.8430322990979453, 0.8918166101353888]
    Cg_max = [2.6039595022272417, 2.8075031145328686, 3.190203542867552, 3.3669396081349827, 3.119055056725933, 3.4168137609713813, 3.2507571328901785, 2.694483655592094, 2.86322335332677, 2.644796921908304, 3.229083502241464,
              2.8673260332191246, 3.422227344083011, 2.9651635029545003, 2.5943834822670437, 2.6662244532512474, 2.686828854517179, 2.975773845136936, 3.297000200543369, 3.266382233098596, 2.6324414938309912, 2.9029340138166013]
    w_min = [0.3502339177183231, 0.3814228548769733, 0.37974698440222776, 0.348290060411828, 0.34110092602545805, 0.3217052111302615, 0.3195283311896024, 0.34240243378145885, 0.30037228910952907, 0.33270079366202915, 0.359050657774262,
             0.3838821865686707, 0.3862168678439895, 0.3897028482846384, 0.32766939238823856, 0.324442381695725, 0.3567028680328851, 0.3934752807560073, 0.3089047866258222, 0.3596323656730636, 0.37533945258353535, 0.33109400113337106]
    w_max = [1.4991547386369657, 1.3854186693832198, 0.9974304892719259, 1.5953920778286852, 1.02023547448896, 1.5658004593793926, 1.3959481962940614, 1.1969004949376743, 1.5935296924549205, 1.6270871110704053, 1.5430944316470852,
             1.5604180382049941, 1.497719505454414, 1.3263534418150051, 1.3723326412501167, 0.9637420080880907, 1.1779992333054716, 1.6577736544893265, 0.9066847482027525, 0.933473945438753, 1.2399964933205223, 0.9957539804613164]
    gwn_std_dev = [0.08940189164426787, 0.07828252440246258, 0.1374085618416958, 0.13476349688279338, 0.07068592153174508, 0.09346168484877235, 0.11749061888723815, 0.09118604158446081, 0.08494852593102714, 0.12567115841096352, 0.12737671215626142,
                   0.13563064448702972, 0.142555383287939, 0.08310016847166565, 0.144557861218378, 0.08779099349974938, 0.09078169976454135, 0.10639840668636942, 0.11374665474488843, 0.0983525959436938, 0.11285708330597363, 0.08160556016613567]

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
