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
    position_bounds = [(-1.0, 1.0)] * num_dimensions
    velocity_bounds = [(-0.2, 0.2)] * num_dimensions
    fitness_threshold = 0.1
    num_particles = 60
    # c1 = 1.8663
    # c2 = 1.94016
    # w = 0.8
    iterations = 100

    c1 = [1.8179607226084602, 1.9563483854542694, 1.7913651248349782, 1.8796099331224823, 1.8317073719754766, 1.9159836314355005, 1.9207647563005337, 1.9732781107769621, 1.672593151100076, 1.6169649628904348, 1.5477600124622246, 1.794670638876168, 1.8026742066131076, 1.973889449185508, 1.5351795871154728, 1.8616412187744347, 1.9473297272567165, 1.5188772339928978, 1.9975051112849544, 1.8383526676197652, 1.6945668002687866, 1.8797165467290893, 1.81229612798822,
          1.9129599457505067, 1.5080172490964587, 1.7354229852577185, 1.9747936953505976, 1.5693305277757865, 1.7109326631560333, 1.523536172598751, 1.5172024370739503, 1.8388117039645613, 1.9743064168402271, 1.560060043785444, 1.6730440831135982, 1.5304371786155884, 1.9048512958176338, 1.9076859160817836, 1.6607188940237643, 1.7271379509033673, 1.7142020685150658, 1.6130044396019518, 1.8049709262740459, 1.604427442941445, 1.6147315167453942, 1.7099156381847371, 1.8224008879600755, 1.8555830227916708, 1.553482997147442, 1.7296590161568748, 1.5458587774032404, 1.8150053533341957, 1.8463147045655655, 1.5465173580277225, 1.9414903557835803, 1.6711483088467745, 1.586234337815111, 1.7908488099426145, 1.9666674949610137, 1.896643533493107, 1.6743291868940888, 1.7053242740543446, 1.689535496949767, 1.8873079437263778, 1.9814512903930541, 1.885696860276864, 1.8690016031943377, 1.5267663204909014, 1.9152624019881186, 1.7129161273212437, 1.6022795848166946, 1.7246428749634226, 1.519114261005193]
    c2 = [1.8475876850787523, 1.7100783378671915, 1.5117907701162863, 1.620254568139488, 1.975641924568552, 1.690875394709206, 1.9558434348716534, 1.9782629805569436, 1.6850047878385623, 1.7362165923391675, 1.5641031982028832, 1.877757640683239, 1.9556866055280266, 1.5198653256425823, 1.8110173531139873, 1.5169211428807952, 1.835432834093627, 1.6298859382529238, 1.78088537331477, 1.6866615806916148, 1.6251701257328013, 1.5351150831378528, 1.5452826697474058,
          1.9963819115154817, 1.5182696603031853, 1.802549286606462, 1.982801429680273, 1.7713419221687863, 1.8358519191824323, 1.7162984317417578, 1.9743366217365876, 1.5382798872761934, 1.5975011400507206, 1.5195138575953377, 1.994471537761728, 1.5742176681742477, 1.5395988957771367, 1.8839994027752673, 1.9481954365317287, 1.9079694477525504, 1.5705609234894329, 1.5039064098038364, 1.912676111308487, 1.6049567068987167, 1.9863671680647113, 1.6422179568945023, 1.525635418317604, 1.924968354190334, 1.6996420753885026, 1.558377980767998, 1.8456115040503704, 1.609385850680988, 1.7798709365839753, 1.628949352349851, 1.746886820473248, 1.5463453550008905, 1.8452731266524887, 1.8457945446939432, 1.6812735079341152, 1.6500146276436445, 1.5845471257269543, 1.8929177108480069, 1.5376271017814405, 1.8456464397379404, 1.6190269878583707, 1.4992048020089144, 1.5677144541355688, 1.5052260869819643, 1.9793891925445073, 1.5655739813436782, 1.7455293116565946, 1.952595125899025, 1.7800473331113036]
    inertia = [0.7787895922215501, 0.7439200185414532, 0.7374185612250636, 0.788327903106991, 0.7318194347330489, 0.7968309015670569, 0.7799305486936005, 0.7886810487029893, 0.7488155765774989, 0.737091280341245, 0.7361396919717981, 0.7439370990523683, 0.7957117600574708, 0.7682933085792577, 0.7853673507481765, 0.7800999073686676, 0.7658995262084219, 0.7351682314289087, 0.7446339496513322, 0.768745408383032, 0.7915757416787457, 0.7809242242216065, 0.7328356584395649,
               0.7395840190710576, 0.7869635143399903, 0.7548647509189881, 0.7802082549112512, 0.7871438338465129, 0.7433564074686858, 0.7441595072626042, 0.7976222463807583, 0.7694291202673501, 0.7775611833504251, 0.7662536953763286, 0.7513269878530942, 0.7903801592531374, 0.7452428321788749, 0.7705175198925256, 0.7496731941029555, 0.7323535909073015, 0.7493452619356409, 0.7681649264691509, 0.7705343047079479, 0.7877625883788891, 0.7487061716910874, 0.7852727965482477, 0.7978385596340314, 0.7431418738237237, 0.7426983305095671, 0.7501236099228585, 0.7410388932902342, 0.7481430221167498, 0.7963934078856498, 0.7652103295223124, 0.777914717694808, 0.7541027292737942, 0.7838101736058507, 0.798304808465211, 0.7996238452702458, 0.7397512952026561, 0.7464995114671552, 0.769908498102603, 0.7776657399642275, 0.7737308649535324, 0.7571449826110022, 0.7651027992527647, 0.7768551033586764, 0.7699732691298232, 0.7591302396008563, 0.75221343551517, 0.7680715235067511, 0.7958849105369838, 0.7643122514280672]
    parameter_dicts = {}

    for i in range(len(c1)):
        parameter_dicts[i] = {
            'c1': c1[i],
            'c2': c2[i],
            'inertia': inertia[i],
        }
        run_pso_partial = partial(run_pso,
                                  iterations=iterations,
                                  position_bounds=position_bounds,
                                  velocity_bounds=velocity_bounds,
                                  fitness_threshold=fitness_threshold,
                                  num_particles=num_particles,
                                  c1=parameter_dicts[i]["c1"],
                                  c2=parameter_dicts[i]["c2"],
                                  inertia=parameter_dicts[i]["inertia"])

        pso_runs = multiprocessing.cpu_count() - 1

        start_time = time.time()
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count() - 1
        ) as pool:
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

        save_opt_ann_gbest_stats(fitness_histories, "gbest", pso_runs, position_bounds, velocity_bounds,
                                 fitness_threshold, num_particles, parameter_dicts[i]["c1"],
                                 parameter_dicts[i]["c2"], parameter_dicts[i]["inertia"], iterations,
                                 elapsed_time, min_best_fitness, mean_best_fitness, max_best_fitness, best_weights)
