# Run the GBest to optimize the weights and biases of an ANN
from __future__ import division
import json
import os
import sys
import time
import numpy as np
import multiprocessing
from functools import partial
import pandas as pd
import tensorflow as tf
from CostFunctions import get_fingerprinted_random_points_calm_data, get_fingerprinted_random_points_noisy_data, get_fingerprinted_data_noisy_as_verification_set, get_fingerprinted_data_as_verification_set

from GBestPSO import GBest_PSO
from Statistics import save_opt_ann_gbest_stats, save_opt_ann_rpso_stats
from pso_options import create_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from collections import defaultdict


def find_combined_best(json_file_path, top=5):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    stats = data.get('verification_stats', [])

    if not stats:
        return []

    sorted_stats = sorted(stats, key=lambda x: (x.get('MedAE', float('inf')), x.get('Max Error', float('inf'))))


    return sorted_stats[:top]


def find_combined_worst(json_file_path, bottom=5):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    stats = data.get('verification_stats', [])

    if not stats:
        return []

    sorted_stats = sorted(stats, key=lambda x: x.get('MedAE', float('inf')))
    sorted_stats.reverse()

    return sorted_stats[:bottom]


X_test, y_test = get_fingerprinted_data_as_verification_set()


def get_ann_stats(weights):
    model, _ = create_model()
    for layer in model.layers:
        original_weights = layer.get_weights()[0]
        original_biases = layer.get_weights()[1]
        num_weights = original_weights.size
        num_biases = original_biases.size

        # Slice off values from the continuous_values array for weights and biases
        sliced_weights = np.array(weights[:num_weights])
        sliced_biases = np.array(
            weights[num_weights: num_weights + num_biases])

        # Update the continuous_values array for the next iteration
        weights = weights[num_weights + num_biases:]

        # Set the sliced weights and biases in the layer
        layer.set_weights(
            [
                sliced_weights.reshape(original_weights.shape),
                sliced_biases.reshape(original_biases.shape),
            ]
        )
    try:
        model.compile(optimizer="adam", loss="mse")

        # Use the model to predict coordinates for the entire verification set
        predicted_locations = model.predict(X_test)

        # Calculate absolute differences between actual and predicted locations
        absolute_errors = np.abs(y_test - predicted_locations)

        # Calculate metrics
        mae = np.mean(absolute_errors)
        mse = np.mean(absolute_errors**2)
        rmse = np.sqrt(mse)
        medae = np.median(absolute_errors)

        # Find the minimum and maximum error
        min_error = np.min(absolute_errors)
        max_error = np.max(absolute_errors)

        return mae, mse, rmse, medae, min_error, max_error, predicted_locations, absolute_errors

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


def flatten_list(input_list):
    output_list = []
    for item in input_list:
        if isinstance(item, list):
            output_list.extend(flatten_list(item))
        else:
            output_list.append(item)
    return output_list

def collect_verification_set_stats():
     # --------------- GBest Calm Data Set -----------------
    opt_ann_gbest_calm_1 = 'opt_ann_gbest_stats/calm_data_set/c120w08/stats_2023-09-24_16-23-32.json'
    opt_ann_gbest_calm_2 = 'opt_ann_gbest_stats/calm_data_set/c1c2149445w0729/stats_2023-09-24_16-39-20.json'
    opt_ann_gbest_calm_3 = 'opt_ann_gbest_stats/calm_data_set/extra_long_single_run/stats_2023-09-24_17-34-25.json'
    opt_ann_gbest_calm_4 = 'opt_ann_gbest_stats/calm_data_set/random_search_params/stats_2023-09-24_17-06-41.json'

    # --------------- GBest Noisy Data Set -----------------
    opt_ann_gbest_noisy_1 = 'opt_ann_gbest_stats/noisy_data_set/c120w08/stats_2023-09-29_13-56-17.json'
    opt_ann_gbest_noisy_2 = 'opt_ann_gbest_stats/noisy_data_set/c1c2149445w0729/stats_2023-09-29_15-00-07.json'
    opt_ann_gbest_noisy_3 = 'opt_ann_gbest_stats/noisy_data_set/extra_long_single_run/stats_2023-09-29_15-12-09.json'
    opt_ann_gbest_noisy_4 = 'opt_ann_gbest_stats/noisy_data_set/random_search_params/stats_2023-09-29_13-37-28.json'

    # --------------- RPSO Calm Data Set -------------------
    opt_ann_rpso_calm_1 = 'opt_ann_rpso_stats/calm_data_set/default_rpso_params_res/stats_2023-09-24_11-21-32.json'
    opt_ann_rpso_calm_2 = 'opt_ann_rpso_stats/calm_data_set/extra_long_single_run/grid_search_params/stats_2023-09-24_14-50-56.json'
    opt_ann_rpso_calm_3 = 'opt_ann_rpso_stats/calm_data_set/extra_long_single_run/random_search_params/stats_2023-09-24_15-06-05.json'
    opt_ann_rpso_calm_4 = 'opt_ann_rpso_stats/calm_data_set/extra_long_single_run/standard_params/stats_2023-09-24_14-26-42.json'
    opt_ann_rpso_calm_5 = 'opt_ann_rpso_stats/calm_data_set/gwn_random_search_res/stats_2023-09-24_04-58-52.json'
    opt_ann_rpso_calm_6 = 'opt_ann_rpso_stats/calm_data_set/no_gwn_val_res/stats_2023-09-24_10-35-22.json'
    opt_ann_rpso_calm_7 = 'opt_ann_rpso_stats/calm_data_set/no_gwn_val_res/stats_2023-09-24_11-43-17.json'

    # --------------- RPSO Noisy Data Set ------------------
    opt_ann_rpso_noisy_1 = 'opt_ann_rpso_stats/noisy_data_set/default_rpso_params/stats_2023-09-29_16-04-52.json'
    opt_ann_rpso_noisy_2 = 'opt_ann_rpso_stats/noisy_data_set/extra_long_single_run/default_params/stats_2023-09-29_16-47-48.json'
    opt_ann_rpso_noisy_3 = 'opt_ann_rpso_stats/noisy_data_set/grid_search_params/stats_2023-09-30_03-22-36.json'
    opt_ann_rpso_noisy_4 = 'opt_ann_rpso_stats/noisy_data_set/gwn_random_search/stats_2023-09-29_15-49-39.json'
    opt_ann_rpso_noisy_5 = 'opt_ann_rpso_stats/noisy_data_set/no_gwn_val/stats_2023-09-29_16-34-34.json'
    opt_ann_rpso_noisy_6 = 'opt_ann_rpso_stats/noisy_data_set/no_gwn_val/stats_2023-09-29_17-05-42.json'

    # --------------- Adam With GBest Calm Data Set -------------------
    opt_adam_gbest_calm_1 = 'opt_adam_params_with_gbest_stats/calm_data_set/500_iterations_during_training/c149445w0729/optimize_ann_optimizer_params_2023-10-12_03-03-37.json'
    opt_adam_gbest_calm_2 = 'opt_adam_params_with_gbest_stats/calm_data_set/500_iterations_during_training/c20w08/optimize_ann_optimizer_params_2023-10-12_01-42-37.json'
    opt_adam_gbest_calm_3 = 'opt_adam_params_with_gbest_stats/calm_data_set/500_iterations_during_training/random_search_params/optimize_ann_optimizer_params_2023-10-12_04-29-53.json'

    # --------------- Adam With GBest Noisy Data Set -------------------
    opt_adam_gbest_noisy_1 = 'opt_adam_params_with_gbest_stats/noisy_data_set/500_iterations_during_training/c149445w0729/optimize_ann_optimizer_params_2023-10-12_12-23-12.json'
    opt_adam_gbest_noisy_2 = 'opt_adam_params_with_gbest_stats/noisy_data_set/500_iterations_during_training/c20w08/optimize_ann_optimizer_params_2023-10-12_10-49-48.json'
    opt_adam_gbest_noisy_3 = 'opt_adam_params_with_gbest_stats/noisy_data_set/500_iterations_during_training/random_search_params/optimize_ann_optimizer_params_2023-10-12_13-57-08.json'

    # --------------- Adam With RPSO Calm Data Set -------------------
    opt_adam_rpso_calm_1 = 'opt_adam_params_with_rpso_stats/calm_data_set/500_iterations_during_training/default_params/optimize_ann_optimizer_params_2023-10-13_00-53-20.json'
    opt_adam_rpso_calm_2 = 'opt_adam_params_with_rpso_stats/calm_data_set/500_iterations_during_training/grid_search_params/optimize_ann_optimizer_params_2023-10-12_23-35-01.json'
    opt_adam_rpso_calm_3 = 'opt_adam_params_with_rpso_stats/calm_data_set/500_iterations_during_training/no_gwn_val/optimize_ann_optimizer_params_2023-10-13_10-58-25.json'
    opt_adam_rpso_calm_4 = 'opt_adam_params_with_rpso_stats/calm_data_set/500_iterations_during_training/random_search_params/optimize_ann_optimizer_params_2023-10-13_02-09-15.json'

    # --------------- Adam With RPSO Noisy Data Set -------------------
    opt_adam_rpso_noisy_1 = 'opt_adam_params_with_rpso_stats/noisy_data_set/gridSearch_params/optimize_ann_optimizer_params_2023-10-12_15-48-52.json'
    opt_adam_rpso_noisy_2 = 'opt_adam_params_with_rpso_stats/noisy_data_set/no_gwn_val/optimize_ann_optimizer_params_2023-10-12_20-06-39.json'
    opt_adam_rpso_noisy_3 = 'opt_adam_params_with_rpso_stats/noisy_data_set/random_search_params/optimize_ann_optimizer_params_2023-10-12_18-44-17.json'
    opt_adam_rpso_noisy_4 = 'opt_adam_params_with_rpso_stats/noisy_data_set/standard_params/optimize_ann_optimizer_params_2023-10-12_17-19-09.json'

    # --------------- SGD With GBest Calm Data Set -------------------
    opt_sgd_gbest_calm_1 = 'opt_sgd_params_with_gbest_stats/calm_data_set/optimize_ann_optimizer_params_2023-10-13_15-09-11.json'
    opt_sgd_gbest_calm_2 = 'opt_sgd_params_with_gbest_stats/calm_data_set/optimize_ann_optimizer_params_2023-10-13_16-35-50.json'
    opt_sgd_gbest_calm_3 = 'opt_sgd_params_with_gbest_stats/calm_data_set/optimize_ann_optimizer_params_2023-10-13_21-46-26.json'

    # --------------- SGD With GBest Noisy Data Set -------------------
    opt_sgd_gbest_noisy_1 = 'opt_sgd_params_with_gbest_stats/noisy_data_set/optimize_ann_optimizer_params_2023-10-13_23-41-29.json'
    opt_sgd_gbest_noisy_2 = 'opt_sgd_params_with_gbest_stats/noisy_data_set/optimize_ann_optimizer_params_2023-10-14_01-14-12.json'
    opt_sgd_gbest_noisy_3 = 'opt_sgd_params_with_gbest_stats/noisy_data_set/optimize_ann_optimizer_params_2023-10-14_02-55-20.json'

    # --------------- SGD With RPSO Calm Data Set -------------------
    opt_sgd_rpso_calm_1 = 'opt_sgd_params_with_rpso_stats/calm_data_set/optimize_ann_optimizer_params_2023-10-15_02-36-24.json'
    opt_sgd_rpso_calm_2 = 'opt_sgd_params_with_rpso_stats/calm_data_set/optimize_ann_optimizer_params_2023-10-15_04-08-39.json'
    opt_sgd_rpso_calm_3 = 'opt_sgd_params_with_rpso_stats/calm_data_set/optimize_ann_optimizer_params_2023-10-15_05-30-08.json'

    # --------------- SGD With RPSO Noisy Data Set -------------------
    opt_sgd_rpso_noisy_1 = 'opt_sgd_params_with_rpso_stats/noisy_data_set/optimize_ann_optimizer_params_2023-10-15_11-06-14.json'
    opt_sgd_rpso_noisy_2 = 'opt_sgd_params_with_rpso_stats/noisy_data_set/optimize_ann_optimizer_params_2023-10-15_12-33-14.json'
    opt_sgd_rpso_noisy_3 = 'opt_sgd_params_with_rpso_stats/noisy_data_set/optimize_ann_optimizer_params_2023-10-15_14-05-30.json'

    # --------------- GBest Random Search Calm Data Set -------------------
    opt_ann_random_search_gbest_calm = 'opt_ann_gbest_uniform_distribution_search/calm_data_set/100_runs/stats_2023-10-08_15-29-34.json'

    # --------------- RPSO Random Search Calm Data Set -------------------
    opt_ann_random_search_rpso_calm = 'opt_ann_rpso_uniform_distribution_search/100_runs/stats_2023-10-09_21-01-27.json'

    file_paths = [
        opt_ann_gbest_calm_1, opt_ann_gbest_calm_2, opt_ann_gbest_calm_3, opt_ann_gbest_calm_4,
        opt_ann_gbest_noisy_1, opt_ann_gbest_noisy_2, opt_ann_gbest_noisy_3, opt_ann_gbest_noisy_4,
        opt_ann_rpso_calm_1, opt_ann_rpso_calm_2, opt_ann_rpso_calm_3, opt_ann_rpso_calm_4, opt_ann_rpso_calm_5, opt_ann_rpso_calm_6, opt_ann_rpso_calm_7,
        opt_ann_rpso_noisy_1, opt_ann_rpso_noisy_2, opt_ann_rpso_noisy_3, opt_ann_rpso_noisy_4, opt_ann_rpso_noisy_5, opt_ann_rpso_noisy_6,
        opt_adam_gbest_calm_1, opt_adam_gbest_calm_2, opt_adam_gbest_calm_3,
        opt_adam_gbest_noisy_1, opt_adam_gbest_noisy_2, opt_adam_gbest_noisy_3,
        opt_adam_rpso_calm_1, opt_adam_rpso_calm_2, opt_adam_rpso_calm_3, opt_adam_rpso_calm_4,
        opt_adam_rpso_noisy_1, opt_adam_rpso_noisy_2, opt_adam_rpso_noisy_3, opt_adam_rpso_noisy_4,
        opt_sgd_gbest_calm_1, opt_sgd_gbest_calm_2, opt_sgd_gbest_calm_3,
        opt_sgd_gbest_noisy_1, opt_sgd_gbest_noisy_2, opt_sgd_gbest_noisy_3,
        opt_sgd_rpso_calm_1, opt_sgd_rpso_calm_2, opt_sgd_rpso_calm_3,
        opt_sgd_rpso_noisy_1, opt_sgd_rpso_noisy_2, opt_sgd_rpso_noisy_3,
        opt_ann_random_search_gbest_calm, opt_ann_random_search_rpso_calm
    ]

    verification_stats = []

    for json_file_path in file_paths:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        try:
            ann_weights = data['best_weights']
        except KeyError:
            try:
                ann_weights = flatten_list(data['best_swarm_weights'])
            except KeyError:
                ann_weights = None

        mae, mse, rmse, medae, min_error, max_error, predicted_locations, absolute_errors = get_ann_stats(
            ann_weights)
        stats = [mae, mse, rmse, medae, min_error, max_error]

        folder_names = json_file_path.split("/")
        folder_names.pop()
        folder_names = [folder for folder in folder_names if folder]
        folder_names = "_".join(folder_names)

        subfolder = "verification_stats"
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

        stats_dict = {
            "File Name": folder_names,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MedAE": medae,
            "Min Error": min_error,
            "Max Error": max_error,
            "Predicted Locations": predicted_locations.tolist(),
            "Absolute Errors": absolute_errors.tolist()
        }

        verification_stats.append(stats_dict)

    result_dict = {
        "verification_stats": verification_stats
    }

    subfolder = "verification_stats/calm_fingerprinted_set"
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)

    output_json_file_name = os.path.join(subfolder, "verification_stats.json")

    with open(output_json_file_name, "w") as json_file:
        json.dump(result_dict, json_file, indent=4)

    print(f"Statistics saved to {output_json_file_name}")

def collect_verification_set_stats_from_random_search():
    subfolder_path = os.path.join("opt_ann_gbest_uniform_distribution_search", "with_velocity_bounds")

    json_data = []
    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)

        if filename.endswith(".json"):
            with open(file_path, 'r') as file:
                try:
                    json_content = json.load(file)
                    ann_weights = json_content['best_weights']
                    mae, mse, rmse, medae, min_error, max_error, predicted_locations, absolute_errors = get_ann_stats(
                        ann_weights)
                    json_data.append({
                        "File Name": filename,
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse,
                        "MedAE": medae,
                        "Min Error": min_error,
                        "Max Error": max_error,
                        "Predicted Locations": predicted_locations.tolist(),
                        "Absolute Errors": absolute_errors.tolist()
                    })
                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {file_path}")


    result_dict = {
        "verification_stats": json_data
    }

    subfolder = "verification_stats/random_distribution_runs_gbest/with_velocity_bounds/noisy_fingerprinted_set"
    
    if not os.path.exists(subfolder):
        os.makedirs(subfolder, exist_ok=True)

    output_json_file_name = os.path.join(subfolder, "verification_stats.json")

    with open(output_json_file_name, "w") as json_file:
        json.dump(result_dict, json_file, indent=4)
    
def read_verification_stats(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except (IOError, json.JSONDecodeError) as e:
        # Handle any potential errors while reading the file or parsing JSON
        print(f"Error reading {file_path}: {e}")
        return None

def find_and_extract_verification_stats(folder_path):
    # Dictionary to store extracted data (file name, MedAE, Max Error, and folder path)
    extracted_data = defaultdict(list)

    # Recursively find and extract data from verification_stats.json files
    for foldername, subfolders, filenames in os.walk(folder_path):
        if 'verification_stats.json' in filenames:
            file_path = os.path.join(foldername, 'verification_stats.json')
            data = read_verification_stats(file_path)
            if data is not None:
                for stats in data["verification_stats"]:
                    file_name = stats["File Name"]
                    medae = stats["MedAE"]
                    max_error = stats["Max Error"]
                    extracted_data[file_name].append({
                        "MedAE": medae,
                        "Max Error": max_error,
                        "Folder Path": foldername
                    })

    return extracted_data

def calculate_combined_performance(data):
    # Calculate the combined performance score (average of MedAE and Max Error) for each file name
    combined_performance = {}
    for file_name, file_data in data.items():
        num_stats = len(file_data)
        total_performance = sum((entry["MedAE"] + entry["Max Error"]) / 2 for entry in file_data) / num_stats
        combined_performance[file_name] = total_performance
    return combined_performance



def get_best_verified_models():
    json_file_path = "verification_stats/noisy_fingerprinted_set/verification_stats.json"

    best = find_combined_best(json_file_path, top=5)
    for x in best:
        print( x["File Name"], x["MedAE"], x["Max Error"])

def get_worst_verified_models():
    json_file_path = "verification_stats/random_distribution_runs_gbest/with_velocity_bounds/calm_random_points_set/verification_stats.json"

    worst = find_combined_worst(json_file_path, bottom=5)
    for x in worst:
        print( x["File Name"], x["MedAE"], x["MAE"], x["MSE"], x["RMSE"], x["Min Error"], x["Max Error"])

if __name__ == "__main__":
#    collect_verification_set_stats()
    # collect_verification_set_stats_from_random_search()
    # get_best_verified_models()
    # get_worst_verified_models()

    # Example usage:
    folder_path = 'verification_stats'
    extracted_data = find_and_extract_verification_stats(folder_path)

    # Calculate the combined performance scores
    combined_performance = calculate_combined_performance(extracted_data)

    # Sort the file names based on combined performance and select the top ten
    sorted_combined_performance = sorted(combined_performance.items(), key=lambda x: x[1])[:10]

    # Print the top ten file names, MedAE, and Max Error values
    for index, (file_name, _) in enumerate(sorted_combined_performance):
        file_data = extracted_data[file_name]
        medae = sum(entry["MedAE"] for entry in file_data) / len(file_data)
        max_error = sum(entry["Max Error"] for entry in file_data) / len(file_data)
        print(f"Top {index + 1} - File Name: {file_name}, Average MedAE: {medae}, Average Max Error: {max_error}")