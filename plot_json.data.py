import os
import json


def gather_and_print_statistics(directory):
    for root, dirs, files in os.walk(directory):
        ann_count = 0
        params_count = 0
        statistics_data = []

        for dir_name in dirs:
            subdirectory = os.path.join(root, dir_name)

            for subroot, subdirs, subfiles in os.walk(subdirectory):
                for subfile in subfiles:
                    if subfile.endswith(".json"):
                        file_path = os.path.join(subroot, subfile)
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                            if 'statistics' in data:
                                statistics_data.append(data['statistics'])
                                if 'opt_ann' in dir_name:
                                    ann_count += 1
                                elif 'opt_' in dir_name and 'opt_ann' not in dir_name:
                                    params_count += 1

        if statistics_data:
            if 'opt_ann' in dir_name:
                print(f"Found JSON optimizing ANN in: {subdirectory}")
            elif 'opt_' in dir_name and 'opt_ann' not in dir_name:
                print(f"Found JSON optimizing params in: {subdirectory}")

            # Print all "statistics" field values for all JSON files in the super-folder
            print("Statistics from JSON files:")
            for index, statistics in enumerate(statistics_data):
                print(f"JSON file {index + 1}: {statistics}")

        if ann_count > 0:
            print(f"Total {ann_count} JSON files optimizing ANN in: {root}")
        if params_count > 0:
            print(
                f"Total {params_count} JSON files optimizing params in: {root}")


# Get the script's current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Example usage:
gather_and_print_statistics(script_dir)
