import os
import json
import matplotlib.pyplot as plt


def find_highest_level_dirs(start_dir='.'):
    """
    Find the highest-level directories under the specified directory.

    Args:
        start_dir (str): The directory to start the search. Default is the current directory ('.').

    Returns:
        list: A list of highest-level directory paths.
    """
    highest_level_dirs = set()

    for dirpath, _, _ in os.walk(start_dir):
        # Split the directory path into components
        dir_components = dirpath.split(os.path.sep)
        if len(dir_components) > 1:
            # Get the second component as the highest-level directory
            highest_level_dir = dir_components[1]
            highest_level_dirs.add(highest_level_dir)

    return list(highest_level_dirs)


def gather_data_and_create_charts(highest_level_dirs):
    for highest_level_dir in highest_level_dirs:
        json_files = find_json_files(highest_level_dir)
        data_list = gather_data(json_files)

        # Create separate charts for each field (elapsed_time, pso_type, function_name)
        create_bar_chart(data_list, 'Elapsed Time',
                         'elapsed_time', highest_level_dir)
        create_bar_chart(data_list, 'PSO Type', 'pso_type', highest_level_dir)
        create_bar_chart(data_list, 'Function Name',
                         'function_name', highest_level_dir)


def find_json_files(directory):
    """
    Find all .json files in the specified directory.

    Args:
        directory (str): The directory to search for .json files.

    Returns:
        list: A list of file paths to the found .json files.
    """
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(directory, filename)
            json_files.append(json_file_path)
    return json_files


def gather_data(json_files):
    data_list = []

    for json_file_path in json_files:
        with open(json_file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                elapsed_time = data.get("elapsed_time")
                pso_type = data.get("pso_type")
                function_name = data.get("function_name")

                data_dict = {
                    "File": json_file_path,
                    "Elapsed Time": elapsed_time,
                    "PSO Type": pso_type,
                    "Function Name": function_name if function_name else os.path.basename(os.path.dirname(json_file_path))
                }

                data_list.append(data_dict)

            except json.JSONDecodeError:
                print(f"File: {json_file_path}, Error: Invalid JSON")

    return data_list


def create_bar_chart(data_list, title, field_name, output_dir):
    values = []
    labels = []

    for data in data_list:
        field_value = data.get(field_name)
        if field_value is not None:
            values.append(field_value)
            labels.append(data['Function Name'])

    if not values:
        print(f"No '{field_name}' data found for {os.path.basename(output_dir)}")
        return

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, align='center', alpha=0.5)
    plt.xlabel('Function Name')
    plt.ylabel(title)
    plt.title(f'{title} - {os.path.basename(output_dir)}')
    plt.xticks(rotation=45, ha="right")

    chart_filename = os.path.join(output_dir, f'{field_name}_chart.png')
    plt.savefig(chart_filename)
    plt.close()


if __name__ == "__main__":
    # Find the highest-level directories
    highest_level_dirs = find_highest_level_dirs()
    # Gather data and create charts for each highest-level directory
    gather_data_and_create_charts(highest_level_dirs)
