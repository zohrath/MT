import numpy as np


def get_ann_node_count(ann_nodes):
    """
    Calculate the sum of elements in each sub-array and return individual counts per array,
    as well as the total count of all elements across all sub-arrays.

    Parameters:
    ann_nodes (list of np.ndarray): A list containing numpy arrays, each representing a sub-array.

    Returns:
    tuple: A tuple containing two elements:
        - individual_ann_node_count (list of int): A list of sums of elements for each sub-array.
        - total_nodes_count (int): Total count of all elements across all sub-arrays.
    """

    total_nodes_count = 0
    individual_ann_node_count = []

    for arr in ann_nodes:
        count = len(arr)
        total_nodes_count += count
        arr_sum = np.sum(arr)
        individual_ann_node_count.append(arr_sum)

    return individual_ann_node_count, total_nodes_count
