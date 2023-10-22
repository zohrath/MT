import math
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_fingerprinted_data_noisy():
    df = pd.read_csv("fingerprints-noisy.csv", delimiter=",")
    df = df[(df['X'] % 2 == 0) | (df['Y'] % 2 == 0)]
    df = df.drop(
        [
            "AP1_dev",
            "AP2_dev",
            "AP3_dev",
            "AP1_dist_dev",
            "AP2_dist_dev",
            "AP3_dist_dev",
            "id"
        ],
        axis=1,
    )

    free_variables = df.drop(["X", "Y"], axis=1).values
    dependent_variables = df[["X", "Y"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        free_variables, dependent_variables, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def get_fingerprinted_random_points_noisy_data():
    df = pd.read_csv("fingerprints-random-points-noisy.csv", delimiter=",")
    df = df.drop(
        [
            "AP1_dev",
            "AP2_dev",
            "AP3_dev",
            "AP1_dist_dev",
            "AP2_dist_dev",
            "AP3_dist_dev",
            "id"
        ],
        axis=1,
    )

    free_variables = df.drop(["X", "Y"], axis=1).values
    dependent_variables = df[["X", "Y"]].values

    scaler = MinMaxScaler()

    free_variables = scaler.fit_transform(free_variables)

    return free_variables, dependent_variables


def get_fingerprinted_random_points_calm_data():
    df = pd.read_csv("fingerprints-random-points-calm.csv", delimiter=",")
    df = df.drop(
        [
            "AP1_dev",
            "AP2_dev",
            "AP3_dev",
            "AP1_dist_dev",
            "AP2_dist_dev",
            "AP3_dist_dev",
            "id"
        ],
        axis=1,
    )

    free_variables = df.drop(["X", "Y"], axis=1).values
    dependent_variables = df[["X", "Y"]].values

    scaler = MinMaxScaler()

    free_variables = scaler.fit_transform(free_variables)

    return free_variables, dependent_variables


def get_fingerprinted_data():
    df = pd.read_csv("fingerprints-calm.csv", delimiter=",")
    df = df[(df['X'] % 2 == 0) | (df['Y'] % 2 == 0)]
    df = df.drop(
        [
            "AP1_dev",
            "AP2_dev",
            "AP3_dev",
            "AP1_dist_dev",
            "AP2_dist_dev",
            "AP3_dist_dev",
        ],
        axis=1,
    )

    free_variables = df.drop(["X", "Y"], axis=1).values
    dependent_variables = df[["X", "Y"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        free_variables, dependent_variables, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def sphere(x):
    return sum(x_i**2 for x_i in x)


def rosenbrock_function(x):
    result = 0.0
    n = len(x)
    for i in range(n - 1):
        term1 = 100.0 * (x[i + 1] - x[i]**2)**2
        term2 = (x[i] - 1)**2
        result += term1 + term2
    return result


def rosenbrock(args):
    return sum(
        (1 - x) ** 2 + 100 * (y - x**2) ** 2 for x, y in zip(args[::2], args[1::2])
    )


def rastrigin_function(x):
    result = 0.0
    for xi in x:
        result += xi**2 - 10 * math.cos(2 * math.pi * xi) + 10
    return result


def rastrigin(X):
    A = 10
    return A + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])


def schwefel_function(x):
    D = len(x)
    result = 0.0
    for i in range(D):
        inner_sum = 0.0
        for j in range(i + 1):
            inner_sum += x[j]
        result += inner_sum**2
    return result


def schwefel(args):
    return 418.9829 * len(args) - sum(x * np.sin(np.sqrt(abs(x))) for x in args)


def griewank_function(x):
    n = len(x)
    sum_term = 0.0
    prod_term = 1.0

    for i in range(1, n + 1):
        sum_term += x[i - 1]**2
        prod_term *= math.cos(x[i - 1] / math.sqrt(i))

    result = 1 + (sum_term / 4000) - prod_term
    return result


def griewank(args):
    term1 = sum(x**2 for x in args) / 4000
    term2 = np.prod(list(np.cos(x / np.sqrt(i + 1))
                    for i, x in enumerate(args)))
    return term1 - term2 + 1


def penalty_function(x):
    if x < -10:
        return 100 * (-x - 10)**4
    elif -10 <= abs(x) <= 10:
        return 0
    else:
        return 100 * (x - 10)**4


def penalized_1_function(x):
    D = len(x)
    sum_term = 0.0

    for i in range(D):
        yi = 1 + (1/4) * (x[i] + 1)
        sum_term += (10 * math.sin(math.pi * yi)**2)

    penalty_term = sum(penalty_function(xi) for xi in x)

    result = (math.pi / D) * (sum_term + penalty_term)

    return result


def penalized1(args):
    penalty_term = (
        0.1
        * np.sqrt(sum(x**2 for x in args))
        * np.sin(50 * np.sqrt(sum(x**2 for x in args)))
    )
    return -((1 + penalty_term) ** 2)


def step_function(x):
    return sum((math.floor(xi + 0.5))**2 for xi in x)


def step(args, lower_bound=-5.0, upper_bound=5.0):
    for x in args:
        if not (lower_bound <= x <= upper_bound):
            return 1
    return 0
