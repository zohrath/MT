import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from CostFunctions import (
    griewank,
    penalized1,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
    step,
)


def get_regression_data():
    df = pd.read_csv("fake_reg.csv")

    X = df[["feature1", "feature2"]].values
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    return X_train, X_test, y_train, y_test


def plot_data(X_train, y_train):
    feature1_values = X_train[:, 0]
    feature2_values = X_train[:, 1]

    plt.figure(figsize=(15, 5))

    # Plot for feature1
    plt.subplot(1, 3, 1)
    plt.scatter(feature1_values, y_train)
    plt.xlabel("Feature 1")
    plt.ylabel("Price")
    plt.title("Scatter plot for Feature 1 vs. Price")

    # Plot the line with hard-coded intercept and slope for feature1
    intercept1 = 100  # Replace with your desired intercept for feature1
    slope1 = 0.5  # Replace with your desired slope for feature1
    plt.plot(feature1_values, intercept1 + slope1 * feature1_values, color="red")

    # Plot for feature2
    plt.subplot(1, 3, 2)
    plt.scatter(feature2_values, y_train)
    plt.xlabel("Feature 2")
    plt.ylabel("Price")
    plt.title("Scatter plot for Feature 2 vs. Price")

    # Plot the line with hard-coded intercept and slope for feature2
    intercept2 = 0.471669  # Replace with your desired intercept for feature2
    slope2 = 0.2767752  # Replace with your desired slope for feature2
    plt.plot(feature2_values, intercept2 + slope2 * feature2_values, color="red")

    # Plot for price
    plt.subplot(1, 3, 3)
    plt.scatter(
        y_train, y_train
    )  # Plotting price against itself, just to show the distribution of prices.
    plt.xlabel("Price")
    plt.ylabel("Price")
    plt.title("Distribution of Prices")

    plt.tight_layout()
    plt.show()


def plot_sphere():
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_sphere = sphere(x_grid, y_grid)

    fig_sphere = plt.figure()
    ax_sphere = fig_sphere.add_subplot(111, projection="3d")
    ax_sphere.plot_surface(x_grid, y_grid, z_sphere, cmap="viridis")
    ax_sphere.set_xlabel("X")
    ax_sphere.set_ylabel("Y")
    ax_sphere.set_zlabel("Z")
    plt.title("Sphere Function")
    plt.show()


def plot_rosenbrock():
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_rosenbrock = rosenbrock(x_grid, y_grid)

    fig_rosenbrock = plt.figure()
    ax_rosenbrock = fig_rosenbrock.add_subplot(111, projection="3d")
    ax_rosenbrock.plot_surface(x_grid, y_grid, z_rosenbrock, cmap="viridis")
    ax_rosenbrock.set_xlabel("X")
    ax_rosenbrock.set_ylabel("Y")
    ax_rosenbrock.set_zlabel("Z")
    plt.title("Rosenbrock Function")
    plt.show()


def plot_rastrigin():
    x_range = np.linspace(-5.12, 5.12, 100)
    y_range = np.linspace(-5.12, 5.12, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_rastrigin = rastrigin(x_grid, y_grid)

    fig_rastrigin = plt.figure()
    ax_rastrigin = fig_rastrigin.add_subplot(111, projection="3d")
    ax_rastrigin.plot_surface(x_grid, y_grid, z_rastrigin, cmap="viridis")
    ax_rastrigin.set_xlabel("X")
    ax_rastrigin.set_ylabel("Y")
    ax_rastrigin.set_zlabel("Z")
    plt.title("Rastrigin Function")
    plt.show()


def plot_schwefel():
    x_range = np.linspace(-500, 500, 100)
    y_range = np.linspace(-500, 500, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_schwefel = schwefel(x_grid, y_grid)

    fig_schwefel = plt.figure()
    ax_schwefel = fig_schwefel.add_subplot(111, projection="3d")
    ax_schwefel.plot_surface(x_grid, y_grid, z_schwefel, cmap="viridis")
    ax_schwefel.set_xlabel("X")
    ax_schwefel.set_ylabel("Y")
    ax_schwefel.set_zlabel("Z")
    plt.title("Schwefel 1.2 Function")
    plt.show()


def plot_griewank():
    x_range = np.linspace(-30, 30, 100)
    y_range = np.linspace(-30, 30, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_griewank = griewank(x_grid, y_grid)

    fig_griewank = plt.figure()
    ax_griewank = fig_griewank.add_subplot(111, projection="3d")
    ax_griewank.plot_surface(x_grid, y_grid, z_griewank, cmap="viridis")
    ax_griewank.set_xlabel("X")
    ax_griewank.set_ylabel("Y")
    ax_griewank.set_zlabel("Z")
    plt.title("Griewank Function")
    plt.show()


def plot_penalized1():
    x_range = np.linspace(-50, 50, 100)
    y_range = np.linspace(-50, 50, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_penalized1 = penalized1(x_grid, y_grid)

    fig_penalized1 = plt.figure()
    ax_penalized1 = fig_penalized1.add_subplot(111, projection="3d")
    ax_penalized1.plot_surface(x_grid, y_grid, z_penalized1, cmap="viridis")
    ax_penalized1.set_xlabel("X")
    ax_penalized1.set_ylabel("Y")
    ax_penalized1.set_zlabel("Z")
    plt.title("Penalized 1 Function")
    plt.show()


def plot_step():
    x_range = np.linspace(-10, 10, 100)
    y_range = np.linspace(-10, 10, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    z_step = np.array(
        [
            step(x, y, lower_bound=-5.0, upper_bound=5.0)
            for x, y in zip(np.ravel(x_grid), np.ravel(y_grid))
        ]
    ).reshape(x_grid.shape)

    fig_step = plt.figure()
    ax_step = fig_step.add_subplot(111, projection="3d")
    ax_step.plot_surface(x_grid, y_grid, z_step, cmap="viridis")
    ax_step.set_xlabel("X")
    ax_step.set_ylabel("Y")
    ax_step.set_zlabel("Z")
    plt.title("Step Function")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot functions")
    parser.add_argument(
        "function",
        choices=[
            "sphere",
            "rosenbrock",
            "rastrigin",
            "schwefel",
            "griewank",
            "penalized1",
            "step",
        ],
        help="Choose the function to plot",
    )
    args = parser.parse_args()

    if args.function == "sphere":
        plot_sphere()
    elif args.function == "rosenbrock":
        plot_rosenbrock()
    elif args.function == "rastrigin":
        plot_rastrigin()
    elif args.function == "schwefel":
        plot_schwefel()
    elif args.function == "griewank":
        plot_griewank()
    elif args.function == "penalized1":
        plot_penalized1()
    elif args.function == "step":
        plot_step()
