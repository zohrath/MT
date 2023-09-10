import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout


def generate_x_coord_model():
    # df['Coordinates'] = (df['X'].astype(str) + '.' + df['Y'].astype(str)).astype(float)
    df = pd.read_csv("fingerprints.csv", delimiter=",")
    df = df[(df["X"] % 2 == 0) | (df["Y"] % 2 == 0)]
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

    # Split
    X_xtrain, X_xtest, y_xtrain, y_xtest = train_test_split(
        free_variables, dependent_variables, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()

    X_xtrain = scaler.fit_transform(X_xtrain)
    X_xtest = scaler.transform(X_xtest)

    Xmodel = Sequential()
    Xmodel.add(Dense(6, activation="relu"))
    Xmodel.add(Dense(6, activation="relu"))
    Xmodel.add(Dense(6, activation="relu"))
    Xmodel.add(Dense(2))
    # 0.05577431
    adam_optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=0.04936118,
        beta_1=0.63566626,  # Custom value for beta_1
        beta_2=0.85479784,  # Custom value for beta_2
        epsilon=1e-07,  # Custom value for epsilon
    )
    # # Define the EarlyStopping callback
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss",  # Metric to monitor (usually validation loss)
    #     patience=50,  # Number of epochs with no improvement after which training will stop
    #     restore_best_weights=True,
    # )

    Xmodel.compile(optimizer=adam_optimizer, loss="mse")

    Xmodel.fit(
        X_xtrain,
        y_xtrain,
        epochs=500,
        validation_data=(X_xtest, y_xtest),
        # callbacks=[early_stopping],
    )
    Xlosses = pd.DataFrame(Xmodel.history.history)

    # Extract loss and validation loss columns
    train_loss = Xlosses["loss"]
    val_loss = Xlosses["val_loss"]

    # # Create a plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_loss, label="Training Loss")
    # plt.plot(val_loss, label="Validation Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training and Validation Loss Over Epochs")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # this should produce (1, 0)
    some_position = [[75, 87, 80, 6920, 17112, 17286]]
    # this should produce (8,6)
    some_position_2 = [[72, 78, 81, 8503, 8420, 8924]]
    transformed_some_position = scaler.transform(some_position)
    transformed_some_position_2 = scaler.transform(some_position_2)

    coord_one = Xmodel.predict(transformed_some_position)
    coord_two = Xmodel.predict(transformed_some_position_2)

    squared_distance_one = (
        sum((p1 - p2) ** 2 for p1, p2 in zip((1, 0), coord_one[0]))
    ) ** 0.5

    squared_distance_two = (
        sum((p1 - p2) ** 2 for p1, p2 in zip((6, 8), coord_two[0]))
    ) ** 0.5

    print(coord_one, squared_distance_one)
    print(coord_two, squared_distance_two)
    Xmodel.save("my_model.keras")


generate_x_coord_model()
