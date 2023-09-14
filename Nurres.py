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
    X_train, X_test, Y_train, Y_test = train_test_split(
        free_variables, dependent_variables, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(6, activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(2))

    adam_optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=0.0600433,
        beta_1=0.54908875,  # Custom value for beta_1
        beta_2=0.65616656,  # Custom value for beta_2
        epsilon=1e-07  # Custom value for epsilon
    )
    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor (usually validation loss)
        patience=500,           # Number of epochs with no improvement after which training will stop
        restore_best_weights=True
    )
    # # Define the EarlyStopping callback
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss",  # Metric to monitor (usually validation loss)
    #     patience=50,  # Number of epochs with no improvement after which training will stop
    #     restore_best_weights=True,
    # )

    model.compile(optimizer=adam_optimizer, loss="mse")

    model.fit(X_train, Y_train, epochs=250,
              validation_data=(X_test, Y_test),
              callbacks=[early_stopping],
              verbose=0)
    Xlosses = pd.DataFrame(model.history.history)

    # Extract loss and validation loss columns
    train_loss = Xlosses["loss"]
    val_loss = Xlosses["val_loss"]

    # this should produce (1, 0)
    some_position = [[75, 87, 80, 6920, 17112, 17286]]
    # this should produce (8,6)
    some_position_2 = [[72, 78, 81, 8503, 8420, 8924]]
    transformed_some_position = scaler.transform(some_position)
    transformed_some_position_2 = scaler.transform(some_position_2)

    coord_one = model.predict(transformed_some_position)
    coord_two = model.predict(transformed_some_position_2)

    squared_distance_one = (
        sum((p1 - p2) ** 2 for p1, p2 in zip((1, 0), coord_one[0])))**0.5

    squared_distance_two = (
        sum((p1 - p2) ** 2 for p1, p2 in zip((6, 8), coord_two[0])))**0.5

    print(coord_one, squared_distance_one)
    print(coord_two, squared_distance_two)
    model.save('my_model.keras')

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


generate_x_coord_model()
