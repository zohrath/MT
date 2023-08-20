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
    print(df.head())
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
    Xmodel.add(Dense(76, activation="relu"))
    Xmodel.add(Dense(86, activation="relu"))
    Xmodel.add(Dense(2))

    Xmodel.compile(optimizer="adam", loss="mse")

    Xmodel.fit(X_xtrain, y_xtrain, epochs=500, validation_data=(X_xtest, y_xtest))
    Xlosses = pd.DataFrame(Xmodel.history.history)

    # Extract loss and validation loss columns
    train_loss = Xlosses["loss"]
    val_loss = Xlosses["val_loss"]

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    some_position = [[75, 87, 80, 6920, 17112, 17286]]  # this should produce (1, 0)
    some_position_2 = [[72, 78, 81, 8503, 8420, 8924]]  # this should produce (8,6)
    transformed_some_position = scaler.transform(some_position_2)

    x_value = Xmodel.predict(transformed_some_position)
    # y_value = Ymodel.predict(transformed_some_position)

    # (x_value, y_value)
    print(x_value)


generate_x_coord_model()
