import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split


data = pd.read_csv("features_3_sec.csv")

mapping = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9,
}

# get columns of data from 20 to 42
X = data.iloc[:, 19:45]

y = np.array([mapping[i] for i in data.iloc[:, -1]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# create a neural network with 2 hidden layers of size 128 nodes each and a softmax output layer with 10 nodes for each class (genre)


def create_neural_network(data):
    """Creates a neural network with 2 hidden layers of size 128 nodes each and a softmax output layer with 10 nodes for each class (genre)

    :param data (pandas.DataFrame): Input data
    :return model: Neural network model
    """

    # create network topology
    model = keras.Sequential()

    model.add(keras.Input(data.shape[1]))

    # 1st hidden layer
    model.add(keras.layers.Dense(4096, activation="relu"))
    # model.add(keras.layers.Dropout(0.3))

    # 1st hidden layer
    model.add(keras.layers.Dense(2048, activation="relu"))
    # model.add(keras.layers.Dropout(0.3))

    # 1st hidden layer
    model.add(keras.layers.Dense(1024, activation="relu"))
    # model.add(keras.layers.Dropout(0.3))

    # 1st hidden layer
    model.add(keras.layers.Dense(512, activation="relu"))
    # model.add(keras.layers.Dropout(0.3))

    # 1st hidden layer
    model.add(keras.layers.Dense(256, activation="relu"))
    # model.add(keras.layers.Dropout(0.3))

    # 2nd hidden layer
    model.add(keras.layers.Dense(128, activation="relu"))
    # model.add(keras.layers.Dropout(0.3))

    # 3nd hidden layer
    model.add(keras.layers.Dense(56, activation="relu"))
    # model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics="accuracy"
    )

    return model


nn = create_neural_network(X)

nn.summary()

nn.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100)

nn.save('./frk-classifier')