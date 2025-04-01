import keras

"""
Module for loading MNIST data and building Keras models.

This module provides functions to load and preprocess the MNIST dataset and to build a neural network model.
It supports both fully connected networks (MLP) and convolutional neural networks (CNN).
"""


def load_mnist():
    """
    Load and preprocess the MNIST dataset.

    This function loads the MNIST dataset, normalizes the image data to the [0, 1] range, and converts the
    digit labels into one-hot encoded vectors.

    Returns
    -------
    x_train : numpy.ndarray
        Normalized training images.
    y_train : numpy.ndarray
        One-hot encoded training labels.
    x_test : numpy.ndarray
        Normalized test images.
    y_test : numpy.ndarray
        One-hot encoded test labels.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    return x_train, y_train, x_test, y_test


def build_model(cnn=True):
    """
    Build a neural network model.

    Constructs a Keras Sequential model. If `cnn` is True, the function builds a convolutional neural network;
    otherwise, it builds a fully connected network (MLP).

    Parameters
    ----------
    cnn : bool, optional
        If True, builds a convolutional neural network. If False, builds a fully connected network. Default is True.

    Returns
    -------
    model : keras.Sequential
        The constructed Keras model.
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28, 1)))
    if not cnn:
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=256, activation="leaky_relu"))
        model.add(keras.layers.Dense(units=128, activation="leaky_relu"))
        model.add(keras.layers.Dense(units=64, activation="relu"))
        model.add(keras.layers.Dense(units=32, activation="relu"))
        model.add(keras.layers.Dense(units=16, activation="relu"))
    else:
        model.add(keras.layers.Conv2D(filters=8, kernel_size=(2, 2), activation="relu", padding="same"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=10, activation="softmax"))
    return model


if __name__ == "__main__":
    epochs = 10
    learning_rate = 0.001
    random_seed = 42

    keras.utils.set_random_seed(random_seed)
    x_train, y_train, x_test, y_test = load_mnist()

    mlp_model = build_model(cnn=False)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    mlp_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    mlp_model.summary()
    history = mlp_model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1, validation_split=0.1)
    test_loss1, test_acc1 = mlp_model.evaluate(x_test, y_test, verbose=1)

    cnn_model = build_model(cnn=True)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    cnn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    cnn_model.summary()
    history = cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1, validation_split=0.1)
    test_loss, test_acc = cnn_model.evaluate(x_test, y_test, verbose=1)

    print("Test results:")
    print("-" * 32)
    print("{:<10} {:>10} {:>10}".format("Model", "Accuracy", "Loss"))
    print("-" * 32)
    print("{:<10} {:>10.4f} {:>10.4f}".format("MLP", test_acc1, test_loss1))
    print("{:<10} {:>10.4f} {:>10.4f}".format("CNN", test_acc, test_loss))
