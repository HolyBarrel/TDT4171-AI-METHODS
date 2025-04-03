import pickle
import keras


def read_file(path):
    with open(file = path, mode = "rb") as file :
        data = pickle.load(file)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
        vocab_size = data["vocab_size"]
        max_length = data["max_length"]

        return x_train, y_train, x_test, y_test, vocab_size, max_length

def padde(trains, max_length): 
    padder = []
    for sequence in trains:
        padder.append(keras.utils.pad_sequences(sequence, maxlen=max_length))
    return padder


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, vocab_size, max_length = read_file("data/keras-data.pickle")

    padded_data = padde([x_train, y_train, x_test, y_test])

    x_train = padded_data[0]
    y_train = padded_data[1]
    x_test = padded_data[2]
    y_test = padded_data[3]

    print(vocab_size)
    print(max_length)