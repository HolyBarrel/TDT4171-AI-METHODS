import numpy as np
import matplotlib.pyplot as plt 

#######################################################
#                                                     #
#             Functions to generate data              #
#                                                     #
#######################################################

def func(X: np.ndarray) -> np.ndarray:
    """The data generating function"""
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2

def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """Add noise to the data generating function"""
    return func(X) + np.random.randn(len(X)) * epsilon

def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Provide training and test data for training the neural network"""
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    y_train = y_train.reshape((y_train.shape[0],1))
    y_test = y_test.reshape((y_test.shape[0],1))

    return X_train, y_train, X_test, y_test


#######################################################
#                                                     #
#             Feed-forward Neural Network             #
#                                                     #
#######################################################

def sigmoid(x):
    """
    Sigmoid activation function
    """
    return 1/(1+np.exp(-x))

def d_sigmoid(y):
    """
    Derivative of the sigmoid activation function - input y is output of sigmoid(x)
    """
    return y*(1-y)

def linear(x):
    """
    Linear activation function, returns the same as the input
    """
    return x

def d_linear(y):
    """
    Derivative of linear activation function, returns 1 in the form of the input y
    """
    return np.ones_like(y)

def mse(y, y_hat):
    """
    Mean squared error loss function
    """
    return np.mean((y - y_hat) ** 2)

def feed_forward(x, hidden_W, hidden_b, out_W, out_b):
    """
    Calculate the forward pass for input_data x
    """
    # Hidden layer calculations
    #print(hidden_b.shape)
    hidden_activations = sigmoid(hidden_W @ x.T + hidden_b.reshape(-1, 1)) 
    # Output calculations
    out_linear = hidden_activations.T @ out_W + out_b
    y_hat = linear(out_linear)
    return y_hat, hidden_activations

def backward(y_hat, y, x, hidden_activations, out_W):
    """
    Backpropagation for the neural net in the assignment
    Parameters are those needed for the calculation

    :param y_hat: predictions from forward pass
    :param y: target data
    :param x: input data
    :param hidden_activations: Activations of hidden layer from forward pass
    :out_W: current output weights

    :returns: the derivative updates to hidden_W, hidden_b, out_W and out_b

    """
    n = len(y)
    # gradients wrt output weights
    out_error = y_hat - y
    out_delta = out_error * d_linear(y_hat)
    d_L_d_ow = hidden_activations  @ (2/n * out_delta) # Size (2,1)
    # gradients wrt output bias
    d_L_d_ob =np.sum(2/n * out_delta, axis=0)   # Size (1,)
    # gradients wrt hidden weights
    hidden_error = out_W @ (2/n * out_delta.T)
    d_L_d_hw =  x @ (hidden_error * d_sigmoid(hidden_activations))  # Size (2,2) 
    # gradients wrt hidden bias
    d_L_d_hb =  np.sum(hidden_error * d_sigmoid(hidden_activations), axis=1)  # Size (2,)

    return d_L_d_hw, d_L_d_hb, d_L_d_ow, d_L_d_ob


def train(x_train, y_train, neural_net, learning_rate=0.01, epochs=10, batch_size=1):
    """
    Training loop for the neural net
    Executes a forward pass that calulates the current prediction,
     then updates the gradient through a backward pass,
     then the gradient is used for gradient decent to tweak the weight 
     toward a minimum for the loss function (mse)
    
    :param x_train: the x values of the training data
    :param y_train: target data for the training
    :param neural_net: the intialization net that is trained
    :param learning_rate: to determine the weighting of the gradient change for each backwards pass
    :param epochs: the number of times the training data is iterated through fully in training
    :param batch_size is used to split up the training data into batches of given integer size

    :returns: the training MSE and the final weights for the nn
    """

    hidden_W, hidden_b, out_W, out_b = neural_net
    error_hist = []

    n_batches = int(np.ceil(x_train.shape[0]/float(batch_size)))

    for e in range(epochs):

        errors = []
        learning_rate *= 0.992
        for i in range(n_batches):
            # Forward pass
            y_hat, hidden_activations = feed_forward(x_train[batch_size*i:batch_size*(i+1)],
                                                     hidden_W, hidden_b, out_W, out_b)
            # Compute error
            error = mse(y_train[batch_size*i:batch_size*(i+1)], y_hat)
            errors.append(error)
            # Backward pass
            d_L_d_hw, d_L_d_hb, d_L_d_ow, d_L_d_ob = backward(y_hat,
                                                              y_train[batch_size*i:batch_size*(i+1)],
                                                              x_train[batch_size*i:batch_size*(i+1)],
                                                              hidden_activations, out_W)
            # Update parameters
            hidden_W = hidden_W - learning_rate * d_L_d_hw
            hidden_b = hidden_b - learning_rate * d_L_d_hb
            out_W = out_W - learning_rate * d_L_d_ow
            out_b = out_b - learning_rate * d_L_d_ob
        
        epoch_error = np.mean(errors)
        error_hist.append(epoch_error)
        if(e % 50 == 0):
            print(f"Epoch {e}: mse {epoch_error:4f}", end="\n")

    return error_hist, epoch_error, (hidden_W, hidden_b, out_W, out_b)

def plot_mse(errors, max_epochs):
    # Plots the mse errors
    plt.title("Mean Squared Error for the Training") 
    plt.xlabel("Traning Epochs (#)") 
    plt.ylabel("MSE") 
    plt.plot(range(max_epochs), errors, color ="red") 
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)
    # Initialize weights for two hidden neurons
    # 2 X 2 W + 2 b
    hidden_weights = np.random.uniform(-0.5, 0.5, (2, 2))
    hidden_bias = np.random.uniform(-0.5, 0.5, 2)
    # Initialize weights for output neuron
    # 2 x 1 W + 1 b
    out_weights = np.random.uniform(-0.5, 0.5, (2, 1))
    out_bias = np.random.uniform(-0.5, 0.5, 1)
    out_bias.reshape(1)
    # This is the neural net
    neural_net = (hidden_weights, hidden_bias, out_weights, out_bias)
    # Training
    max_epochs = 350
    errors, avg_train_mse, neural_net_trained = train(X_train, y_train, neural_net, learning_rate=0.003, epochs=max_epochs,
                                          batch_size=1)
    # Calculate mse on test data
    y_hat_test, _ = feed_forward(X_test, *neural_net_trained)
    test_mse = mse(y_test, y_hat_test)

    print(f"Training done after {max_epochs} epochs")
    print(f"Training MSE: {avg_train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")

    plot_mse(errors, max_epochs)
