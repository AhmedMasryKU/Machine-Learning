from numpy import genfromtxt
import csv
import numpy as np
import math
import matplotlib.pyplot as plt

# Hyper parameters. 
eta = 0.0005
epsilon = 1e-3
max_iteration = 500
H = 20

def read_csv(filename):
    with open(filename, newline='') as f_input:
        return [list(map(float, row)) for row in csv.reader(f_input)]

def safe_log(num):
    return np.log(num + 1e-100)

# Gradient dW fundtin
def dW(Z, X, Y, Y_predicted, V):
    dW_tmp = np.matmul((Y - Y_predicted), np.transpose(V[1:21,:]))
    dW_tmp *= (Z*(np.ones(shape = Z.shape)-Z))
    return np.matmul(np.transpose(np.concatenate((np.ones(shape = (X.shape[0],1)), X), axis = 1)), dW_tmp)

# Gradient dV function.
def dV(Y, Y_predicted, Z):
    return np.matmul(np.transpose(np.concatenate((np.ones(shape = (Z.shape[0],1)), Z), axis = 1)), (Y - Y_predicted))

# Sigmoid function. 
def sigmoid(a):
    return 1 / (1 + np.exp(-(a)))

# Softmax function.
def softmax(Z, V):
    Z_with_bias = np.concatenate((np.ones(shape = (Z.shape[0],1)), Z), axis = 1)
    scores = np.exp(np.matmul(Z_with_bias, V))
    return scores / np.sum(scores, axis = 1, keepdims = True)

# Calculte Confusion Matrix. 
def calculate_conf(X, Y, W, V, k):
    Z = np.matmul(np.concatenate((np.ones(shape = (X.shape[0],1)), X), axis = 1), W)
    Z_sig = sigmoid(Z)
    Y_predicted = softmax(Z_sig, V)
    Y_predicted_vec = np.argmax(Y_predicted, axis = 1)
    conf = np.zeros(shape = (k, k))
    for i in range(len(Y)):
        conf[Y_predicted_vec[i], int(Y[i]) - 1] += 1
    return conf

def main():
    # Load data. 
    images = np.array(read_csv("hw03_images.csv"))
    labels = np.array(read_csv("hw03_labels.csv"))

    # Split the data. 
    images_train = images[0:500]
    images_test = images[500:1000]

    labels_train = labels[0:500]
    labels_test = labels[500:1000]

    # Load initial W an w0. 
    W = np.array(read_csv("initial_W.csv"))
    V = np.array(read_csv("initial_V.csv"))

    # Number of classes and examples. 
    k = int(np.max(labels))
    n = int(len(labels) / 2)


    # One hot vectors for the correct classes. 
    Y = np.zeros((n, k))
    Y[np.arange(n), (labels_train.astype(int) - 1).flatten()] = 1

    # Gradient Desecnt algorithm. 
    error = 100
    iteration = 0
    errors = []

    while iteration <  max_iteration:
        # predict scores. 
        Z = np.matmul(np.concatenate((np.ones(shape = (images_train.shape[0],1)), images_train), axis = 1), W)
        Z_sig = sigmoid(Z)
        Y_predicted = softmax(Z_sig, V)
        # calculate error. 
        error = - np.sum(Y * safe_log(Y_predicted))
        errors.append(error)

        # update parameters. 
        W = W + eta * dW(Z_sig, images_train, Y, Y_predicted, V)
        V = V + eta * dV(Y, Y_predicted, Z_sig)

        # Finish if the difference between the previous and current error is less than epsilon.
        if iteration != 0 and abs(errors[iteration] - errors[iteration - 1]) < epsilon:
            break
        
        iteration += 1

    plt.plot(errors)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.show()

    # Training data confusion matrix. 
    print("Training data confusion matrix")
    print(calculate_conf(images_train, labels_train, W, V, k))
    # Testing data confusion matrix. 
    print("Testing data confusion matrix")
    print(calculate_conf(images_test, labels_test, W, V, k))

if __name__=='__main__':
    main()