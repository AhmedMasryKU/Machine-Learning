from numpy import genfromtxt
import csv
import numpy as np
import math
import matplotlib.pyplot as plt

# Hyper parameters. 
eta = 0.0001
epsilon = 1e-3
max_iteration = 500

def read_csv(filename):
    with open(filename, newline='') as f_input:
        return [list(map(float, row)) for row in csv.reader(f_input)]

def safe_log(num):
    return math.log(num + 1e-12)


# Gradient dW fundtin
def dW(X, Y, Y_predicted):
    return np.matmul(np.transpose(X), (Y - Y_predicted) * Y_predicted * (1-Y_predicted))

# Gradient dW0 function.
def dW0(Y, Y_predicted):
    dw0 = np.sum((Y - Y_predicted) * Y_predicted * (1-Y_predicted), axis = 0)
    dw0 = dw0.reshape((len(dw0), 1))
    return dw0

# Sigmoid function. 
def sigmoid(a):
    return 1 / (1 + np.exp(-(a)))

def predict(X, W, w0):
    A = np.concatenate((X, np.ones(shape = (X.shape[0],1))), axis = 1)
    B = np.concatenate((W, w0.T), axis = 0)
    return sigmoid(np.matmul(A, B))

def calculate_conf(X, Y, W, w0, k):
    Y_predicted_mat = predict(X, W, w0)
    Y_predicted_vec = np.argmax(Y_predicted_mat, axis = 1)
    
    conf = np.zeros(shape = (k, k))
    
    for i in range(len(Y)):
        conf[Y_predicted_vec[i], int(Y[i]) - 1] += 1
    return conf

def main():
    # Load data. 
    images = np.array(read_csv("hw02_images.csv"))
    labels = np.array(read_csv("hw02_labels.csv"))

    # Split the data. 
    images_train = images[0:500]
    images_test = images[500:1000]

    labels_train = labels[0:500]
    labels_test = labels[500:1000]

    # Load initial W an w0. 
    W = np.array(read_csv("initial_W.csv"))
    w0 = np.array(read_csv("initial_w0.csv"))

   # Number of classes and exmaples. 
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
        Y_predicted = predict(images_train, W, w0)
        
        # calculate error. 
        error = 0.5 * np.sum(np.square(Y-Y_predicted))
        errors.append(error)
        
        # save old parameters. 
        W_old = W
        w0_old = w0
        
        # update parameters. 
        W = W + eta * dW(images_train, Y, Y_predicted)
        w0 = w0 + eta * dW0(Y, Y_predicted)
        
        iteration += 1
        
        if math.sqrt(np.sum(np.square(w0 - w0_old)) + np.sum(np.square(W - W_old))) < epsilon:
            break
        
    plt.plot(errors)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.show()

    # Training data confusion matrix. 
    print("Training data confusion matrix")
    print(calculate_conf(images_train, labels_train, W, w0, k))
    # Testing data confusion matrix. 
    print("Testing data confusion matrix")
    print(calculate_conf(images_test, labels_test, W, w0, k))

if __name__=='__main__':
    main()