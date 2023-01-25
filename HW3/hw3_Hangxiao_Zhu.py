#!/usr/bin/python3
# Homework 3 Code
import numpy as np
from sklearn.preprocessing import StandardScaler


def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Add a column of 1s to the data matrix
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # Label the data points based on the weight vector
    # Use a cutoff probability of 0.5
    sigmoid = 1 / (1 + np.exp(-np.dot(X, w)))
    y_hat = np.where(sigmoid >= 0.5, 1, -1)

    # Calculate the binary error
    binary_error = np.sum(y_hat != y) / y.shape[0]

    return binary_error


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold, lambda_value, regularization):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Initialize iteration counter, weight vector
    t = 0
    w = w_init

    # Add a column of 1s to the data matrix
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # Get the number of data points
    N = X.shape[0]

    while t < max_its:
        # Calculate the gradient
        grad = np.dot(X.T, ((-y) / (1 + np.exp(y * np.dot(X, w))))) / N
        # Check if the magnitude of every element of gradient is smaller than grad_threshold
        if all(abs(grad) < grad_threshold):
            break
        # Update the weight vector with L1 or L2 regularization
        if regularization == "L1":
            w_temp = w - eta * grad
            w = w_temp - eta * lambda_value * np.sign(w_temp)
            # For each dimension i, if w_i and w_temp_i have different signs and w_temp_i is not 0, set w_i to 0
            w = np.where(np.sign(w) != np.sign(w_temp), 0, w)
        elif regularization == "L2":
            w = (1 - 2 * eta * lambda_value) * w - eta * grad
        # Update the iteration counter
        t = t + 1

    # Calculate the in-sample error
    e_in = np.sum(np.log(1 + np.exp(-y * np.dot(X, w)))) / N

    return t, w, e_in


def main():
    # Load the data
    X_train, X_test, y_train, y_test = np.load("digits_preprocess.npy", allow_pickle=True)

    # Convert labels from 0/1 to -1/1
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # Normalize the features, i.e. scale each feature by subtracting the mean and dividing by the standard deviation
    # Use the same mean and standard deviation for both training and test data
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Use learning rate 0.01
    eta = 0.01

    # Terminate if the magnitude of every element of gradient is less than 10^-6
    grad_threshold = 1e-6

    # Train the model three times with maximum number of iterations 10^4
    max_its = 1e4

    # Initialize the weight vector to be all zeros with one bias term at the beginning
    w_init = np.zeros(X_train.shape[1] + 1)

    # Examine different lambda values: 0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1 for both L1 and L2 regularization
    lambda_values = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]

    # L1 or L2 regularization
    regularization = ["L1", "L2"]

    for reg in regularization:
        print("Regularization: " + reg)
        for lambda_value in lambda_values:
            print("Lambda value: " + str(lambda_value))
            # Train the model three times with maximum number of iterations 10^4
            t, w, e_in = logistic_reg(X_train_norm, y_train, w_init, max_its, eta, grad_threshold, lambda_value, reg)
            # Get the number of 0s in the learned weight vector
            num_zeros = np.sum(w == 0)
            binary_error_on_test = find_binary_error(w, X_test_norm, y_test)
            print("Number of zeros in the learned weight vector: " + str(num_zeros))
            print('Binary classification error on test set: ' + str(binary_error_on_test))
            print("")


if __name__ == "__main__":
    main()
