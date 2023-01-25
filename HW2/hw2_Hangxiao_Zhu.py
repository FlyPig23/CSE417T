#!/usr/bin/python3
# Homework 2 Code
import numpy as np
import pandas as pd
import time


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


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
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
        # Update the weight vector
        w = w - eta * grad
        # Update the iteration counter
        t = t + 1

    # Calculate the in-sample error
    e_in = np.sum(np.log(1 + np.exp(-y * np.dot(X, w)))) / N

    return t, w, e_in


def main():
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')

    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')

    # Convert labels from 0/1 to -1/1
    train_data['heartdisease::category|0|1'] = train_data['heartdisease::category|0|1'].replace(0, -1)
    test_data['heartdisease::category|0|1'] = test_data['heartdisease::category|0|1'].replace(0, -1)

    # Extract training data matrix and labels
    X_train = train_data.drop('heartdisease::category|0|1', axis=1).to_numpy()
    y_train = train_data['heartdisease::category|0|1'].to_numpy()

    # Extract test data matrix and labels
    X_test = test_data.drop('heartdisease::category|0|1', axis=1).to_numpy()
    y_test = test_data['heartdisease::category|0|1'].to_numpy()

    # Use learning rate 10^-5, terminate if the magnitude of every element of gradient is less than 10^-3
    eta = 1e-5
    grad_threshold = 1e-3

    # Initialize the weight vector to be all zeros with one bias term at the beginning
    w_init = np.zeros(X_train.shape[1] + 1)

    # Train the model three times with maximum number of iterations:10^4, 10^5, 10^6
    max_its = np.zeros(3)
    max_its[0] = 1e4
    max_its[1] = 1e5
    max_its[2] = 1e6

    # Run gradient descent for each of the three cases
    for i in range(3):
        # Track how long the training process took in seconds
        start_time = time.time()
        t, w, e_in = logistic_reg(X_train, y_train, w_init, max_its[i], eta, grad_threshold)
        end_time = time.time()
        training_time = end_time - start_time
        binary_error_on_train = find_binary_error(w, X_train, y_train)
        binary_error_on_test = find_binary_error(w, X_test, y_test)
        print('Maximum number of iterations: %d' % max_its[i])
        print('Training time: ' + str(training_time))
        print('Number of iterations: ' + str(t))
        print('In-sample error: ' + str(e_in))
        print('Binary classification error on training set: ' + str(binary_error_on_train))
        print('Binary classification error on test set: ' + str(binary_error_on_test))
        print('')

    # Normalize the features, i.e., scale each feature by subtracting the mean and dividing by the standard
    # deviation for each of the features
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Experiment with different learning rates: 0.01, 0.1, 1, 4, 7, 7.5, 7.6, 7,7
    new_eta = np.zeros(8)
    new_eta[0] = 0.01
    new_eta[1] = 0.1
    new_eta[2] = 1
    new_eta[3] = 4
    new_eta[4] = 7
    new_eta[5] = 7.5
    new_eta[6] = 7.6
    new_eta[7] = 7.7

    # Terminate if the magnitude of every element of gradient is less than 10^-6
    new_grad_threshold = 1e-6

    # Run gradient descent for each of the eight cases, with maximum number of iterations 10^6
    for i in range(8):
        # Track how long the training process took in seconds
        start_time = time.time()
        t, w, e_in = logistic_reg(X_train_norm, y_train, w_init, max_its[2], new_eta[i], new_grad_threshold)
        end_time = time.time()
        training_time = end_time - start_time
        binary_error_on_train = find_binary_error(w, X_train_norm, y_train)
        binary_error_on_test = find_binary_error(w, X_test_norm, y_test)
        print('Learning rate: ' + str(new_eta[i]))
        print('Training time: ' + str(training_time))
        print('Number of iterations: ' + str(t))
        print('In-sample error: ' + str(e_in))
        print('Binary classification error on training set: ' + str(binary_error_on_train))
        print('Binary classification error on test set: ' + str(binary_error_on_test))
        print('')


if __name__ == "__main__":
    main()
