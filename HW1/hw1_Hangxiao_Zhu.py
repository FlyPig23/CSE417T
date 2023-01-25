#!/usr/bin/python3
# Homework 1 Code
import numpy as np
import matplotlib.pyplot as plt


def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Initialize the weight vector w as a zero weight vector, where the first dimension is the bias term.
    w = np.zeros(data_in.shape[1] - 1)
    # Initialize the number of iterations
    iterations = 0

    # Initialize the boolean variable to indicate whether the algorithm has converged
    converged = False
    # Run the algorithm until it converges
    while not converged:
        x = data_in[:, :-1]
        y = data_in[:, -1]
        predictions = np.sign(np.dot(x, w))
        if np.array_equal(predictions, y):
            converged = True
        else:
            # Find the index of the first misclassified data point
            index = np.where(predictions != y)[0][0]
            # Update the weight vector
            w = w + (y[index] * x[index])
            # Increment the number of iterations
            iterations += 1

    return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW1
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    for i in range(num_exp):

        '''Initialize weight vector w, where the first dimension is the bias term, which is initialized to 0.
        The rest of the dimensions are initialized to random values between 0 and 1.'''
        # Initialize the weight vector
        w = np.zeros(d + 1)
        # Assign random values to each dimension except the first
        w[1:] = np.random.uniform(0, 1, (1, d))
        # Get the transpose of the weight vector
        w_t = np.transpose(w)

        '''Initialize the training set as a matrix with each row representing an (x,y) pair, with the x vector augmented 
        with an initial 1 (i.e., x_0), and the label y in the last column.
        For each training data point x, sample each dimension from a uniform distribution between -1 and 1. And then 
        insert x_0 = 1 at the beginning of the vector. 
        The label y is determined by the sign of the dot product of w and x.'''
        # Initialize the training set
        data_in = np.zeros((N, d + 2))
        # Insert x_0 = 1 at the beginning of each row
        data_in[:, 0] = 1
        # Assign random values to each data point's dimensions except the first and last
        data_in[:, 1:-1] = np.random.uniform(-1, 1, (N, d))
        # Calculate the label y to each data point
        data_in[:, -1] = np.sign(np.dot(data_in[:, :-1], w_t))

        # Run PLA on the input data
        w_trained, iterations = perceptron_learn(data_in)
        w_trained_t = np.transpose(w_trained)

        # Use the optimal weight vector w to calculate the maximum number of iterations
        # Find the maximum value of norm(x) for all x in the training set
        r = np.amax(np.linalg.norm(data_in[:, :-1], axis=1))
        # Find the minimum dot product value of w and x for all x in the training set
        rho = np.amin(np.abs(np.dot(data_in[:, :-1], w_trained_t)))
        # rho = np.abs(np.amin(np.dot(data_in[:, :-1], w_trained_t)))
        # Calculate the norm of w
        w_norm = np.linalg.norm(w_trained)
        # Calculate the theoretical bound
        t_max = (r ** 2) * (w_norm ** 2) / (rho ** 2)
        # Calculate the difference between the theoretical bound and the actual number of iterations and store it
        num_iters[i] = iterations
        bounds_minus_ni[i] = t_max - iterations

    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    main()
