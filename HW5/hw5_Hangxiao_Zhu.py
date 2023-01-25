#!/usr/bin/python3
# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use

    # Get number of training examples
    N = X_train.shape[0]

    # Initialize weights
    weights = np.ones(N) / N

    # Initialize epsilon_t
    epsilon = np.zeros(n_trees)

    # Initialize alpha_t
    alpha = np.zeros(n_trees)

    # Initialize decision stumps
    g = np.zeros(n_trees, dtype=object)

    # Initialize training and testing error
    train_error_array = np.zeros(n_trees)
    test_error_array = np.zeros(n_trees)

    # Iterate through trees
    for t in range(n_trees):
        # Learn decision stumps using information gain as the weak learner
        # Use the weights to weight the training examples
        g[t] = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        g[t].fit(X_train, y_train, sample_weight=weights)

        # Calculate epsilon_t
        epsilon[t] = np.sum(weights * (g[t].predict(X_train) != y_train))

        # Calculate alpha_t
        alpha[t] = np.log((1 - epsilon[t]) / epsilon[t]) / 2

        # Update weights and normalize
        weights = weights * np.exp(-alpha[t] * y_train * g[t].predict(X_train))
        weights = weights / np.sum(weights)

        # Calculate training and testing error
        y_train_pred = np.sign(np.sum(np.array([g[i].predict(X_train) for i in range(t + 1)]).T * alpha[:t + 1], axis=1))
        y_test_pred = np.sign(np.sum(np.array([g[i].predict(X_test) for i in range(t + 1)]).T * alpha[:t + 1], axis=1))
        train_error_array[t] = np.sum(y_train_pred != y_train) / N
        test_error_array[t] = np.sum(y_test_pred != y_test) / y_test.shape[0]

    # Plot training and testing error
    plt.plot(range(1, n_trees + 1), train_error_array, label='Training Error')
    plt.plot(range(1, n_trees + 1), test_error_array, label='Testing Error')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    # Calculate training and testing error
    train_error = train_error_array[-1]
    test_error = test_error_array[-1]

    return train_error, test_error


def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_trees = 200

    # Split data
    # The first only includes digits classified as 1 or 3
    # The second only includes digits classified as 3 or 5
    # The first integer in each row is the label
    X_train_one_three = og_train_data[np.logical_or(og_train_data[:, 0] == 1, og_train_data[:, 0] == 3), 1:]
    y_train_one_three = og_train_data[np.logical_or(og_train_data[:, 0] == 1, og_train_data[:, 0] == 3), 0]
    X_test_one_three = og_test_data[np.logical_or(og_test_data[:, 0] == 1, og_test_data[:, 0] == 3), 1:]
    y_test_one_three = og_test_data[np.logical_or(og_test_data[:, 0] == 1, og_test_data[:, 0] == 3), 0]

    X_train_three_five = og_train_data[np.logical_or(og_train_data[:, 0] == 3, og_train_data[:, 0] == 5), 1:]
    y_train_three_five = og_train_data[np.logical_or(og_train_data[:, 0] == 3, og_train_data[:, 0] == 5), 0]
    X_test_three_five = og_test_data[np.logical_or(og_test_data[:, 0] == 3, og_test_data[:, 0] == 5), 1:]
    y_test_three_five = og_test_data[np.logical_or(og_test_data[:, 0] == 3, og_test_data[:, 0] == 5), 0]

    # Replace 3 with -1
    y_train_one_three[y_train_one_three == 3] = -1
    y_test_one_three[y_test_one_three == 3] = -1
    y_train_three_five[y_train_three_five == 3] = -1
    y_test_three_five[y_test_three_five == 3] = -1

    # Replace 5 with 1
    y_train_three_five[y_train_three_five == 5] = 1
    y_test_three_five[y_test_three_five == 5] = 1

    # Run AdaBoost
    train_error_one_three, test_error_one_three = \
        adaboost_trees(X_train_one_three, y_train_one_three, X_test_one_three, y_test_one_three, num_trees)
    train_error_three_five, test_error_three_five = \
        adaboost_trees(X_train_three_five, y_train_three_five, X_test_three_five, y_test_three_five, num_trees)

    print('Training error for digits 1 and 3: {}'.format(train_error_one_three))
    print('Testing error for digits 1 and 3: {}'.format(test_error_one_three))
    print('Training error for digits 3 and 5: {}'.format(train_error_three_five))
    print('Testing error for digits 3 and 5: {}'.format(test_error_three_five))


if __name__ == "__main__":
    main_hw5()
