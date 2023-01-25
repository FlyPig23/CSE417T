#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function

    # Repeatedly uniformly sample N points from D with replacement to obtain num_bags bootstrapped datasets
    # Use sklearn.tree.DecisionTreeClassifier to learn a hypothesis from each of the bootstrapped datasets
    # 1. For each bag, learn a fully grown tree and use information gain as the split criterion
    # 2. Use the majority vote of the hypotheses as the final hypothesis
    # 3. Use the out-of-bag data to estimate the out-of-bag error
    bootstrapped_indices = np.random.choice(X_train.shape[0], (num_bags, X_train.shape[0]), replace=True)
    # Bootstrap each dataset and learn a hypothesis
    hypotheses = []
    oob_errors = np.zeros(num_bags)
    oob_indices = np.ones((num_bags, X_train.shape[0]), dtype=bool)
    oob_points = np.zeros(X_train.shape[0], dtype=bool)
    for i in range(num_bags):
        # Bootstrap the dataset
        X_train_bag = X_train[bootstrapped_indices[i], :]
        y_train_bag = y_train[bootstrapped_indices[i]]
        # Learn a hypothesis
        g = DecisionTreeClassifier(criterion='entropy')
        g.fit(X_train_bag, y_train_bag)
        hypotheses.append(g)
        # Determine whether a point is in a bootstrapped dataset
        # If a point is not in a bootstrapped dataset, it is out-of-bag
        # Use the final hypothesis to predict the labels of the out-of-bag data
        # Compute the out-of-bag error
        oob_indices[i, bootstrapped_indices[i]] = False
        oob_points = np.logical_or(oob_points, oob_indices[i])
        # Calculate the aggregation of hypothesis that x_n is out-of-bag
        G_oob = np.zeros(y_train.shape)
        for j in range(i + 1):
            G_oob[oob_indices[j]] += hypotheses[j].predict(X_train[oob_indices[j]])
        G_oob = np.sign(G_oob)
        oob_errors[i] = np.sum(G_oob[oob_points] != y_train[oob_points]) / np.sum(oob_points)

    # Aggregate the hypotheses using majority vote
    # Use the final hypothesis to predict the labels of the test data
    y_pred = np.zeros(y_test.shape)
    for i in range(num_bags):
        y_pred += hypotheses[i].predict(X_test)
    y_pred = np.sign(y_pred / num_bags)

    # Compute the out-of-bag classification error of the final learned ensemble
    out_of_bag_error = oob_errors[-1]

    # Compute the classification error of the final learned ensemble on test data
    test_error = np.sum(y_test != y_pred) / y_test.shape[0]

    # Plot the out-of-bag error as a function of the number of bags from 1 to num_bags
    plt.plot(np.arange(1, num_bags + 1), oob_errors)
    plt.xlabel('Number of bags')
    plt.ylabel('Out-of-bag error')
    plt.show()

    return out_of_bag_error, test_error


def main_hw4():

    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_bags = 200

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

    # Run bagged trees
    out_of_bag_error_one_three, test_error_one_three = \
        bagged_trees(X_train_one_three, y_train_one_three, X_test_one_three, y_test_one_three, num_bags)
    out_of_bag_error_three_five, test_error_three_five = \
        bagged_trees(X_train_three_five, y_train_three_five, X_test_three_five, y_test_three_five, num_bags)

    # Learn a single decision tree model from the entire training dataset
    # Use the learned model to predict the labels of the test data
    # Compute the test classification error of the final learned ensemble
    g_1 = DecisionTreeClassifier(criterion='entropy')
    g_1.fit(X_train_one_three, y_train_one_three)
    y_pred_one_three = g_1.predict(X_test_one_three)
    test_error_one_three_single = np.sum(y_test_one_three != y_pred_one_three) / y_test_one_three.shape[0]

    g_2 = DecisionTreeClassifier(criterion='entropy')
    g_2.fit(X_train_three_five, y_train_three_five)
    y_pred_three_five = g_2.predict(X_test_three_five)
    test_error_three_five_single = np.sum(y_test_three_five != y_pred_three_five) / y_test_three_five.shape[0]

    # Print results
    print('Out-of-bag error for digits 1 and 3: {}'.format(out_of_bag_error_one_three))
    print('Test error for digits 1 and 3: {}'.format(test_error_one_three))
    print('Test error for digits 1 and 3 (single tree): {}'.format(test_error_one_three_single))
    print('Out-of-bag error for digits 3 and 5: {}'.format(out_of_bag_error_three_five))
    print('Test error for digits 3 and 5: {}'.format(test_error_three_five))
    print('Test error for digits 3 and 5 (single tree): {}'.format(test_error_three_five_single))


if __name__ == "__main__":
    main_hw4()
