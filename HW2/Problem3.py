import numpy as np


def main():
    # Assume that the input dimension is one
    # Assume that the input variable x is uniformly distributed in [-1, 1]
    # The dataset consists of 2 points {x1, x2} and assume the target function is f(x) = x^2
    # Thus, the full dataset is D = {(x1, x1^2), (x2, x2^2)}
    # The learning algorithm returns the line fitting these two points as g
    # (Hypothesis consists of functions of the form h(x) = ax + b)