import numpy as np
import matplotlib.pyplot as plt


def flip_coin(times):
    return np.random.choice([0, 1], size=times)

def flip_coins(number_of_coins, number_of_flips_each_coin, repeat_times):
    v1 = np.zeros(repeat_times)
    vrand = np.zeros(repeat_times)
    vmin = np.zeros(repeat_times)
    for i in range(repeat_times):
        result = np.zeros((number_of_coins, number_of_flips_each_coin))
        crand = np.random.choice(number_of_coins)
        for j in range(number_of_coins):
            result[j] = flip_coin(number_of_flips_each_coin)

        # Count c1 heads fraction
        v1[i] = np.sum(result[0]) / number_of_flips_each_coin
        # Count crand heads fraction
        vrand[i] = np.sum(result[crand]) / number_of_flips_each_coin
        # Count cmin heads fraction
        cmin = np.argmin(np.sum(result, axis=1))
        vmin[i] = np.sum(result[cmin]) / number_of_flips_each_coin

    return v1, vrand, vmin


def bad_experiment_est(v1, vrand, vmin, number_of_flips_each_coin):
    # Calculate the probability of bad experiment as a function of epsilon
    epsilons = np.linspace(0, 0.5, 100)
    v1_bad = np.zeros(epsilons.shape)
    vrand_bad = np.zeros(epsilons.shape)
    vmin_bad = np.zeros(epsilons.shape)
    hoeffding_bound = np.zeros(epsilons.shape)
    for i, epsilon in enumerate(epsilons):
        # Calculate the Hoeffding bound
        hoeffding_bound[i] = 2 * np.exp(-2 * number_of_flips_each_coin * (epsilon ** 2))
        # Calculate the probability of bad experiment
        v1_bad[i] = np.sum(np.abs(v1 - 0.5) > epsilon) / v1.shape[0]
        vrand_bad[i] = np.sum(np.abs(vrand - 0.5) > epsilon) / vrand.shape[0]
        vmin_bad[i] = np.sum(np.abs(vmin - 0.5) > epsilon) / vmin.shape[0]

    plt.plot(epsilons, v1_bad, label="v1")
    plt.plot(epsilons, vrand_bad, label="vrand")
    plt.plot(epsilons, vmin_bad, label="vmin")
    plt.plot(epsilons, hoeffding_bound, label="Hoeffding bound")
    plt.title("Probability of bad experiment")
    plt.xlabel("Epsilon")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()


def main():
    v1, vrand, vmin = flip_coins(1000, 10, 1000)
    plt.hist(v1, bins=100)
    plt.title("Histogram of v1")
    plt.xlabel("Heads fraction")
    plt.ylabel("Count")
    plt.show()

    plt.hist(vrand, bins=100)
    plt.title("Histogram of vrand")
    plt.xlabel("Heads fraction")
    plt.ylabel("Count")
    plt.show()

    plt.hist(vmin, bins=100)
    plt.title("Histogram of vmin")
    plt.xlabel("Heads fraction")
    plt.ylabel("Count")
    plt.show()

    bad_experiment_est(v1, vrand, vmin, 10)


if __name__ == "__main__":
    main()
