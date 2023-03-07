import math

from matplotlib import pyplot as plt

from Utils import read_file
import numpy as np

def file_name_creator(prefix, capacity,noise, method = "percentage"):
    if method == "incremental":
        str = "incremental_" + prefix + "_c{}_n{}.csv".format(capacity, noise)
    elif method == "percentage":
        str = prefix + "_c{}_n{}.csv".format(capacity,noise)
    return str

def plot_sim(prefix = 'knapsack', capacities = None, noises= None, method = "percentage"):
    if capacities is None:
        capacities = [12, 24, 48, 72, 96, 120, 144, 168, 196, 220]
    if noises is None:
        noises = [noise for noise in range(0,101,10)]
    capacities_list = []
    for c in capacities:
        noise_regrets = []
        for noise in noises:
            file_name = file_name_creator(prefix, c, noise, method)
            data = np.array(read_file(file_name,"PnOFeasibility/sim_results")).astype(float)
            noise_regrets.append(np.mean(data))
        capacities_list.append(noise_regrets)

    columns = 2
    rows = int(math.ceil(len(capacities_list) / columns))
    fig, axs = plt.subplots(rows, columns)
    for ax, regrets, c in zip(axs.flat, capacities_list, capacities):
        ax.plot(regrets)
        ax.set_title("C: {}".format(c))
    plt.show()

if __name__ == "__main__":
    noises = [noise for noise in range(0, 101)]
    # plot_sim(capacities= None, noises=noises, method = "percentage")
    #
    incremental_noises = [noise for noise in range(0, 1000,10)]
    plot_sim(capacities= None, noises=incremental_noises, method = "incremental")