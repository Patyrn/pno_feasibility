import csv
import math
import os
from functools import partial

import numpy as np

from EnergyDataUtil import get_energy_data
from KnapsackSolver import get_opt_params_knapsack
from Solver import get_optimal_average_objective_value, get_optimization_objective
from Utils import get_train_test_split
import matplotlib.pyplot as plt
import multiprocessing as mp


def get_objective(pred_Y, weights, Y=None, opt_params=None, pool=None):
    if Y is None:
        Y = pred_Y
    map_func = partial(get_objective_worker, opt_params=opt_params)
    iter = zip([pred_Y], [Y], [weights])
    # print('ho',Y, weights)
    objective_values = pool.starmap(map_func, iter)
    optimal_objective_values, solutions = zip(*objective_values)

    optimal_objective_values = np.concatenate(optimal_objective_values)

    return optimal_objective_values


def get_objective_worker(pred_Y, Y, weights, opt_params):
    # parallel computing will process one set a time so predicted solutions should be indexed to get rid of unncessary 2d list.

    optimal_average_objective_value, solutions = get_optimization_objective(pred_Y=pred_Y, Y=Y, weights=weights,
                                                                            opt_params=opt_params)
    return optimal_average_objective_value, solutions[0]


def get_noisy_set(problem_set, noise_percentage, method = "percentage"):
    noise = np.zeros(len(problem_set))

    for i, coefficient in enumerate(problem_set):
        if method == "percentage":
            noise[i] = np.random.normal(0, coefficient * noise_percentage / 100)
        elif method == "incremental":
            noise[i] = np.random.normal(0, noise_percentage)
    noisy_set = problem_set + noise.reshape(problem_set.shape)
    return noisy_set


def save_sims_results(file_name, noise_index, values):
    file_path = get_file_path(filename=file_name)
    with open(file_path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerow([values])


def get_file_path(filename, folder_path='sim_results'):
    """
    Constructs filepath. dataset is expected to be in the "data" folder
    :param filename:
    :return:
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, folder_path, filename)
    return file_path


def noise_test(capacities=None, n_iter=100, kfold=0, noise_start=50, noise_end=100, noise_step=5):
    if capacities is None:
        capacities = [12, 24, 48, 72, 96, 120, 144, 168, 196, 220]
    dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False,
                              kfold=kfold, noise_level=0)
    problem_sets = dataset.get('benchmarks_Y')
    weights_sets = dataset.get('benchmarks_weights')
    noise_range = range(noise_start, noise_end + 1, noise_step)

    true_objectives_list = [np.zeros(len(problem_sets)) for c in capacities]

    mypool = mp.Pool(processes=4)

    for c_index, (c, true_objectives) in enumerate(zip(capacities, true_objectives_list)):
        opt_params = get_opt_params_knapsack(capacity=c)
        true_objectives_list[c_index] = get_objective(problem_sets, weights_sets, opt_params=opt_params, pool=mypool)
        # complete true_objective lists

    noisy_regrets_list = []
    ref_problem_set = [[problem_set for i in range(n_iter)] for problem_set in problem_sets]
    for n in range(n_iter):
        for c_index, c in enumerate(capacities):
            opt_params = get_opt_params_knapsack(capacity=c)
            noisy_regrets = np.zeros(len(noise_range))
            for noise_index, noise in enumerate(noise_range):
                noisy_objective_samples = np.zeros((len(problem_sets), n_iter))
                noisy_sets = []
                noisy_weights = []
                for set_index, (problem_set, weights) in enumerate(zip(problem_sets, weights_sets)):
                    noisy_set = get_noisy_set(problem_set, noise)
                    noisy_sets.append(noisy_set)
                    noisy_weights.append(weights)
                print("Capacity: {}, Noise: {}%, n_iter: {}".format(c, noise, n))
                noisy_objectives = np.array(
                    get_objective(pred_Y=noisy_sets, Y=problem_sets, weights=noisy_weights,
                                  opt_params=opt_params, pool=mypool))

                noisy_regret = np.mean((true_objectives_list[c_index] - noisy_objectives))
                file_name = "knapsack_c{}_n{}.csv".format(c, noise)
                save_sims_results(file_name=file_name, noise_index=noise, values=noisy_regret)
                noisy_regrets[noise_index] = np.mean((true_objectives_list[c_index] - noisy_objective_samples.T).T)
            noisy_regrets_list.append(noisy_regrets)
    # columns = 2
    # rows = int(math.ceil(len(noisy_regrets_list) / columns))
    # fig, axs = plt.subplots(rows, columns)
    # for ax, regrets, c in zip(axs.flat, noisy_regrets_list, capacities):
    #     ax.plot(regrets)
    #     ax.set_title("C: {}".format(c))
    # plt.show()

    # need to mean noisy regret samples and record it for each problem set for all capacity and noise ranges!!!!!
    print('finished')

def noise_test_incremental(capacities=None, n_iter=100, kfold=0, noise_start=50, noise_end=100, noise_step=5):
    if capacities is None:
        capacities = [12, 24, 48, 72, 96, 120, 144, 168, 196, 220]
    dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False,
                              kfold=kfold, noise_level=0)
    problem_sets = dataset.get('benchmarks_Y')
    weights_sets = dataset.get('benchmarks_weights')
    noise_range = range(noise_start, noise_end + 1, noise_step)

    true_objectives_list = [np.zeros(len(problem_sets)) for c in capacities]

    mypool = mp.Pool(processes=4)

    for c_index, (c, true_objectives) in enumerate(zip(capacities, true_objectives_list)):
        opt_params = get_opt_params_knapsack(capacity=c)
        true_objectives_list[c_index] = get_objective(problem_sets, weights_sets, opt_params=opt_params,
                                                      pool=mypool)
        # complete true_objective lists

    noisy_regrets_list = []
    ref_problem_set = [[problem_set for i in range(n_iter)] for problem_set in problem_sets]
    for n in range(n_iter):
        for c_index, c in enumerate(capacities):
            opt_params = get_opt_params_knapsack(capacity=c)
            noisy_regrets = np.zeros(len(noise_range))
            for noise_index, noise in enumerate(noise_range):
                noisy_objective_samples = np.zeros((len(problem_sets), n_iter))
                noisy_sets = []
                noisy_weights = []
                for set_index, (problem_set, weights) in enumerate(zip(problem_sets, weights_sets)):
                    noisy_set = get_noisy_set(problem_set, noise, "incremental")
                    noisy_sets.append(noisy_set)
                    noisy_weights.append(weights)
                print("Capacity: {}, Noise: {}%, n_iter: {}".format(c, noise, n))
                noisy_objectives = np.array(
                    get_objective(pred_Y=noisy_sets, Y=problem_sets, weights=noisy_weights,
                                  opt_params=opt_params, pool=mypool))

                noisy_regret = np.mean((true_objectives_list[c_index] - noisy_objectives))
                file_name = "incremental_knapsack_c{}_n{}.csv".format(c, noise)
                save_sims_results(file_name=file_name, noise_index=noise, values=noisy_regret)
                noisy_regrets[noise_index] = np.mean((true_objectives_list[c_index] - noisy_objective_samples.T).T)
            noisy_regrets_list.append(noisy_regrets)
    # columns = 2
    # rows = int(math.ceil(len(noisy_regrets_list) / columns))
    # fig, axs = plt.subplots(rows, columns)
    # for ax, regrets, c in zip(axs.flat, noisy_regrets_list, capacities):
    #     ax.plot(regrets)
    #     ax.set_title("C: {}".format(c))
    # plt.show()


if __name__ == "__main__":
    # capacities = [220]
    capacities = None
    noise_test(kfold=0, capacities=capacities, n_iter=100, noise_start=0, noise_end=101, noise_step=1)

    # noise_test_incremental(kfold=0, capacities=capacities, n_iter=100, noise_start=0, noise_end=1000, noise_step=10)
