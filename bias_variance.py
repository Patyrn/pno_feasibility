import copy
import multiprocessing
from functools import partial

from tkinter import *

import numpy as np
from matplotlib import cm
from matplotlib.patches import Rectangle
from sklearn import linear_model
from mlxtend.evaluate import bias_variance_decomp
import multiprocessing as mp

from EnergyDataUtil import get_energy_data
from KnapsackSolver import get_opt_params_knapsack
from Solver import get_optimization_objective, get_optimal_average_objective_value
from Utils import get_train_test_split
from bias_variance_decomp import bias_variance_decomp_pno

import matplotlib.pyplot as plt

# def get_bias_var_problem_sets(model, x,y):

def prepare_icon_dataset(kfold=0, is_shuffle=False):
    dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False,
                              kfold=kfold, noise_level=0)
    # combine weights with X first
    # may need to split weights

    train_set, test_set = get_train_test_split(dataset, random_seed=0, is_shuffle=is_shuffle)

    X_train = train_set.get('X').T
    Y_train = train_set.get('Y').T

    X_val = X_train[0:2880, :]
    Y_val = Y_train[0:2880]

    X_train = X_train[2880:, :]
    Y_train = Y_train[2880:, :]

    benchmarks_train_X = train_set.get('benchmarks_X')
    benchmarks_train_Y = train_set.get('benchmarks_Y')
    benchmarks_weights_train = train_set.get('benchmarks_weights')

    benchmarks_val_X = benchmarks_train_X[0:60]
    benchmarks_val_Y = benchmarks_train_Y[0:60]
    benchmarks_weights_val = benchmarks_weights_train[0:60]

    benchmarks_train_X = benchmarks_train_X[60:]
    benchmarks_train_Y = benchmarks_train_Y[60:]
    benchmarks_weights_train = benchmarks_weights_train[60:]

    train_dict = {"X": X_train, "Y": Y_train, "X_sets": benchmarks_train_X, "Y_sets": benchmarks_train_Y,
                  "Weights_sets": benchmarks_weights_train}

    val_dict = {"X": X_val, "Y": Y_val, "X_sets": benchmarks_val_X, "Y_sets": benchmarks_val_Y,
                "Weights_sets": benchmarks_weights_val}

    test_dict = {"X": test_set.get('X').T, "Y": test_set.get('Y').T, "X_sets": test_set.get('benchmarks_X'),
                 "Y_sets": test_set.get('benchmarks_Y'), "Weights_sets": test_set.get('benchmarks_weights')}

    return train_dict, val_dict, test_dict


def scikit_get_regret(model, X, Y, weights, opt_params=None, pool=None):

    pred_Ys = []
    for x in X:
        pred_Ys.append(model.predict(x))

    if pool is None:
        objective_values_predicted_items = get_optimization_objective(Y=Y, pred_Y=pred_Ys,
                                                                      weights=weights,
                                                                      opt_params=opt_params,
                                                                      )
        optimal_objective_values = get_optimal_average_objective_value(Y=Y, weights=weights,
                                                                       opt_params=opt_params,
                                                                       )

        regret = np.median(optimal_objective_values - objective_values_predicted_items)
    else:
        map_func = partial(get_regret_worker, opt_params=opt_params)
        iter = zip(Y, pred_Ys, weights)
        objective_values = pool.starmap(map_func, iter)
        objective_values_predicted_items, predicted_solutions, optimal_objective_values, solutions = zip(*objective_values)
        predicted_solutions = [solution for solution in predicted_solutions]

        optimal_objective_values = np.concatenate(optimal_objective_values)
        objective_values_predicted_items = np.concatenate(objective_values_predicted_items)
        regret = optimal_objective_values - objective_values_predicted_items

    return regret, optimal_objective_values, list(predicted_solutions), list(solutions)

def get_regret_worker(Y, pred_Ys, weights, opt_params):
    #parallel computing will process one set a time so predicted solutions should be indexed to get rid of unncessary 2d list.
    print('ho', Y, weights)
    average_objective_value_with_predicted_items, predicted_solutions = get_optimization_objective(Y=[Y], pred_Y=[pred_Ys],
                                                                              weights=[weights],
                                                                              opt_params=opt_params,
                                                                              )
    optimal_average_objective_value, solutions = get_optimal_average_objective_value(Y=[Y], weights=[weights],
                                                                          opt_params=opt_params,
                                                                          )
    return average_objective_value_with_predicted_items, predicted_solutions[0], optimal_average_objective_value, solutions[0]


def get_bias_var(kfold = 0, is_shuffle = False):
    train_dict, val_dict, test_dict = prepare_icon_dataset(kfold= kfold, is_shuffle= is_shuffle)

    X_train_sets = train_dict.get("X_sets")
    Y_train_sets = train_dict.get("Y_sets")

    X_train = train_dict.get("X")
    Y_train = train_dict.get("Y")


    X_test_sets = test_dict.get("X_sets")
    Y_test_sets = test_dict.get("Y_sets")
    weights_test_sets = test_dict.get("Weights_sets")
    X_test = test_dict.get("X")
    Y_test = test_dict.get("Y")

    mypool = multiprocessing.Pool(processes=4)

    scikit_regression = linear_model.Ridge()
    # scikit_regression = linear_model.Ridge().fit(X_train, Y_train)

    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        scikit_regression, X_train_sets, Y_train_sets, X_test, Y_test.flatten(),loss='mse', random_seed=42)

    capacities = [12,24,48,72,96,120,144,168,196,220]
    scikit_regression.fit(X_train, Y_train)
    opt_params_list = [get_opt_params_knapsack(capacity=capacity) for capacity in capacities]
    regrets = []
    for opt_param in opt_params_list:
        regret, __, predicted_solutions, solutions = scikit_get_regret(scikit_regression, X_test_sets, Y_test_sets, weights_test_sets, opt_params=opt_param, pool=mypool)
        regrets.append(regret)

    regret_str = "Regrets: "
    for regret, capacity in zip(regrets,capacities):
        regret_str += ", {} ({})".format(round(regret,3), capacity)
    print("Kfold: {}, Shuffle: {}".format(kfold, is_shuffle))
    print("avg_expected_loss: {}, avg_bias: {}, avg_var: {}".format(avg_expected_loss, avg_bias, avg_var))
    print(regret_str)
    # LINEAR REGRESSION ONE LAYER

def bias_variance_sets(capacities = None, kfold = 0, is_shuffle = False):
    train_dict, val_dict, test_dict = prepare_icon_dataset(kfold= kfold, is_shuffle= is_shuffle)

    X_train_sets = train_dict.get("X_sets")
    Y_train_sets = train_dict.get("Y_sets")

    X_train = train_dict.get("X")
    Y_train = train_dict.get("Y")


    X_test_sets = test_dict.get("X_sets")
    Y_test_sets = test_dict.get("Y_sets")
    weights_test_sets = test_dict.get("Weights_sets")
    X_test = test_dict.get("X")
    Y_test = test_dict.get("Y")


    mypool = multiprocessing.Pool(processes=8)
    scikit_regression = linear_model.Ridge()
    # scikit_regression = linear_model.Ridge().fit(X_train, Y_train)

    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        scikit_regression, X_train, Y_train, X_test, Y_test.flatten(),loss='mse', random_seed=42)

    if capacities is None:
        capacities = [12,24,48,72,96,120,144,168,196,220]
    scikit_regression.fit(X_train, Y_train)
    opt_params_list = [get_opt_params_knapsack(capacity=capacity) for capacity in capacities]
    regrets = []
    for opt_param in opt_params_list:
        regret, __, __,__= scikit_get_regret(scikit_regression, X_test_sets, Y_test_sets, weights_test_sets, opt_params=opt_param, pool=mypool)
        regrets.append(regret)

    regret_str = "Regrets: "
    for regret, capacity in zip(regrets,capacities):
        regret_str += ", {} ({})".format(round(np.median(regret),3), capacity)
    print("Kfold: {}, Shuffle: {}".format(kfold, is_shuffle))
    print("avg_expected_loss: {}, avg_bias: {}, avg_var: {}".format(avg_expected_loss, avg_bias, avg_var))
    print(regret_str)

    print("---------------PNO-----------------")

    # LINEAR REGRESSION ONE LAYER
    avg_expected_loss_set, avg_bias_set, avg_var_set = bias_variance_decomp_pno(scikit_regression, X_train, Y_train, X_test_sets, Y_test_sets, loss='mse', random_seed=42)
    print("Kfold: {}, Shuffle: {}".format(kfold, is_shuffle))
    for this_regret, opt_param in zip(regrets, opt_params_list):
        for regret, avg_bias, avg_var, avg_expected_loss in zip(this_regret,avg_expected_loss_set,avg_bias_set,avg_var_set):
            print("regret: {}, bias: {}, var: {}, loss: {}".format(round(regret,3), round(avg_expected_loss, 3), round(avg_bias, 3), round(avg_var, 3)))

    fig = plt.figure()


    for i,capacity in enumerate(capacities):
        # ax = fig.add_subplot(5, 2, i + 1, projection = '3d')
        ax = fig.add_subplot(5, 2, i+1)
        ax.set_title('C: {}'.format(capacity))
        cmap = cm.get_cmap()
        color = cmap(np.log(regrets[i]))[..., :3]

        plt.scatter(avg_bias_set, avg_var_set, c=color, alpha=0.5)

        # ax.scatter3D(avg_bias_set, avg_var_set, regrets[i], c=regrets[i], cmap='Greens')
        ax.set_title("Capacity {}".format(capacity))
        ax.set_xlabel('bias')
        ax.set_ylabel('var')

    plt.show()

def plot_impactful_sets(capacity, kfold, is_shuffle):
    train_dict, val_dict, test_dict = prepare_icon_dataset(kfold= kfold, is_shuffle= is_shuffle)

    X_train_sets = train_dict.get("X_sets")
    Y_train_sets = train_dict.get("Y_sets")

    X_train = train_dict.get("X")
    Y_train = train_dict.get("Y")


    X_test_sets = test_dict.get("X_sets")
    Y_test_sets = test_dict.get("Y_sets")
    weights_test_sets = test_dict.get("Weights_sets")
    X_test = test_dict.get("X")
    Y_test = test_dict.get("Y")

    mypool = multiprocessing.Pool(processes=8)

    scikit_regression = linear_model.Ridge()
    scikit_regression.fit(X_train, Y_train)
    opt_param = get_opt_params_knapsack(capacity=capacity)
    regret, __, predicted_solutions, solutions = scikit_get_regret(scikit_regression, X_test_sets, Y_test_sets, weights_test_sets, opt_params=opt_param, pool=mypool)

    pred_Ys = []
    for x in X_test_sets:
        pred_Ys.append(scikit_regression.predict(x))


    sorted_benchmarks = copy.deepcopy(Y_test_sets)
    sorted_regret = sorted(regret)
    sorted_benchmarks = [x for _, x in sorted(zip(regret,sorted_benchmarks), key=lambda pair: pair[0])]
    sorted_preds = [x for _, x in sorted(zip(regret,pred_Ys), key=lambda pair: pair[0])]
    sorted_solutions = [x for _, x in sorted(zip(regret,solutions), key=lambda pair: pair[0])]
    sorted_predicted_solutions = [x for _, x in sorted(zip(regret,predicted_solutions), key=lambda pair: pair[0])]

    for pred, set, predicted_solution, solution in zip((sorted_preds), (sorted_benchmarks), (sorted_predicted_solutions), (sorted_solutions)):
    # for pred, set, predicted_solution, solution in zip((sorted_preds), (sorted_benchmarks),
    #                                                        (sorted_predicted_solutions),
    #                                                        (sorted_solutions)):
        plot_problem_set(pred = pred, Y= set, predicted_solution=predicted_solution, solution=solution)

def plot_problem_set(pred, Y, predicted_solution= None, solution= None):
    index = [i for i in range(len(Y))]
    alpha = 0.5
    fig = plt.figure()
    ax = plt.gca()
    plt.scatter(index, y=Y, alpha=alpha, c='b')
    plt.scatter(index, y=pred, alpha=alpha, c='r')
    rectangle_width = 1
    rectange_height = 50
    width_offset = rectangle_width / 2
    height_offset = rectange_height / 2
    if solution is not None:
        for item in solution:
            ax.add_patch(Rectangle((int(item) - width_offset, Y[int(item)] - height_offset), rectangle_width, rectange_height , facecolor="b", alpha=alpha))
    if predicted_solution is not None:
        for item in predicted_solution:
            ax.add_patch(Rectangle((int(item) - width_offset, Y[int(item)] - height_offset), rectangle_width, rectange_height, facecolor="r",alpha=alpha))
    plt.show()


if __name__ == "__main__":
    # for kfold in range(5):
    #     get_bias_var(kfold=kfold, is_shuffle=True)

    capacities = None
    # capacities = [12]

    # bias_variance_sets(capacities, kfold=1, is_shuffle=True)


    plot_impactful_sets(capacity=12, kfold=0, is_shuffle=False)