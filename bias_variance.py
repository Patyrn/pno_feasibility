import copy
import math
import multiprocessing
import os.path
import time
from functools import partial

from tkinter import *
import seaborn as sns
import numpy as np
import torch
from matplotlib import cm
from matplotlib.patches import Rectangle
from numpy import sqrt
from sklearn import linear_model
from mlxtend.evaluate import bias_variance_decomp
import multiprocessing as mp

from EnergyDataUtil import get_energy_data
from IconEasySolver import get_icon_instance_params
from KnapsackSolver import get_opt_params_knapsack
from ReLu_DNL.ReLu_DNL import relu_ppo
from Solver import get_optimization_objective, get_optimal_average_objective_value
from Utils import get_train_test_split
from bias_variance_decomp import bias_variance_decomp_pno

import matplotlib.pyplot as plt

# def get_bias_var_problem_sets(model, x,y):
colour_palette = ["#7f0085", "#7f07c9", "#aeffde", "#00dd27", "#9b6415", "#fc6700", "#f9b09e"]


def prepare_icon_dataset(kfold=0, is_shuffle=False, unit_weight = False):
    dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=unit_weight,
                              kfold=kfold, noise_level=0, is_sorted=False)
    # combine weights with X first
    # may need to split weights

    train_set, test_set = get_train_test_split(dataset, random_seed=81, is_shuffle=is_shuffle)

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


def scikit_get_regret(model, X, Y, weights, opt_params=None, pool=None, pred_Ys = None):
    if pred_Ys is None:
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

        regret = optimal_objective_values - objective_values_predicted_items
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
    # print('ho', Y, weights)
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


    scikit_regression = linear_model.Ridge().fit(X_train, Y_train)
    opt_param = get_opt_params_knapsack(capacity=12)
    pred = scikit_regression.predict(X_test)
    error = np.mean((Y_test - pred)**2)
    std = np.std((Y_test - pred)**2)
    print("Regression error: {}, std: {}".format(error,std))
    regret, __, predicted_solutions, solutions = scikit_get_regret(scikit_regression, X_test_sets, Y_test_sets, weights_test_sets, opt_params=opt_param, pool=mypool)

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

    # plt.show()

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
    # sorted_regret = sorted(regret, reverse=True)
    is_reverse = True
    sorted_regret = regret
    sorted_benchmarks = [x for _, x in sorted(zip(sorted_regret,sorted_benchmarks), key=lambda pair: pair[0], reverse=is_reverse)]
    sorted_preds = [x for _, x in sorted(zip(sorted_regret,pred_Ys), key=lambda pair: pair[0], reverse=is_reverse)]
    sorted_solutions = [x for _, x in sorted(zip(sorted_regret,solutions), key=lambda pair: pair[0], reverse=is_reverse)]
    sorted_predicted_solutions = [x for _, x in sorted(zip(sorted_regret,predicted_solutions), key=lambda pair: pair[0], reverse=is_reverse)]

    for pred, set, predicted_solution, solution in zip((sorted_preds), (sorted_benchmarks), (sorted_predicted_solutions), (sorted_solutions)):
    # for pred, set, predicted_solution, solution in zip((sorted_preds), (sorted_benchmarks),
    #                                                        (sorted_predicted_solutions),
    #                                                        (sorted_solutions)):
        plot_problem_set(pred = pred, Y= set, predicted_solution=predicted_solution, solution=solution)

def plot_impactful_sets_dnl(capacity, kfold, is_shuffle):
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
    relu_path = os.path.join(os.path.dirname(__file__),'models/dnl22.pth')
    # relu_dnl = relu_ppo()
    relu_dnl = torch.load(relu_path)
    relu_dnl.eval()

    opt_param = get_opt_params_knapsack(capacity=capacity)
    dnl_regret, __, dnl_predicted_solutions, solutions = scikit_get_regret(scikit_regression, X_test_sets, Y_test_sets, weights_test_sets, opt_params=opt_param, pool=mypool)

    relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets, weights_test_sets, pool=mypool)

    pred_Ys_dnl = []
    pred_Ys_relu = []
    for x in X_test_sets:
        pred_Ys_dnl.append(scikit_regression.predict(x))
        pred_Ys_relu.append(relu_dnl.predict(x))


    sorted_benchmarks = copy.deepcopy(Y_test_sets)
    # sorted_regret = sorted(regret, reverse=True)
    is_reverse = False

    # regret
    #
    regret = relu_regret - dnl_regret
    sorted_regret = regret
    print("regret1",np.median(relu_regret) - np.median(dnl_regret))
    print("regret2",np.median(regret))
    # print("relu_regret",np.median(relu_regret))
    # print("reg_regret",np.median(dnl_regret))


    sorted_benchmarks = [x for _, x in sorted(zip(sorted_regret,sorted_benchmarks), key=lambda pair: pair[0], reverse=is_reverse)]
    sorted_preds_dnl = [x for _, x in sorted(zip(sorted_regret,pred_Ys_dnl), key=lambda pair: pair[0], reverse=is_reverse)]
    sorted_preds_relu = [x for _, x in
                        sorted(zip(sorted_regret, pred_Ys_relu), key=lambda pair: pair[0], reverse=is_reverse)]



    sorted_predicted_solutions_dnl = [x for _, x in sorted(zip(sorted_regret,dnl_predicted_solutions), key=lambda pair: pair[0], reverse=is_reverse)]

    sorted_predicted_solutions_relu = [x for _, x in sorted(zip(sorted_regret,relu_predicted_solutions), key=lambda pair: pair[0], reverse=is_reverse)]

    sorted_solutions = [x for _, x in
                        sorted(zip(sorted_regret, solutions), key=lambda pair: pair[0], reverse=is_reverse)]

    sorted_regret = [x for _, x in
                         sorted(zip(sorted_regret, sorted_regret), key=lambda pair: pair[0], reverse=is_reverse)]

    print(sorted_regret)


    plot_std_vs_regret(sorted_regret, sorted_preds_dnl,sorted_preds_relu, sorted_benchmarks, sorted_predicted_solutions_dnl, sorted_predicted_solutions_relu, sorted_solutions)

    for pred_dnl, pred_relu,  Y, predicted_solution_dnl,predicted_solution_relu, solution in zip((sorted_preds_dnl),(sorted_preds_relu), (sorted_benchmarks), (sorted_predicted_solutions_dnl), (sorted_predicted_solutions_relu), (sorted_solutions)):
    # for pred, set, predicted_solution, solution in zip((sorted_preds), (sorted_benchmarks),
    #                                                        (sorted_predicted_solutions),
    #                                                        (sorted_solutions)):
        plot_problem_set(pred_dnl = pred_dnl,pred_relu=pred_relu, Y= Y, predicted_solution_dnl=predicted_solution_dnl, predicted_solution_relu=predicted_solution_relu,
                         solution=solution)

def plot_std_vs_regret(sorted_regret, sorted_preds_dnl,sorted_preds_relu, sorted_benchmarks, sorted_predicted_solutions_dnl, sorted_predicted_solutions_relu, sorted_solutions):
    dnl_std = []
    relu_std = []
    normal_std = []

    for pred_dnl, pred_relu, Y, predicted_solution_dnl, predicted_solution_relu, solution in zip((sorted_preds_dnl),
                                                                                                 (sorted_preds_relu),
                                                                                                 (sorted_benchmarks), (
                                                                                                 sorted_predicted_solutions_dnl),
                                                                                                   (
                                                                                                 sorted_predicted_solutions_relu),
                                                                                                 (sorted_solutions)):
        dnl_std.append(np.std((pred_dnl)[:,0]))
        relu_std.append(np.std((pred_relu)[:,0]))
        normal_std.append(np.std(Y))



    alpha = 0.5
    fig = plt.figure()
    ax = plt.gca()
    std_diff = [x -y for x,y in zip(relu_std,dnl_std)]
    plt.scatter(sorted_regret, y=normal_std, alpha=alpha, c='b')
    plt.scatter(sorted_regret, y=dnl_std, alpha=alpha, c='r')
    plt.scatter(sorted_regret, y=relu_std, alpha=alpha, c='y')


    z = np.polyfit(sorted_regret, normal_std, 1)
    p = np.poly1d(z)
    plt.plot(sorted_regret, p(sorted_regret), "b-")


    z = np.polyfit(sorted_regret, dnl_std, 1)
    p = np.poly1d(z)
    plt.plot(sorted_regret, p(sorted_regret), "r-")

    z = np.polyfit(sorted_regret, relu_std, 1)
    p = np.poly1d(z)
    plt.plot(sorted_regret, p(sorted_regret), "y-")

    # plt.scatter(sorted_regret, y=dnl_std, alpha=alpha, c='r')
    # plt.scatter(sorted_regret, y=relu_std, alpha=alpha, c='y')

    # ax.set_title("dnl: {}, relu: {}".format(sum(Y[predicted_solution_dnl]),sum(Y[predicted_solution_relu])))
    # ax.legend(['true', 'reg', 'dnl'])
    ax.legend(['true', 'reg', 'dnl'])


    plt.figure()
    counts, bins = np.histogram(sorted_regret)
    plt.stairs(counts, bins)
    print('regret', np.median(sorted_regret))
    # plt.show()

def plot_problem_set(pred_dnl,pred_relu, Y, predicted_solution_dnl= None, predicted_solution_relu= None,solution= None):
    index = [i for i in range(len(Y))]
    alpha = 0.5
    fig = plt.figure()
    ax = plt.gca()
    Y = (Y-min(Y))/(max(Y)-min(Y))
    pred_dnl = (pred_dnl - min(pred_dnl)) / (max(pred_dnl) - min(pred_dnl))
    pred_relu = (pred_relu - min(pred_relu)) / (max(pred_relu) - min(pred_relu))


    plt.scatter(index, y=Y, alpha=alpha, c='b')
    plt.scatter(index, y=pred_dnl, alpha=alpha, c='r')
    plt.scatter(index, y=pred_relu, alpha=alpha, c='y')
    rectangle_width = max(Y)/1
    rectange_height = max(Y)/10
    width_offset = rectangle_width / 2
    height_offset = rectange_height / 2
    if solution is not None:
        for item in solution:
            ax.add_patch(Rectangle((int(item) - width_offset, Y[int(item)] - height_offset), rectangle_width, rectange_height , facecolor="b", alpha=alpha))
    if predicted_solution_dnl is not None:
        for item in predicted_solution_dnl:
            ax.add_patch(Rectangle((int(item) - width_offset, pred_dnl[int(item)] - height_offset), rectangle_width, rectange_height, facecolor="r",alpha=alpha))

    if predicted_solution_relu is not None:
        for item in predicted_solution_relu:
            ax.add_patch(Rectangle((int(item) - width_offset, pred_relu[int(item)] - height_offset), rectangle_width, rectange_height, facecolor="y",alpha=alpha))
    ax.set_title("reg: {}, relu: {}".format(sum(Y[predicted_solution_dnl]),sum(Y[predicted_solution_relu])))
    ax.legend(['true', 'reg', 'dnl'])
    # print("objective: {}, pred: {}".format(sum(Y[solution]),sum(Y[predicted_solution])))

    # plt.show()

def gen_icon_regret(n_iter = None, instance = 30):
    # dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True,
    #                           kfold=0, noise_level=0, is_sorted=False)
    #
    # benchmarks_X = dataset.get('benchmarks_X')
    # benchmarks_Y = dataset.get('benchmarks_Y')
    # benchmarks_weights = dataset.get('benchmarks_weights')
    #
    # benchmarks_Y = np.array(benchmarks_Y).reshape((789, -1))
    # benchmarks_mean = np.mean(benchmarks_Y, axis=0)
    # benchmarks_std = np.std(benchmarks_Y, axis=0)

    dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True,
                              kfold=0, noise_level=0, is_sorted=False)
    train_set, test_set = get_train_test_split(dataset, random_seed=81, is_shuffle=True)

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

    benchmarks_Y = test_set.get('benchmarks_Y')
    benchmarks_weights = test_set.get('benchmarks_weights')
    benchmarks_X = test_set.get('benchmarks_X')
    benchmarks_Y = np.array(benchmarks_Y).reshape((158, -1))
    benchmarks_mean = np.mean(benchmarks_Y, axis=0)
    benchmarks_std = np.std(benchmarks_Y, axis=0)

    # x = [x for x in range(48)]
    # plt.plot(x,benchmarks_mean, 'k', color='#CC4F1B')
    # plt.fill_between(x,benchmarks_mean - benchmarks_std, benchmarks_mean + benchmarks_std,
    #                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    # plt.show()
    script_dir = os.path.dirname(__file__)

    mean_val = 0
    std_val = 1000

    mean_step = 10
    std_step = 100
    if n_iter is None:
        n_iter = 10
    mean_sample_size = int(mean_val / mean_step) + 1
    std_sample_size = int(std_val / std_step) + 1
    noise_mean = np.linspace(0, mean_val, mean_sample_size)
    noise_std = np.linspace(0, std_val, std_sample_size)

    std_regrets = np.zeros(mean_sample_size)
    mean_regrets = np.zeros(std_sample_size)
    mypool = mp.Pool(processes=8)
    # capacities = [5,10,15,20,25,30,35,40]


    map_func = partial(gen_icon_worker_add,instance = instance,mean_sample_size=mean_sample_size,std_sample_size=std_sample_size,noise_mean=noise_mean,noise_std=noise_std,benchmarks_Y=benchmarks_Y,benchmarks_weights=benchmarks_weights,mean_val=mean_val,std_val=std_val)
    iter = range(n_iter)

    regrets = np.array(mypool.map(map_func, iter))


    f_path = os.path.join(script_dir,
                          'noise_test_var/additive/icon_add_i{}m{}std{}n{}.npy'.format(instance, mean_val, std_val,
                                                                                         n_iter))
    np.save(f_path, regrets)

    mean_val = 0
    std_val = 50

    mean_step = 10
    std_step = 5

    mean_sample_size = int(mean_val / mean_step) + 1
    std_sample_size = int(std_val / std_step) + 1
    noise_mean = np.linspace(0, mean_val, mean_sample_size)
    noise_std = np.linspace(0, std_val, std_sample_size)

    map_func = partial(gen_icon_worker_mult, instance = instance, mean_sample_size=mean_sample_size,
                       std_sample_size=std_sample_size, noise_mean=noise_mean, noise_std=noise_std,
                       benchmarks_Y=benchmarks_Y, benchmarks_weights=benchmarks_weights, mean_val=mean_val,
                       std_val=std_val)
    iter = range(n_iter)

    regrets = np.array(mypool.map(map_func, iter))

    f_path = os.path.join(script_dir,'noise_test_var/multiplicative/icon_mult_i{}m{}std{}n{}.npy'.format(instance, mean_val, std_val, n_iter))
    np.save(f_path, regrets)

    print("blalba")

def gen_icon_worker_mult(n_iter, instance,mean_sample_size,std_sample_size,noise_mean,noise_std,benchmarks_Y,benchmarks_weights,mean_val,std_val):
    opt_params = get_icon_instance_params(instance,folder_path='data/icon_instances/easy')
    regrets = np.zeros((mean_sample_size, std_sample_size))

    for i, mean in enumerate(noise_mean):
        for j, std in enumerate(noise_std):
            for benchmark_Y, weights in zip(benchmarks_Y, benchmarks_weights):
                print("i: {} n: {} mean: {} std: {}".format(instance, n_iter, mean, std))
                np.random.seed(int(time.time()))
                noisy_Y = benchmark_Y +  np.random.normal(mean, benchmark_Y* std/100)
                # noisy_Y = benchmark_Y + np.random.normal(mean, std)
                tov, predicted_sol, tov_opt, sol = get_regret_worker(benchmark_Y, noisy_Y, weights.astype(int),
                                                                     opt_params)
                regrets[i, j] = np.median(tov_opt - tov)
    return regrets
def gen_icon_worker_add(n_iter, instance , mean_sample_size, std_sample_size, noise_mean, noise_std, benchmarks_Y,
                    benchmarks_weights, mean_val, std_val):
    opt_params = get_icon_instance_params(instance, folder_path='data/icon_instances/easy')
    regrets = np.zeros((mean_sample_size, std_sample_size))
    script_dir = os.path.dirname(__file__)
    for i, mean in enumerate(noise_mean):
        for j, std in enumerate(noise_std):
            for benchmark_Y, weights in zip(benchmarks_Y, benchmarks_weights):
                np.random.seed(int(time.time()))
                print("i: {} n: {} mean: {} std: {}".format(instance, n_iter, mean, std))
                noisy_Y = benchmark_Y + np.random.normal(mean, std)
                tov, predicted_sol, tov_opt, sol = get_regret_worker(benchmark_Y, noisy_Y, weights.astype(int),
                                                                     opt_params)
                regrets[i, j] = np.median(tov_opt - tov)
    return regrets

def gen_MSE_regret(n_iter = None):
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False,
                              kfold=0, noise_level=0, is_sorted=False)

        benchmarks_X = dataset.get('benchmarks_X')
        benchmarks_Y = dataset.get('benchmarks_Y')
        benchmarks_weights = dataset.get('benchmarks_weights')

        benchmarks_Y = np.array(benchmarks_Y).reshape((789,-1))
        benchmarks_mean = np.mean(benchmarks_Y, axis=0)
        benchmarks_std = np.std(benchmarks_Y, axis=0)

        # x = [x for x in range(48)]
        # plt.plot(x,benchmarks_mean, 'k', color='#CC4F1B')
        # plt.fill_between(x,benchmarks_mean - benchmarks_std, benchmarks_mean + benchmarks_std,
        #                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        # plt.show()

        mean_val = 0
        std_val = 100

        mean_step = 10
        std_step = 5
        if n_iter is None:
            n_iter = 20
        mean_sample_size = int(mean_val/mean_step) +1
        std_sample_size = int(std_val/std_step) + 1
        noise_mean = np.linspace(0,mean_val, mean_sample_size)
        noise_std = np.linspace(0,std_val, std_sample_size)

        std_regrets = np.zeros(mean_sample_size)
        mean_regrets = np.zeros(std_sample_size)
        mypool = mp.Pool(processes=8)
        # capacities = [5,10,15,20,25,30,35,40]
        capacities = [12, 24, 48, 96, 120, 144, 196]
        map_func = partial(gen_worker_add,n_iter=n_iter,mean_sample_size=mean_sample_size,std_sample_size=std_sample_size,noise_mean=noise_mean,noise_std=noise_std,benchmarks_Y=benchmarks_Y,benchmarks_weights=benchmarks_weights,mean_val=mean_val,std_val=std_val)
        iter = [[c] for c in capacities]

        mypool.starmap(map_func, iter)

        # mean_val = 50
        # std_val = 50
        #
        # mean_step = 10
        # std_step = 2
        #
        # mean_sample_size = int(mean_val/mean_step) +1
        # std_sample_size = int(std_val/std_step) + 1
        # noise_mean = np.linspace(0,mean_val, mean_sample_size)
        # noise_std = np.linspace(0,std_val, std_sample_size)
        #
        #
        # map_func = partial(gen_worker_mult, n_iter=n_iter, mean_sample_size=mean_sample_size,
        #                    std_sample_size=std_sample_size, noise_mean=noise_mean, noise_std=noise_std,
        #                    benchmarks_Y=benchmarks_Y, benchmarks_weights=benchmarks_weights, mean_val=mean_val,
        #                    std_val=std_val)
        # iter = [[c] for c in capacities]
        #
        # mypool.starmap(map_func, iter)
        #
        # print("blalba")

def gen_worker_mult(c,n_iter,mean_sample_size,std_sample_size,noise_mean,noise_std,benchmarks_Y,benchmarks_weights,mean_val,std_val):
    opt_params = get_opt_params_knapsack(c)
    regrets = np.zeros((n_iter, mean_sample_size, std_sample_size))
    script_dir = os.path.dirname(__file__)
    for n in range(n_iter):
        for i, mean in enumerate(noise_mean):
            for j, std in enumerate(noise_std):
                for benchmark_Y, weights in zip(benchmarks_Y, benchmarks_weights):
                    print("c: {} n: {} mean: {} std: {}".format(c, n, mean, std))
                    np.random.seed(int(time.time()))
                    noisy_Y = benchmark_Y +  np.random.normal(mean, benchmark_Y* std/100)
                    # noisy_Y = benchmark_Y + np.random.normal(mean, std)
                    tov, predicted_sol, tov_opt, sol = get_regret_worker(benchmark_Y, noisy_Y, weights.astype(int),
                                                                         opt_params)
                    regrets[n, i, j] = np.median(tov_opt - tov)
    f_path = os.path.join(script_dir,'noise_test_var/multiplicative/w_mult_c{}m{}std{}n{}.npy'.format(c, mean_val, std_val, n_iter))
    np.save(f_path, regrets)

def gen_worker_add(c, n_iter, mean_sample_size, std_sample_size, noise_mean, noise_std, benchmarks_Y,
                    benchmarks_weights, mean_val, std_val):
    opt_params = get_opt_params_knapsack(c)
    regrets = np.zeros((n_iter, mean_sample_size, std_sample_size))
    script_dir = os.path.dirname(__file__)
    for n in range(n_iter):
        for i, mean in enumerate(noise_mean):
            for j, std in enumerate(noise_std):
                for benchmark_Y, weights in zip(benchmarks_Y, benchmarks_weights):
                    np.random.seed(int(time.time()))
                    print("c: {} n: {} mean: {} std: {}".format(c, n, mean, std))
                    noisy_Y = benchmark_Y + np.random.normal(mean, std)
                    tov, predicted_sol, tov_opt, sol = get_regret_worker(benchmark_Y, noisy_Y, weights.astype(int),
                                                                         opt_params)
                    regrets[n, i, j] = np.median(tov_opt - tov)
    f_path = os.path.join(script_dir,
                          'noise_test_var/additive/w_add_c{}m{}std{}n{}.npy'.format(c, mean_val, std_val,
                                                                                         n_iter))
    np.save(f_path, regrets)

def read_icon_regret(f_name = "noise_test_var/c12m10std10n50.npy", n_iter = 50):
    pepper_noise_var = 2
    scatter_spread_coef = 3
    std_start = 0
    std_end = 1000
    std_step = 5

    mean_start = 0
    mean_end = 0
    mean_step = 10
    means = [m for m in range(mean_start,mean_end,mean_step)]

    n_iter = 20
    f_mean = 0
    f_std = 50
    loads = [30]
    sns.set()
    # capacities = [5,10,15,20,25]
    capacities = [ 48, 96, 120, 144, 196]
    colour_palette = ["#7f0085", "#7f07c9", "#aeffde", "#00dd27", "#9b6415", "#fc6700", "#f9b09e"]
    # for c in capacities:
    #
    #     f_name = "noise_test_var/c{}m{}std{}n{}.npy".format(c,f_mean,f_std,n_iter)
    #     data = np.load(f_name)[:,0,std_start:std_end]
    #
    #     regret_mean = np.mean(data,axis=0).flatten()
    #     regret_std = np.std(data,axis=0).flatten()
    #     col = regret_mean.size
    #     x = [std_start + x*5 for x in range(col)]
    #
    #
    #
    #     fig, axarr = plt.subplots(2,1)
    #     fig.suptitle("Error std vs regret(N_iter={}) (C={})".format(n_iter, c))
    #     # axarr[0].plot(data[0])
    #     axarr[0].plot(x,regret_mean, 'k', color='#CC4F1B')
    #     axarr[0].fill_between(x,regret_mean - regret_std, regret_mean + regret_std,
    #                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    #     for n in range(n_iter):
    #         axarr[0].scatter(np.array(x) + np.random.rand(len(x)) * scatter_spread_coef,data[n].flatten() + np.random.rand(data[n].flatten().size) * scatter_spread_coef*3)
    #     axarr[0].set_xlabel("std")
    #     axarr[0].set_ylabel("regret")
    #
    #     # axarr[0].plot(data[0])
    #     axarr[1].plot(x, regret_mean, 'k', color='#CC4F1B')
    #     axarr[1].fill_between(x, regret_mean - regret_std, regret_mean + regret_std,
    #                      alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    #     for n in range(n_iter):
    #         axarr[1].scatter(np.array(x) ,
    #                     data[n].flatten() )
    #     axarr[1].set_xlabel("std")
    #     axarr[1].set_ylabel("regret")

    # Plotting all capacities at mean 0 error
    c_regrets = np.zeros(len(loads))
    fig,ax = plt.subplots(tight_layout=True)
    curr_mean = 0
    ax.set_title("Effect of Multiplicative Error on Regret(N_iter={})".format(n_iter, curr_mean*10), fontsize=15)
    patterns = ["-", "-", "-",'-', '--','--',':',':']
    markers = ['.',',','o','v','None','None','None','None']
    for i,l in enumerate(loads):
        # f_name = "noise_test_var/additive/icon_add_i{}m{}std{}n{}.npy".format(l,f_mean,f_std,n_iter)
        f_name = "noise_test_var/multiplicative/icon_mult_i{}m{}std{}n{}.npy".format(l, f_mean, f_std, n_iter)
        data = np.load(f_name)[:,curr_mean,int(std_start/std_step):int(std_end/std_step)]

        regret_mean = np.mean(data,axis=0).flatten()
        regret_max = np.max(data,axis=0).flatten()
        regret_min = np.min(data,axis=0).flatten()
        col = regret_mean.size
        x = [std_start + x*std_step for x in range(col)]
        ax.plot(x, regret_mean, 'k', color=colour_palette[i], label=l, linestyle=patterns[i], marker = markers[i], linewidth = 3)
        ax.fill_between(x,regret_min, regret_max,
                        alpha=0.4, facecolor=colour_palette[i])
        ax.set_xlabel("Error Deviation",fontsize= 20)
        ax.set_ylabel("Regret", fontsize=20)
    ax.legend()

    fig_name = 'icon_mult_loads_m{}std{}n{}fill.pdf'.format(curr_mean*10, f_std, n_iter)
    fig.savefig(fig_name)
    plt.show()
    # plt.show()
    #
    # # Plot different mean vs std for each capacities
    # # for i,c in enumerate(capacities):
    # #     fig,ax = plt.subplots()
    # #     fig.suptitle("Error std vs regret(N_iter={}) C={}".format(n_iter, c))
    # #     patterns = ["-", "-", "-",'-', '--','--',':',':']
    # #     markers = ['.',',','o','v','None','None','None','None']
    # #
    # #     # f_name = "noise_test_var/additive/w_add_c{}m{}std{}n{}.npy".format(c, f_mean, f_std, n_iter)
    # #     # f_name = "noise_test_var/c{}m{}std{}n{}.npy".format(c,f_mean,f_std,n_iter)
    # #     f_name = "noise_test_var/multiplicative/w_mult_c{}m{}std{}n{}.npy".format(c, f_mean, f_std, n_iter)
    # #     data = np.load(f_name)[:,int(mean_start/mean_step):int(mean_end/mean_step),int(std_start/std_step):int(std_end/std_step)]
    # #     for j,mean in enumerate(means):
    # #         regret_mean = np.mean(data[:,j,:],axis=0).flatten()
    # #         regret_max = np.max(data[:,j,:],axis=0).flatten()
    # #         regret_min = np.min(data[:,j,:],axis=0).flatten()
    # #         col = regret_mean.size
    # #         x = [std_start + x*5 for x in range(col)]
    # #         ax.plot(x, regret_mean, 'k', color=colour_palette[j], label=mean, linestyle=patterns[j], marker = markers[j])
    # #         ax.fill_between(x,regret_min , regret_max,
    # #                         alpha=0.1, facecolor=colour_palette[j])
    # #     ax.set_xlabel("std")
    # #     ax.set_ylabel("regret")
    # #     ax.legend()
    # #     fig_name = 'w_mult_means_c{}std{}n{}fill.png'.format(c, f_std, n_iter)
    # #     plt.savefig(fig_name)
    #

    print('hey')

def read_MSE_regret(f_name = "noise_test_var/c12m10std10n50.npy", n_iter = 50):
    pepper_noise_var = 2
    scatter_spread_coef = 1
    std_start = 0
    std_end = 140
    std_step = 5

    mean_start = 0
    mean_end = 0
    mean_step = 10
    means = [m for m in range(mean_start,mean_end,mean_step)]

    n_iter = 50
    f_mean = 50
    f_std = 50
    # capacities = [12]
    sns.set()
    capacities = [12, 24, 48, 96, 120, 144,196]
    # capacities = [5,10,15,20,25]
    # capacities = [ 48, 96, 120, 144, 196]
    colour_palette = ["#7f0085", "#7f07c9", "#aeffde", "#00dd27", "#9b6415", "#fc6700", "#f9b09e"]
    # for c in capacities:
    #
    #     f_name = "noise_test_var/c{}m{}std{}n{}.npy".format(c,f_mean,f_std,n_iter)
    #     data = np.load(f_name)[:,0,std_start:std_end]
    #
    #     regret_mean = np.mean(data,axis=0).flatten()
    #     regret_std = np.std(data,axis=0).flatten()
    #     col = regret_mean.size
    #     x = [std_start + x*std_step for x in range(col)]


        # fig, axarr = plt.subplots(2,1)
        # # axarr[0].plot(data[0])
        # axarr[1].set_title("(b) Perturbed Samples for Visualization")
        # axarr[1].plot(x,regret_mean, 'k', color='#CC4F1B')
        # axarr[1].fill_between(x,np.maximum(0,regret_mean - regret_std), regret_mean + regret_std,
        #                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        # for n in range(n_iter):
        #     axarr[1].scatter(np.array(x) + np.random.rand(len(x)) * scatter_spread_coef,data[n].flatten() + np.random.rand(data[n].flatten().size) * scatter_spread_coef*20)
        # axarr[1].set_xlabel("Accuracy Error Standard Deviation")
        # axarr[1].set_ylabel("Regret")
        #
        # # axarr[0].plot(data[0])
        # axarr[0].set_title("(a) Effect of Accuracy Error on Regret(N_iter={}) (C={})".format(n_iter, c))
        # axarr[0].plot(x, regret_mean, 'k', color='#CC4F1B')
        # axarr[0].fill_between(x, np.maximum(0,regret_mean - regret_std), regret_mean + regret_std,
        #                  alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        # for n in range(n_iter):
        #     axarr[0].scatter(np.array(x) ,
        #                 data[n].flatten() )
        # axarr[0].set_ylabel("Regret")
        # fig.savefig("MSE_regret_samples.pdf")
    #
    # Plotting all capacities at mean 0 error
    c_regrets = np.zeros(len(capacities))
    fig,ax = plt.subplots(tight_layout=True)
    curr_mean = 0
    fig.suptitle("Effect of Multiplicative Error on Regret(N_iter={})".format(n_iter, curr_mean*10), fontsize=15)
    patterns = ["-", "-", "-",'-', '--','--',':',':']
    markers = ['.',',','o','v','None','None','None','None']
    for i,c in enumerate(capacities):
        # f_name = "noise_test_var/additive/w_add_c{}m{}std{}n{}.npy".format(c,f_mean,f_std,n_iter)
        f_name = "noise_test_var/multiplicative/w_mult_c{}m{}std{}n{}.npy".format(c, f_mean, f_std, n_iter)
        data = np.load(f_name)[:,curr_mean,int(std_start/std_step):int(std_end/std_step)]

        regret_mean = np.mean(data,axis=0).flatten()
        regret_max = np.max(data,axis=0).flatten()
        regret_min = np.min(data,axis=0).flatten()
        col = regret_mean.size
        x = [std_start + x*2 for x in range(col)]
        ax.plot(x, regret_mean, 'k', color=colour_palette[i], label=c, linestyle=patterns[i], marker = markers[i])
        ax.fill_between(x,regret_min, regret_max,
                        alpha=0.6, facecolor=colour_palette[i])
        ax.set_xlabel("Error Deviation", fontsize=20)
        ax.set_ylabel("Regret", fontsize=20)
    ax.legend(fontsize=15, loc="upper left")

    fig_name = 'w_mult_capacities_m{}std{}n{}fill.pdf'.format(curr_mean*10, f_std, n_iter)
    plt.savefig(fig_name)

    #NO Fill

    fig, ax = plt.subplots(tight_layout=True)
    curr_mean = 0
    fig.suptitle("Effect of Multiplicative Error on Regret(N_iter={})".format(n_iter, curr_mean * 10), fontsize=15)
    patterns = ["-", "-", "-", '-', '--', '--', ':', ':']
    markers = ['.', ',', 'o', 'v', 'None', 'None', 'None', 'None']
    for i, c in enumerate(capacities):
        # f_name = "noise_test_var/additive/w_add_c{}m{}std{}n{}.npy".format(c, f_mean, f_std, n_iter)
        f_name = "noise_test_var/multiplicative/w_mult_c{}m{}std{}n{}.npy".format(c, f_mean, f_std, n_iter)
        data = np.load(f_name)[:, curr_mean, int(std_start / std_step):int(std_end / std_step)]

        regret_mean = np.mean(data, axis=0).flatten()
        regret_max = np.max(data, axis=0).flatten()
        regret_min = np.min(data, axis=0).flatten()
        col = regret_mean.size
        x = [std_start + x * 2 for x in range(col)]
        ax.plot(x, regret_mean, 'k', color=colour_palette[i], label=c, linestyle=patterns[i], marker=markers[i], linewidth = 3)
        ax.set_xlabel("Error Deviation", fontsize=20)
        ax.set_ylabel("Regret", fontsize=20)
    ax.legend(fontsize=15)

    fig_name = 'w_mult_capacities_m{}std{}n{}.pdf'.format(curr_mean * 10, f_std, n_iter)
    plt.savefig(fig_name)

    # Plot different mean vs std for each capacities
    # for i,c in enumerate(capacities):
    #     fig,ax = plt.subplots()
    #     fig.suptitle("Error std vs regret(N_iter={}) C={}".format(n_iter, c))
    #     patterns = ["-", "-", "-",'-', '--','--',':',':']
    #     markers = ['.',',','o','v','None','None','None','None']
    #
    #     # f_name = "noise_test_var/additive/w_add_c{}m{}std{}n{}.npy".format(c, f_mean, f_std, n_iter)
    #     # f_name = "noise_test_var/c{}m{}std{}n{}.npy".format(c,f_mean,f_std,n_iter)
    #     f_name = "noise_test_var/multiplicative/w_mult_c{}m{}std{}n{}.npy".format(c, f_mean, f_std, n_iter)
    #     data = np.load(f_name)[:,int(mean_start/mean_step):int(mean_end/mean_step),int(std_start/std_step):int(std_end/std_step)]
    #     for j,mean in enumerate(means):
    #         regret_mean = np.mean(data[:,j,:],axis=0).flatten()
    #         regret_max = np.max(data[:,j,:],axis=0).flatten()
    #         regret_min = np.min(data[:,j,:],axis=0).flatten()
    #         col = regret_mean.size
    #         x = [std_start + x*5 for x in range(col)]
    #         ax.plot(x, regret_mean, 'k', color=colour_palette[j], label=mean, linestyle=patterns[j], marker = markers[j])
    #         ax.fill_between(x,regret_min , regret_max,
    #                         alpha=0.1, facecolor=colour_palette[j])
    #     ax.set_xlabel("std")
    #     ax.set_ylabel("regret")
    #     ax.legend()
    #     fig_name = 'w_mult_means_c{}std{}n{}fill.png'.format(c, f_std, n_iter)
    #     plt.savefig(fig_name)

    plt.show()
    print('hey')

def mse_regression_tests(is_shuffle = True, kfold = 0,c=12):
    train_dict, val_dict, test_dict = prepare_icon_dataset(kfold= kfold, is_shuffle= is_shuffle)
    sns.set()
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
    sns.set()

    scikit_regression = linear_model.Ridge().fit(X_train, Y_train)
    opt_param = get_opt_params_knapsack(capacity=12)
    pred = scikit_regression.predict(X_test)
    error = np.mean(np.abs((Y_test - pred)))
    std = np.std((Y_test - pred))
    max_norm = sum(abs(pred))
    pred_norm = [float(i) / max_norm for i in pred]
    print("Regression error: {}, std: {}".format(error,std))
    x = [x for x in range(len(Y_test))]
    sorted_pred_regression = [x for _, x in sorted(zip(Y_test, pred))]
    sorted_pred_regression_norm = [x for _, x in sorted(zip(Y_test, pred_norm))]
    # plt.scatter([5,10],[20,10])

    relu_path = os.path.join(os.path.dirname(__file__),'models/dnl_knapsack_c{}f0l01.pth'.format(c))
    # relu_dnl = relu_ppo()
    relu_dnl = torch.load(relu_path)
    relu_dnl.eval()

    spo_path = os.path.join(os.path.dirname(__file__),'models/spo_knap_c{}f0l01.pth'.format(c))
    spo = torch.load(spo_path)

    # relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets,
    #                                                                                            weights_test_sets,
    #                                                                                            pool=mypool)

    # pred_Ys_relu = []
    sns.set()
    pred_Ys_dnl = relu_dnl.predict(X_test)
    max_norm = sum(abs(pred_Ys_dnl))
    pred_Ys_dnl_norm = [float(i) / max_norm for i in pred_Ys_dnl]

    sorted_pred_dnl = [x for _, x in sorted(zip(Y_test, pred_Ys_dnl))]
    sorted_pred_dnl_norm = [x for _, x in sorted(zip(Y_test, pred_Ys_dnl_norm))]

    pred_Ys_spo= spo.predict(X_test, qid=False)
    max_norm = sum(abs(pred_Ys_spo))
    pred_Ys_spo_norm = [float(i) / max_norm for i in pred_Ys_spo]


    sorted_pred_spo = [x for _, x in sorted(zip(Y_test, pred_Ys_spo))]
    sorted_pred_spo_norm = [x for _, x in sorted(zip(Y_test, pred_Ys_spo_norm))]


    max_norm = sum(abs(Y_test.flatten()))
    norm_Y = [float(i) / max_norm for i in Y_test.flatten()]



    fig_icon, ax_icon = plt.subplots(tight_layout=True)
    ax_icon.scatter(x, sorted(norm_Y), c='r', label='Samples')

    ax_icon.set_title('Icon Test Samples (Sorted)')
    ax_icon.set_xlabel("Sample No")
    ax_icon.set_ylabel("Energy Price")
    ax_icon.legend()
    fig_icon.savefig("Icon_Weighted.pdf")
    fig_icon.savefig("Icon_Weighted.png")

    fig_icon_density, ax_icon_density = plt.subplots(tight_layout=True)

    sns.kdeplot(np.array(norm_Y), ax=ax_icon_density)
    norm_Y_mean = np.mean(np.array(norm_Y))
    norm_Y_median = np.median(np.array(norm_Y))
    ax_icon_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    ax_icon_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')
    ax_icon_density.legend()
    fig_icon_density.savefig("Icon_Weighted_density.pdf")
    fig_icon_density.savefig("Icon_Weighted_density.png")



    fig_reg, ax_reg = plt.subplots(tight_layout=True)
    ax_reg.scatter(x,sorted_pred_regression_norm,c='b', label='regression')
    ax_reg.scatter(x, sorted(norm_Y), c = 'r', label='real')
    ax_reg.set_title('Regression_and_Real_Data (Sorted)')
    ax_reg.legend()
    ax_reg.set_ylabel('Value')
    ax_reg.set_xlabel("Samples")
    fig_reg.savefig("Regression_Icon_Weighted.pdf")
    fig_reg.savefig("Regression_Icon_Weighted.png")

    fig_reg_density, ax_reg_density = plt.subplots(tight_layout=True)

    sns.kdeplot(np.array(sorted_pred_regression_norm), ax=ax_reg_density)
    norm_Y_mean = np.mean(np.array(sorted_pred_regression_norm))
    norm_Y_median = np.median(np.array(sorted_pred_regression_norm))

    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')
    ax_reg_density.legend()

    fig_reg_density.savefig("Regression_Icon_Weighted_density.pdf")
    fig_reg_density.savefig("Regression_Icon_Weighted_density.png")

    fig1, axs1 = plt.subplots(4,1,tight_layout=True)
    axs1[0].scatter(x,sorted_pred_regression_norm,c='b', label='regression')
    axs1[0].scatter(x, sorted(norm_Y), c = 'r', label='real')
    axs1[0].set_title('Regression and Real Data (Sorted)')
    axs1[0].legend()


    axs1[1].scatter(x,sorted_pred_regression_norm,c='b', label = 'Regression')
    n=100
    moving_avg = np.cumsum(sorted_pred_regression_norm, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    axs1[1].scatter(x[n-1:],moving_avg,c='r', label='Moving Avg')

    axs1[1].set_title('Regression (Sorted, Moving Average)')
    axs1[1].legend()


    axs1[2].scatter(x,sorted_pred_dnl_norm,c='b', label='dnl')
    moving_avg = np.cumsum(sorted_pred_dnl_norm, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    axs1[2].scatter(x[n-1:],moving_avg,c='r', label= 'Moving Avg')
    axs1[2].set_title('DNL (Sorted, Moving Average)')
    axs1[2].legend()

    axs1[3].scatter(x,sorted_pred_spo_norm,c='b', label='Spo')
    moving_avg = np.cumsum(sorted_pred_spo_norm, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    axs1[3].scatter(x[n-1:],moving_avg,c='r', label= 'Moving Avg')
    axs1[3].set_xlabel('Sample No')
    axs1[3].set_title('SPO (Sorted, Moving Average)')
    axs1[3].legend()

    fig1.savefig('model_forecasts_wknap_c{}.pdf'.format(c))
    fig1.savefig('model_forecasts_wknap_c{}.png'.format(c))
    # plt.scatter(x, sorted(Y_test.flatten()), c = 'r')

    fig2, axs2 = plt.subplots(tight_layout=True)

    axs2.scatter(x, sorted(norm_Y), c=colour_palette[0], label='real')

    moving_avg = np.cumsum(sorted_pred_regression_norm, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    axs2.scatter(x[n-1:],moving_avg,c=colour_palette[4], label='Regression')

    moving_avg = np.cumsum(sorted_pred_dnl_norm, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    axs2.scatter(x[n-1:],moving_avg,c=colour_palette[2], label= 'DNL')

    moving_avg = np.cumsum(sorted_pred_spo_norm, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    axs2.scatter(x[n-1:],moving_avg,c=colour_palette[3], label= 'SPO')
    axs2.legend()
    axs2.set_xlabel("Samples", fontsize = 20)
    axs2.set_ylabel("Value", fontsize = 20)
    axs2.legend(fontsize = 20)


    fig2.savefig('model_forecasts_merged_wknap_c{}.pdf'.format(c))
    fig2.savefig('model_forecasts_merged_wknap_c{}.png'.format(c))


    fig_merged_density, ax_merged_density = plt.subplots(tight_layout=True)

    sns.kdeplot(np.array(sorted(norm_Y)).flatten(), ax=ax_merged_density, label='Real', c=colour_palette[0])
    norm_Y_mean = np.mean(np.array(sorted(norm_Y)))
    norm_Y_median = np.median(np.array(sorted(norm_Y)))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')


    sns.kdeplot(np.array(sorted_pred_regression_norm).flatten(), ax=ax_merged_density,label='Regression',c=colour_palette[1])
    norm_Y_mean = np.mean(np.array(sorted_pred_regression_norm))
    norm_Y_median = np.median(np.array(sorted_pred_regression_norm))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')

    sns.kdeplot(np.array(sorted_pred_dnl_norm).flatten(), ax=ax_merged_density,label='DNL',c=colour_palette[2])
    norm_Y_mean = np.mean(np.array(sorted_pred_dnl_norm))
    norm_Y_median = np.median(np.array(sorted_pred_dnl_norm))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')


    sns.kdeplot(np.array(sorted_pred_spo_norm).flatten(), ax=ax_merged_density,label='SPO',c=colour_palette[3])
    norm_Y_mean = np.mean(np.array(sorted_pred_spo_norm))
    norm_Y_median = np.median(np.array(sorted_pred_spo_norm))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')
    ax_merged_density.set_xlabel("Value", fontsize = 20)
    ax_merged_density.legend(fontsize = 20)
    fig_merged_density.savefig("Merged_Weighted_density_c{}.pdf".format(c))
    fig_merged_density.savefig("Merged_Weighted_density_c{}.png".format(c))


    fig3, axs3 = plt.subplots(4,1,tight_layout=True)
    sorted_Y = sorted(Y_test)
    sorted_pred_regression_error = np.abs(np.array(sorted_pred_regression) - np.array(sorted_Y))
    max_norm = sum(abs(sorted_pred_regression_error))
    sorted_pred_regression_error_norm = [float(i) / max_norm for i in sorted_pred_regression_error]


    sorted_pred_dnl_error = np.abs(np.array(sorted_pred_dnl) - np.array(sorted_Y))
    max_norm = sum(abs(sorted_pred_dnl_error))
    sorted_pred_dnl_error_norm = [float(i) / max_norm for i in sorted_pred_dnl_error]


    sorted_pred_spo_error = np.abs(np.array(sorted_pred_spo) - np.array(sorted_Y).flatten())
    max_norm = sum(abs(sorted_pred_spo_error))
    sorted_pred_spo_error_norm = [float(i) / max_norm for i in sorted_pred_spo_error]

    a = 0.1

    axs3[0].scatter(x, sorted_pred_regression_error_norm, c='b', label='regression')
    axs3[0].scatter(x, sorted(norm_Y), c='r', label='real',alpha = a)
    axs3[0].set_title('Regression and Real Data (Error,Sorted)')
    axs3[0].legend()

    axs3[1].scatter(x, sorted_pred_regression_error_norm, c='b', label='Regression')
    n = 100
    moving_avg = np.cumsum(sorted_pred_regression_error_norm, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    # axs3[1].scatter(x[n - 1:], moving_avg, c='r', label='Moving Avg',alpha = a)

    axs3[1].set_title('Regression (Error, Sorted)')
    axs3[1].legend()

    axs3[2].scatter(x, sorted_pred_dnl_error_norm, c='b', label='dnl')
    moving_avg = np.cumsum(sorted_pred_dnl_error_norm, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    # axs3[2].scatter(x[n - 1:], moving_avg, c='r', label='Moving Avg',alpha = a)
    axs3[2].set_title('DNL (Error, Sorted)')
    axs3[2].legend()

    axs3[3].scatter(x, sorted_pred_spo_error_norm, c='b', label='Spo')
    moving_avg = np.cumsum(sorted_pred_spo_error_norm, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    # axs3[3].scatter(x[n - 1:], moving_avg, c='r', label='Moving Avg', alpha = a)
    axs3[3].set_xlabel('Sample No')
    axs3[3].set_title('SPO (Error, Sorted   )')
    axs3[3].legend()

    fig3.savefig('model_forecasts_abserrors_wknap_c12.pdf')
    fig3.savefig('model_forecasts_abserrors_wknap_c12.png')

    # plt.show()
def ppo_different_problems(is_shuffle = True, kfold = 0):
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


    scikit_regression = linear_model.Ridge().fit(X_train, Y_train)
    opt_param = get_opt_params_knapsack(capacity=12)
    pred = scikit_regression.predict(X_test)
    error = np.mean((Y_test - pred)**2)
    std = np.std((Y_test - pred)**2)
    max_norm = sum(pred)
    pred = [float(i) / max_norm for i in pred]
    print("Regression error: {}, std: {}".format(error,std))
    plt.figure()
    x = [x for x in range(len(Y_test))]
    sorted_pred_regression = [x for _, x in sorted(zip(Y_test, pred))]
    # plt.scatter([5,10],[20,10])

    "C 12"
    relu_path = os.path.join(os.path.dirname(__file__),'models/dnl_knapsack_c12f0l01.pth')
    # relu_dnl = relu_ppo()
    relu_dnl = torch.load(relu_path)
    relu_dnl.eval()

    # relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets,
    #                                                                                            weights_test_sets,
    #                                                                                            pool=mypool)

    # pred_Ys_relu = []

    pred_Ys_relu_c12= relu_dnl.predict(X_test)
    max_norm = sum(pred_Ys_relu_c12)
    pred_Ys_relu_c12 = [float(i) / max_norm for i in pred_Ys_relu_c12]
    sorted_pred_dnl_c12 = [x for _, x in sorted(zip(Y_test, pred_Ys_relu_c12))]

    "C 48"

    relu_path = os.path.join(os.path.dirname(__file__),'models/dnl_knapsack_c48f0l01.pth')
    # relu_dnl = relu_ppo()
    relu_dnl = torch.load(relu_path)
    relu_dnl.eval()

    # relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets,
    #                                                                                            weights_test_sets,
    #                                                                                            pool=mypool)

    # pred_Ys_relu = []

    pred_Ys_relu_c48= relu_dnl.predict(X_test)
    max_norm = sum(pred_Ys_relu_c48)
    pred_Ys_relu_c48 = [float(i) / max_norm for i in pred_Ys_relu_c48]
    sorted_pred_dnl_c48 = [x for _, x in sorted(zip(Y_test, pred_Ys_relu_c48))]

    "C 196"

    relu_path = os.path.join(os.path.dirname(__file__),'models/dnl_knapsack_c196f0l01.pth')
    # relu_dnl = relu_ppo()
    relu_dnl = torch.load(relu_path)
    relu_dnl.eval()

    # relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets,
    #                                                                                            weights_test_sets,
    #                                                                                            pool=mypool)

    # pred_Ys_relu = []

    pred_Ys_relu_c196= relu_dnl.predict(X_test)
    max_norm = sum(pred_Ys_relu_c196)
    pred_Ys_relu_c196 = [float(i) / max_norm for i in pred_Ys_relu_c196]
    sorted_pred_dnl_c196 = [x for _, x in sorted(zip(Y_test, pred_Ys_relu_c196))]

    "C 220"

    relu_path = os.path.join(os.path.dirname(__file__),'models/dnl_knapsack_c220f0l01.pth')
    # relu_dnl = relu_ppo()
    relu_dnl = torch.load(relu_path)
    relu_dnl.eval()

    # relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets,
    #                                                                                            weights_test_sets,
    #                                                                                            pool=mypool)

    # pred_Ys_relu = []
    sns.set()
    pred_Ys_relu_c220= relu_dnl.predict(X_test)
    max_norm = sum(pred_Ys_relu_c220)
    pred_Ys_relu_c220 = [float(i) / max_norm for i in pred_Ys_relu_c220]
    sorted_pred_dnl_c220 = [x for _, x in sorted(zip(Y_test, pred_Ys_relu_c220))]

    max_norm = sum(Y_test.flatten())
    norm_Y = [float(i) / max_norm for i in Y_test.flatten()]

    n=100

    fig, ax = plt.subplots(4,1, tight_layout=True)
    ax[0].scatter(x,sorted_pred_dnl_c12,c='b')
    # plt.scatter(x, sorted(norm_Y), c = 'r')

    moving_avg = np.cumsum(sorted_pred_dnl_c12, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    ax[0].scatter(x[n-1:],moving_avg,c='r')
    ax[0].set_title('C 12')
    ax[0].set_xticklabels([])


    ax[1].scatter(x, sorted_pred_dnl_c48, c='b')
    # plt.scatter(x, sorted(norm_Y), c = 'r')

    moving_avg = np.cumsum(sorted_pred_dnl_c48, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    ax[1].scatter(x[n - 1:], moving_avg, c='r')
    ax[1].set_title('C 48')
    ax[1].set_xticklabels([])




    ax[2].scatter(x, sorted_pred_dnl_c196, c='b')
    # plt.scatter(x, sorted(norm_Y), c = 'r')

    moving_avg = np.cumsum(sorted_pred_dnl_c196, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    ax[2].scatter(x[n - 1:], moving_avg, c='r')
    ax[2].set_title('C 196')
    ax[2].set_xticklabels([])


    ax[3].scatter(x, sorted_pred_dnl_c220, c='b')
    # plt.scatter(x, sorted(norm_Y), c = 'r')

    moving_avg = np.cumsum(sorted_pred_dnl_c220, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    ax[3].scatter(x[n - 1:], moving_avg, c='r')
    ax[3].set_title('C 220')
    ax[3].set_xlabel('Samples')
    fig.supylabel('Predicted Value')

    fig.savefig('w_dnl_knap.pdf')
    fig.savefig('w_dnl_knap.png')
    # plt.show()

    fig_merged_density, ax_merged_density = plt.subplots()


    sns.kdeplot(np.array((sorted_pred_dnl_c12)).flatten(), ax=ax_merged_density, label='c12')
    norm_Y_mean = np.mean(np.array((sorted_pred_dnl_c12)))
    norm_Y_median = np.median(np.array((sorted_pred_dnl_c12)))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')

    sns.kdeplot(np.array(sorted_pred_dnl_c48).flatten(), ax=ax_merged_density, label='c48')
    norm_Y_mean = np.mean(np.array(sorted_pred_dnl_c48))
    norm_Y_median = np.median(np.array(sorted_pred_dnl_c48))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')

    sns.kdeplot(np.array(sorted_pred_dnl_c196).flatten(), ax=ax_merged_density, label='c196')
    norm_Y_mean = np.mean(np.array(sorted_pred_dnl_c196))
    norm_Y_median = np.median(np.array(sorted_pred_dnl_c196))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')

    sns.kdeplot(np.array(sorted_pred_dnl_c220).flatten(), ax=ax_merged_density, label='c220')
    norm_Y_mean = np.mean(np.array(sorted_pred_dnl_c220))
    norm_Y_median = np.median(np.array(sorted_pred_dnl_c220))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')

    ax_merged_density.legend()
    fig_merged_density.savefig("Merged_Weighted_density_dnl_different_c.pdf")
    fig_merged_density.savefig("Merged_Weighted_density_dnl_different_c.png")


def ppo_different_problems_spo(is_shuffle = True, kfold = 0):
    train_dict, val_dict, test_dict = prepare_icon_dataset(kfold= kfold, is_shuffle= is_shuffle)

    X_train_sets = train_dict.get("X_sets")
    Y_train_sets = train_dict.get("Y_sets")

    X_train = train_dict.get("X")
    Y_train = train_dict.get("Y")

    sns.set()
    X_test_sets = test_dict.get("X_sets")
    Y_test_sets = test_dict.get("Y_sets")
    weights_test_sets = test_dict.get("Weights_sets")
    X_test = test_dict.get("X")
    Y_test = test_dict.get("Y")

    mypool = multiprocessing.Pool(processes=4)


    scikit_regression = linear_model.Ridge().fit(X_train, Y_train)
    opt_param = get_opt_params_knapsack(capacity=12)
    pred = scikit_regression.predict(X_test)
    error = np.mean((Y_test - pred)**2)
    std = np.std((Y_test - pred)**2)
    max_norm = sum(pred)
    pred = [float(i) / max_norm for i in pred]
    print("Regression error: {}, std: {}".format(error,std))
    plt.figure()
    x = [x for x in range(len(Y_test))]
    sorted_pred_regression = [x for _, x in sorted(zip(Y_test, pred))]
    # plt.scatter([5,10],[20,10])

    "C 12"
    spo_path = os.path.join(os.path.dirname(__file__),'models/spo_knap_c12f0l01.pth')
    spo = torch.load(spo_path)
    # spo.eval()

    # relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets,
    #                                                                                            weights_test_sets,
    #                                                                                            pool=mypool)

    # pred_Ys_relu = []

    pred_Ys_spo_c12= spo.predict(X_test,qid = False)
    max_norm = sum(pred_Ys_spo_c12)
    pred_Ys_spo_c12 = [float(i) / max_norm for i in pred_Ys_spo_c12]
    sorted_pred_spo_c12 = [x for _, x in sorted(zip(Y_test, pred_Ys_spo_c12))]

    "C 48"

    spo_path = os.path.join(os.path.dirname(__file__),'models/spo_knap_c48f0l01.pth')
    spo = torch.load(spo_path)


    # relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets,
    #                                                                                            weights_test_sets,
    #                                                                                            pool=mypool)

    # pred_Ys_relu = []

    pred_Ys_spo_c48= spo.predict(X_test,qid = False)
    max_norm = sum(pred_Ys_spo_c48)
    pred_Ys_spo_c48 = [float(i) / max_norm for i in pred_Ys_spo_c48]
    sorted_pred_spo_c48 = [x for _, x in sorted(zip(Y_test, pred_Ys_spo_c48))]

    "C 196"

    spo_path = os.path.join(os.path.dirname(__file__),'models/spo_knap_c196f0l01.pth')
    spo = torch.load(spo_path)

    # relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets,
    #                                                                                            weights_test_sets,
    #                                                                                            pool=mypool)

    # pred_Ys_relu = []

    pred_Ys_spo_c196= spo.predict(X_test,qid = False)
    max_norm = sum(pred_Ys_spo_c196)
    pred_Ys_spo_c196 = [float(i) / max_norm for i in pred_Ys_spo_c196]
    sorted_pred_spo_c196 = [x for _, x in sorted(zip(Y_test, pred_Ys_spo_c196))]

    "C 220"

    spo_path = os.path.join(os.path.dirname(__file__),'models/spo_knap_c220f0l01.pth')
    spo = torch.load(spo_path)

    # relu_regret, relu_test_obj, relu_predicted_solutions, relu_solutions = relu_dnl.get_regret(X_test_sets, Y_test_sets,
    #                                                                                            weights_test_sets,
    #                                                                                            pool=mypool)

    # pred_Ys_relu = []

    pred_Ys_spo_c220= spo.predict(X_test,qid = False)
    max_norm = sum(pred_Ys_spo_c220)
    pred_Ys_spo_c220 = [float(i) / max_norm for i in pred_Ys_spo_c220]
    sorted_pred_spo_c220 = [x for _, x in sorted(zip(Y_test, pred_Ys_spo_c220))]

    max_norm = sum(Y_test.flatten())
    norm_Y = [float(i) / max_norm for i in Y_test.flatten()]

    n=100

    fig, ax = plt.subplots(4,1, tight_layout=True)
    ax[0].scatter(x,sorted_pred_spo_c12,c='b')
    # plt.scatter(x, sorted(norm_Y), c = 'r')

    moving_avg = np.cumsum(sorted_pred_spo_c12, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    ax[0].scatter(x[n-1:],moving_avg,c='r')
    ax[0].set_title('C 12')
    ax[0].set_xticklabels([])

    ax[1].scatter(x, sorted_pred_spo_c48, c='b')
    # plt.scatter(x, sorted(norm_Y), c = 'r')

    moving_avg = np.cumsum(sorted_pred_spo_c48, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    ax[1].scatter(x[n - 1:], moving_avg, c='r')
    ax[1].set_title('C 48')
    ax[1].set_xticklabels([])


    ax[2].scatter(x, sorted_pred_spo_c196, c='b')
    # plt.scatter(x, sorted(norm_Y), c = 'r')

    moving_avg = np.cumsum(sorted_pred_spo_c196, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    ax[2].scatter(x[n - 1:], moving_avg, c='r')
    ax[2].set_title('C 196')
    ax[2].set_xticklabels([])

    ax[3].scatter(x, sorted_pred_spo_c220, c='b')
    # plt.scatter(x, sorted(norm_Y), c = 'r')

    moving_avg = np.cumsum(sorted_pred_spo_c220, dtype=float)
    moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    moving_avg = moving_avg[n - 1:] / n
    ax[3].scatter(x[n - 1:], moving_avg, c='r')
    ax[3].set_title('C 220')

    ax[3].set_xlabel('Samples')
    fig.supylabel('Predicted Value')

    fig.savefig('w_knap_spo.pdf')
    fig.savefig('w_knap_spo.png')
    # plt.show()

    fig_merged_density, ax_merged_density = plt.subplots()

    sns.set()
    sns.kdeplot(np.array((sorted_pred_spo_c12)).flatten(), ax=ax_merged_density, label='c12')
    norm_Y_mean = np.mean(np.array((sorted_pred_spo_c12)))
    norm_Y_median = np.median(np.array((sorted_pred_spo_c12)))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')

    sns.kdeplot(np.array(sorted_pred_spo_c48).flatten(), ax=ax_merged_density, label='c48')
    norm_Y_mean = np.mean(np.array(sorted_pred_spo_c48))
    norm_Y_median = np.median(np.array(sorted_pred_spo_c48))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')

    sns.kdeplot(np.array(sorted_pred_spo_c196).flatten(), ax=ax_merged_density, label='c196')
    norm_Y_mean = np.mean(np.array(sorted_pred_spo_c196))
    norm_Y_median = np.median(np.array(sorted_pred_spo_c196))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')

    sns.kdeplot(np.array(sorted_pred_spo_c220).flatten(), ax=ax_merged_density, label='c220')
    norm_Y_mean = np.mean(np.array(sorted_pred_spo_c220))
    norm_Y_median = np.median(np.array(sorted_pred_spo_c220))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5,label = 'mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label = 'median')
    ax_merged_density.xlabel("Value", fontsize = 20)
    ax_merged_density.legend(fontsize = 20)
    fig_merged_density.savefig("Merged_Weighted_density_spo_different_c.pdf")
    fig_merged_density.savefig("Merged_Weighted_density_spo_different_c.png")

def average_problem_set(kfold, is_shuffle):
    sns.set()

    train_dict, val_dict, test_dict = prepare_icon_dataset(kfold=kfold, is_shuffle=is_shuffle)

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
    relu_path = os.path.join(os.path.dirname(__file__), 'models/dnl_knapsack_c12f0l01.pth')
    spo_path = os.path.join(os.path.dirname(__file__), 'models/spo_knap_c12f0l01.pth')
    # relu_dnl = relu_ppo()
    relu_dnl_c12 = torch.load(relu_path)
    relu_dnl_c12.eval()

    spo_c12 = torch.load(spo_path)



    relu_path = os.path.join(os.path.dirname(__file__), 'models/dnl_knapsack_c48f0l01.pth')
    spo_path = os.path.join(os.path.dirname(__file__), 'models/spo_knap_c48f0l01.pth')
    # relu_dnl = relu_ppo()
    relu_dnl_c48 = torch.load(relu_path)
    relu_dnl_c48.eval()

    spo_c48 = torch.load(spo_path)

    relu_path = os.path.join(os.path.dirname(__file__), 'models/dnl_knapsack_c196f0l01.pth')
    spo_path = os.path.join(os.path.dirname(__file__), 'models/spo_knap_c196f0l01.pth')
    # relu_dnl = relu_ppo()
    relu_dnl_c196 = torch.load(relu_path)
    relu_dnl_c196.eval()

    spo_c196 = torch.load(spo_path)

    relu_path = os.path.join(os.path.dirname(__file__), 'models/dnl_knapsack_c220f0l01.pth')
    spo_path = os.path.join(os.path.dirname(__file__), 'models/spo_knap_c220f0l01.pth')
    # relu_dnl = relu_ppo()
    relu_dnl_c220 = torch.load(relu_path)
    relu_dnl_c220.eval()

    spo_c220 = torch.load(spo_path)



    regression_pred = []
    dnl_pred_c12 = []
    dnl_pred_c48 = []
    dnl_pred_c196 = []
    dnl_pred_c220 = []

    spo_pred_c12 = []
    spo_pred_c48 = []
    spo_pred_c196 = []
    spo_pred_c220 = []


    real_v = []
    for x,y in zip(X_test_sets,Y_test_sets):
        regression_pred_this = scikit_regression.predict(x)
        dnl_pred_this_c12 = relu_dnl_c12.predict(x)
        dnl_pred_this_c48 = relu_dnl_c48.predict(x)
        dnl_pred_this_c196 = relu_dnl_c196.predict(x)
        dnl_pred_this_c220 = relu_dnl_c220.predict(x)

        spo_pred_this_c12 = spo_c12.predict(x, qid = False)
        spo_pred_this_c48 = spo_c48.predict(x, qid = False)
        spo_pred_this_c196 = spo_c196.predict(x, qid = False)
        spo_pred_this_c220 = spo_c220.predict(x, qid = False)

        real_v_this = y.flatten()

        dnl_pred_this_c12 = [x for _, x in sorted(zip(y.flatten(), dnl_pred_this_c12), key=lambda pair: pair[0])]
        dnl_pred_this_c48 = [x for _, x in sorted(zip(y.flatten(), dnl_pred_this_c48), key=lambda pair: pair[0])]
        dnl_pred_this_c196 = [x for _, x in sorted(zip(y.flatten(), dnl_pred_this_c196), key=lambda pair: pair[0])]
        dnl_pred_this_c220 = [x for _, x in sorted(zip(y.flatten(), dnl_pred_this_c220), key=lambda pair: pair[0])]

        spo_pred_this_c12 = [x for _, x in sorted(zip(y.flatten(), spo_pred_this_c12), key=lambda pair: pair[0])]
        spo_pred_this_c48 = [x for _, x in sorted(zip(y.flatten(), spo_pred_this_c48), key=lambda pair: pair[0])]
        spo_pred_this_c196 = [x for _, x in sorted(zip(y.flatten(), spo_pred_this_c196), key=lambda pair: pair[0])]
        spo_pred_this_c220 = [x for _, x in sorted(zip(y.flatten(), spo_pred_this_c220), key=lambda pair: pair[0])]

        regression_pred_this = [x for _, x in
                            sorted(zip(y.flatten(), regression_pred_this), key=lambda pair: pair[0])]
        real_v_this = sorted(real_v_this)



        max_norm = sum(real_v_this)
        real_v_this = [[float(i) / max_norm] for i in real_v_this]
        real_v.append(real_v_this)


        max_norm = sum([abs(x) for x in regression_pred_this])
        regression_pred_this = [float(i) / max_norm for i in regression_pred_this]
        regression_pred.append(regression_pred_this)


        max_norm = sum([abs(x) for x in dnl_pred_this_c12])
        regression_pred_this = [float(i) / max_norm for i in dnl_pred_this_c12]
        dnl_pred_c12.append(regression_pred_this)

        max_norm = sum([abs(x) for x in dnl_pred_this_c48])
        regression_pred_this = [float(i) / max_norm for i in dnl_pred_this_c48]
        dnl_pred_c48.append(regression_pred_this)

        max_norm = sum([abs(x) for x in dnl_pred_this_c196])
        regression_pred_this = [float(i) / max_norm for i in dnl_pred_this_c196]
        dnl_pred_c196.append(regression_pred_this)

        max_norm = sum([abs(x) for x in dnl_pred_this_c220])
        regression_pred_this = [float(i) / max_norm for i in dnl_pred_this_c220]
        dnl_pred_c220.append(regression_pred_this)

        max_norm = sum([abs(x) for x in spo_pred_this_c12])
        regression_pred_this = [float(i) / max_norm for i in spo_pred_this_c12]
        spo_pred_c12.append(regression_pred_this)

        max_norm = sum([abs(x) for x in spo_pred_this_c48])
        regression_pred_this = [float(i) / max_norm for i in spo_pred_this_c48]
        spo_pred_c48.append(regression_pred_this)

        max_norm = sum([abs(x) for x in spo_pred_this_c196])
        regression_pred_this = [float(i) / max_norm for i in spo_pred_this_c196]
        spo_pred_c196.append(regression_pred_this)

        max_norm = sum([abs(x) for x in spo_pred_this_c220])
        regression_pred_this = [float(i) / max_norm for i in spo_pred_this_c220]
        spo_pred_c220.append(regression_pred_this)


    dnl_pred_c12 = np.array(dnl_pred_c12)
    dnl_pred_c48 = np.array(dnl_pred_c48)
    dnl_pred_c196 = np.array(dnl_pred_c196)
    dnl_pred_c220 = np.array(dnl_pred_c220)

    spo_pred_c12 = np.array(spo_pred_c12)
    spo_pred_c48 = np.array(spo_pred_c48)
    spo_pred_c196 = np.array(spo_pred_c196)
    spo_pred_c220 = np.array(spo_pred_c220)

    regression_pred = np.array(regression_pred)

    dnl_pred_mean_c12 = np.mean(np.array(dnl_pred_c12), axis=0).flatten()
    dnl_pred_mean_c48 = np.mean(np.array(dnl_pred_c48), axis=0).flatten()
    dnl_pred_mean_c196 = np.mean(np.array(dnl_pred_c196), axis=0).flatten()
    dnl_pred_mean_c220 = np.mean(np.array(dnl_pred_c220), axis=0).flatten()

    spo_pred_mean_c12 = np.mean(np.array(spo_pred_c12), axis=0).flatten()
    spo_pred_mean_c48 = np.mean(np.array(spo_pred_c48), axis=0).flatten()
    spo_pred_mean_c196 = np.mean(np.array(spo_pred_c196), axis=0).flatten()
    spo_pred_mean_c220 = np.mean(np.array(spo_pred_c220), axis=0).flatten()


    regression_pred_mean = np.mean(np.array(regression_pred),axis=0).flatten()
    real_v_mean = np.mean(np.array(real_v), axis=0).flatten()

    real_v_std = np.std(np.array(real_v), axis=0).flatten()

    dnl_pred_std_c12 = np.std(np.array(dnl_pred_c12), axis=0).flatten()
    dnl_pred_std_c48 = np.std(np.array(dnl_pred_c48), axis=0).flatten()
    dnl_pred_std_c196 = np.std(np.array(dnl_pred_c196), axis=0).flatten()
    dnl_pred_std_c220 = np.std(np.array(dnl_pred_c220), axis=0).flatten()

    spo_pred_std_c12 = np.std(np.array(spo_pred_c12), axis=0).flatten()
    spo_pred_std_c48 = np.std(np.array(spo_pred_c48), axis=0).flatten()
    spo_pred_std_c196 = np.std(np.array(spo_pred_c196), axis=0).flatten()
    spo_pred_std_c220 = np.std(np.array(spo_pred_c220), axis=0).flatten()

    regression_pred_std = np.std(np.array(regression_pred),axis=0).flatten()

    x = [x for x in range(len(real_v_mean))]

    fig_real, ax_real = plt.subplots()

    ax_real.plot(real_v_mean, c=colour_palette[0], label='Real', linewidth = 3)
    ax_real.fill_between(x, real_v_mean - real_v_std / 2, real_v_mean + real_v_std / 2,
                    alpha=0.1, facecolor=colour_palette[0])

    ax_real.set_title('Icon Average Problem Set', fontsize= 20)
    ax_real.set_xlabel('Samples', fontsize=20)
    ax_real.set_ylabel('Value', fontsize=20)
    ax_real.legend(fontsize= 15)
    fig_real.savefig("icon_problem_set.pdf")
    fig_real.savefig("icon_problem_set.png")


    fig,ax = plt.subplots(tight_layout=True)



    ax.plot(dnl_pred_mean_c12, c = colour_palette[0], label='c12', linewidth = 3)
    ax.fill_between(x, dnl_pred_mean_c12 - dnl_pred_std_c12/2, dnl_pred_mean_c12 + dnl_pred_std_c12/2,
                    alpha=0.1, facecolor=colour_palette[0])

    ax.plot(dnl_pred_mean_c48, c = colour_palette[1], label='c48', linewidth = 3)
    ax.fill_between(x, dnl_pred_mean_c48 - dnl_pred_std_c48/2, dnl_pred_mean_c48 + dnl_pred_std_c48/2,
                    alpha=0.1, facecolor=colour_palette[1])


    ax.plot(dnl_pred_mean_c196, c = colour_palette[2], label='c196', linewidth = 3)
    ax.fill_between(x, dnl_pred_mean_c196 - dnl_pred_std_c196/2, dnl_pred_mean_c196 + dnl_pred_std_c196/2,
                    alpha=0.1, facecolor=colour_palette[2])

    ax.plot(dnl_pred_mean_c220, c = colour_palette[3], label='c220', linewidth = 3)
    ax.fill_between(x, dnl_pred_mean_c220 - dnl_pred_std_c220/2, dnl_pred_mean_c220 + dnl_pred_std_c220/2,
                    alpha=0.1, facecolor=colour_palette[3])



    ax.plot(regression_pred_mean, c=colour_palette[4], label='regression', linewidth = 3)
    ax.fill_between(x, regression_pred_mean - regression_pred_std/2, regression_pred_mean + regression_pred_std/2,
                    alpha=0.1, facecolor=colour_palette[4])

    ax.plot(real_v_mean, c = colour_palette[5], label='real', linewidth = 3)
    ax.fill_between(x, real_v_mean - real_v_std/2, real_v_mean + real_v_std/2,
                    alpha=0.1, facecolor=colour_palette[5])
    ax.set_title('DnL Average Problem Set', fontsize= 20)
    ax.set_xlabel('Samples', fontsize=20)
    ax.set_ylabel('Value', fontsize=20)
    ax.legend(fontsize= 15)
    fig.savefig("dnl_wknap_avg_forecasts.pdf")
    fig.savefig("dnl_wknap_avg_forecasts.png")



    fig2,ax2 = plt.subplots(tight_layout=True)


    x = [x for x in range(len(real_v_mean))]
    ax2.plot(spo_pred_mean_c12, c = colour_palette[0], label='c12',linewidth = 3)
    ax2.fill_between(x, spo_pred_mean_c12 - spo_pred_std_c12/2, spo_pred_mean_c12 + dnl_pred_std_c12/2,
                    alpha=0.1, facecolor=colour_palette[0])

    ax2.plot(spo_pred_mean_c48, c = colour_palette[1], label='c48',linewidth = 3)
    ax2.fill_between(x, spo_pred_mean_c48 - spo_pred_std_c48/2, spo_pred_mean_c48 + spo_pred_std_c48/2,
                    alpha=0.1, facecolor=colour_palette[1])


    ax2.plot(spo_pred_mean_c196, c = colour_palette[2], label='c196',linewidth = 3)
    ax2.fill_between(x, spo_pred_mean_c196 - spo_pred_std_c196/2, spo_pred_mean_c196 + spo_pred_std_c196/2,
                    alpha=0.1, facecolor=colour_palette[2])

    ax2.plot(spo_pred_mean_c220, c = colour_palette[3], label='c220',linewidth = 3)
    ax2.fill_between(x, spo_pred_mean_c220 - spo_pred_std_c220/2, spo_pred_mean_c220 + spo_pred_std_c220/2,
                    alpha=0.1, facecolor=colour_palette[3])



    ax2.plot(regression_pred_mean, c=colour_palette[4], label='regression',linewidth = 3)
    ax2.fill_between(x, regression_pred_mean - regression_pred_std/2, regression_pred_mean + regression_pred_std/2,
                    alpha=0.1, facecolor=colour_palette[4])

    ax2.plot(real_v_mean, c = colour_palette[5], label='real',linewidth = 3)
    ax2.fill_between(x, real_v_mean - real_v_std/2, real_v_mean + real_v_std/2,
                    alpha=0.1, facecolor=colour_palette[5])

    ax2.legend(fontsize= 15)
    ax2.set_title('SPO+ Average Problem Set', fontsize= 20)
    ax2.set_xlabel('Samples', fontsize= 20)
    ax2.set_ylabel('Value', fontsize=20)
    fig2.savefig("spo_wknap_avg_forecasts.pdf")
    fig2.savefig("spo_wknap_avg_forecasts.png")
    # plt.show()
    sorted_benchmarks = copy.deepcopy(Y_test_sets)
    # sorted_regret = sorted(regret, reverse=True)


    is_reverse = False


    # regret
    #




def regression_random_mse_test_knap(c=12,kfold = 0,is_shuffle = True):
    train_dict, val_dict, test_dict = prepare_icon_dataset(kfold=kfold, is_shuffle=is_shuffle)
    sns.set()
    X_train_sets = train_dict.get("X_sets")
    Y_train_sets = train_dict.get("Y_sets")

    X_train = train_dict.get("X")
    Y_train = train_dict.get("Y")
    weights_train_sets = train_dict.get("Weights_sets")


    X_test_sets = test_dict.get("X_sets")
    Y_test_sets = test_dict.get("Y_sets")
    weights_test_sets = test_dict.get("Weights_sets")
    X_test = test_dict.get("X")
    Y_test = test_dict.get("Y")

    mypool = multiprocessing.Pool(processes=4)
    sns.set()

    scikit_regression = linear_model.Ridge().fit(X_train, Y_train)

    pred = scikit_regression.predict(X_train)
    error = np.mean(((Y_train - pred)))
    error_sum = np.sum(np.abs((Y_train - pred)))
    std = np.std((Y_train - pred))
    max_norm = sum(abs(pred))
    pred_norm = [float(i) / max_norm for i in pred]
    print("Original Regression error: {}, std: {}".format(error, std))

    n_iter = 10
    noisy_Ys = []

    for n in range(n_iter):
        np.random.seed(int(time.time()))


        sum_y = sum(Y_train)
        weighted_mean = error
        t = int(time.time() * 1000.0)
        np.random.seed(((t & 0xff000000) >> 24) +
                    ((t & 0x00ff0000) >> 8) +
                    ((t & 0x0000ff00) << 8) +
                    ((t & 0x000000ff) << 24))

        this_noise = np.random.normal(error, std, Y_train.size)
        weighted_this_noise = np.random.normal(weighted_mean, std, Y_train.size)
        mean_sample = np.mean(Y_train)

        # noisy_Y = copy.deepcopy(Y_train)
        # np.random.shuffle(noisy_Y)
        noisy_Y = np.array([y + noise for y,noise in zip(Y_train,this_noise)])
        # noisy_Y = np.array([y + y*(noise)/(mean_sample) for y, noise in zip(Y_train, this_noise)])
        noisy_Ys.append(noisy_Y)

        diff = (Y_train - noisy_Y)
        this_err = np.mean((Y_train - noisy_Y))
        this_std = np.std((Y_train - noisy_Y))
        print("Iter: {}, error: {}, std: {}".format(n,this_err,this_std))





    noisy_regrets = []
    mypool = multiprocessing.Pool(processes=4)
    capacities = [12,24,48,96,120,196]
    regrets_by_capacities = [[] for c in capacities]
    original_regrets_by_capacities = []
    for ci,c in enumerate(capacities):
        opt_param = get_opt_params_knapsack(capacity=c)
        for n in range(n_iter):
            if n == 0:
                this_regret, __, predicted_solutions, solutions = scikit_get_regret(scikit_regression, X_train_sets, Y_train_sets,
                                                                               weights_train_sets, opt_params=opt_param,
                                                                               pool=mypool)

                print("Iter: {}, regret: {},".format("Without Noise",np.mean(this_regret)))
                original_regrets_by_capacities.append(this_regret)
            this_noisy_Y = noisy_Ys[n]
            noisy_Y_sets = [np.array(this_noisy_Y[i*48:((i+1)*48)]) for i in range(len(Y_train_sets))]
            this_regret, __, predicted_solutions, solutions = scikit_get_regret(None, X_train_sets, Y_train_sets,
                                                                           weights_train_sets, opt_params=opt_param, pred_Ys = noisy_Y_sets,
                                                                           pool=mypool)

            print("Iter: {}, regret: {},".format(n,np.mean(this_regret)))
            regrets_by_capacities[ci].append(this_regret)

    fig_icon, ax_icon = plt.subplots()
    for ci,c in enumerate(capacities):
        this_original_regret = np.array(original_regrets_by_capacities[ci])
        this_noisy_regret = np.array(regrets_by_capacities[ci])


        ax_icon.scatter(c, np.mean(this_original_regret), c='b', label='original regret')
        ax_icon.scatter(c, np.mean(this_noisy_regret), c='r', label='rand_noisy_regrets')

    ax_icon.set_title('Noise redistribution mean regret(N={})'.format(n_iter))
    ax_icon.set_xlabel("Capacities")
    ax_icon.set_ylabel("Regret")
    ax_icon.legend(labels=['orig,redist'])
    ax_icon.set_xticks(capacities)
    fig_icon.savefig("knapsack_redist_error.pdf")
    fig_icon.savefig("knapsack_redist_error.png")


    sorted_pred_regression = [x for _, x in sorted(zip(Y_train, pred))]
    # sorted_pred_regression_norm = [x for _, x in sorted(zip(Y_train, pred_norm))]


    # max_noisy_norm = sum(abs(noisy_Y))
    # pred_norm = [float(i) / max_noisy_norm for i in noisy_Y]
    # x = [x for x in range(len(Y_train))]
    # sorted_pred_regression_noisy = [x for _, x in sorted(zip(Y_train, noisy_Y))]
    # sorted_pred_regression_noisy_norm = [x for _, x in sorted(zip(Y_train, pred_norm))]
    # # plt.scatter([5,10],[20,10])


    # max_norm = sum(abs(Y_train.flatten()))
    # norm_Y = [float(i) / max_norm for i in Y_train.flatten()]
    #
    # fig_icon, ax_icon = plt.subplots()
    # ax_icon.scatter(x, sorted(norm_Y), c='r', label='Samples')
    #
    # ax_icon.set_title('Icon Test Samples (Sorted)')
    # ax_icon.set_xlabel("Sample No")
    # ax_icon.set_ylabel("Energy Price")
    # ax_icon.legend()




    # fig_icon_density, ax_icon_density = plt.subplots()
    #
    # sns.kdeplot(np.array(norm_Y), ax=ax_icon_density)
    # norm_Y_mean = np.mean(np.array(norm_Y))
    # norm_Y_median = np.median(np.array(norm_Y))
    # ax_icon_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5, label='mean')
    # ax_icon_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label='median')
    # ax_icon_density.legend()
    #
    #
    # fig_reg, ax_reg = plt.subplots(2,1)
    # ax_reg[0].scatter(x, sorted_pred_regression_norm, c='b', label='regression')
    # ax_reg[0].scatter(x, sorted(norm_Y), c='r', label='real')
    # ax_reg[0].set_title('Regression_and_Real_Data (Sorted)')
    # ax_reg[0].legend()
    #
    # ax_reg[1].scatter(x, sorted_pred_regression_noisy_norm, c='b', label='regression')
    # ax_reg[1].scatter(x, sorted(norm_Y), c='r', label='real')
    # ax_reg[1].set_title('Regression_and_Real_Data Noisy (Sorted)')
    # ax_reg[1].legend()
    #
    #
    #
    # fig_reg_density, ax_reg_density = plt.subplots()
    #
    # sns.kdeplot(np.array(sorted_pred_regression_norm), ax=ax_reg_density)
    # norm_Y_mean = np.mean(np.array(sorted_pred_regression_norm))
    # norm_Y_median = np.median(np.array(sorted_pred_regression_norm))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5, label='mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label='median')
    # ax_reg_density.legend()
    #
    #
    # fig1, axs1 = plt.subplots(4, 1)
    # axs1[0].scatter(x, sorted_pred_regression_norm, c='b', label='regression')
    # axs1[0].scatter(x, sorted(norm_Y), c='r', label='real')
    # axs1[0].set_title('Regression and Real Data (Sorted)')
    # axs1[0].legend()
    #
    # axs1[1].scatter(x, sorted_pred_regression_norm, c='b', label='Regression')
    # n = 100
    # moving_avg = np.cumsum(sorted_pred_regression_norm, dtype=float)
    # moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    # moving_avg = moving_avg[n - 1:] / n
    # axs1[1].scatter(x[n - 1:], moving_avg, c='r', label='Moving Avg')
    #
    # axs1[1].set_title('Regression (Sorted, Moving Average)')
    # axs1[1].legend()





    plt.show()

def regression_random_mse_test_sched(kfold = 0,is_shuffle = True):
    train_dict, val_dict, test_dict = prepare_icon_dataset(kfold=kfold, is_shuffle=is_shuffle, unit_weight= True)
    sns.set()
    X_train_sets = train_dict.get("X_sets")
    Y_train_sets = train_dict.get("Y_sets")

    X_train = train_dict.get("X")
    Y_train = train_dict.get("Y")
    weights_train_sets = train_dict.get("Weights_sets")


    X_test_sets = test_dict.get("X_sets")
    Y_test_sets = test_dict.get("Y_sets")
    weights_test_sets = test_dict.get("Weights_sets")
    X_test = test_dict.get("X")
    Y_test = test_dict.get("Y")

    mypool = multiprocessing.Pool(processes=4)
    sns.set()

    scikit_regression = linear_model.Ridge().fit(X_train, Y_train)

    pred = scikit_regression.predict(X_train)
    error = np.mean(((Y_train - pred)))
    error_sum = np.sum(np.abs((Y_train - pred)))
    std = np.std((Y_train - pred))
    max_norm = sum(abs(pred))
    pred_norm = [float(i) / max_norm for i in pred]
    print("Original Regression error: {}, std: {}".format(error, std))

    n_iter = 10
    noisy_Ys = []

    for n in range(n_iter):
        np.random.seed(int(time.time()))


        sum_y = sum(Y_train)
        weighted_mean = error
        t = int(time.time() * 1000.0)
        np.random.seed(((t & 0xff000000) >> 24) +
                    ((t & 0x00ff0000) >> 8) +
                    ((t & 0x0000ff00) << 8) +
                    ((t & 0x000000ff) << 24))

        this_noise = np.random.normal(error, std, Y_train.size)
        weighted_this_noise = np.random.normal(weighted_mean, std, Y_train.size)
        mean_sample = np.mean(Y_train)

        # noisy_Y = copy.deepcopy(Y_train)
        # np.random.shuffle(noisy_Y)
        noisy_Y = np.array([y + noise for y,noise in zip(Y_train,this_noise)])
        # noisy_Y = np.array([y + y*(noise)/(mean_sample) for y, noise in zip(Y_train, this_noise)])
        noisy_Ys.append(noisy_Y)

        diff = (Y_train - noisy_Y)
        this_err = np.mean((Y_train - noisy_Y))
        this_std = np.std((Y_train - noisy_Y))
        print("Iter: {}, error: {}, std: {}".format(n,this_err,this_std))





    noisy_regrets = []
    mypool = multiprocessing.Pool(processes=4)
    m=2
    loads = [400, 40,41,42,43,44,45,46,47,48]
    regrets_by_loads = [[] for c in loads]
    original_regrets_by_capacities = []
    for li,l in enumerate(loads):
        opt_param =     opt_params = get_icon_instance_params(l, folder_path='data/icon_instances/easy')
        for n in range(n_iter):
            if n == 0:
                this_regret, __, predicted_solutions, solutions = scikit_get_regret(scikit_regression, X_train_sets, Y_train_sets,
                                                                               weights_train_sets, opt_params=opt_param,
                                                                               pool=mypool)

                print("Iter: {}, regret: {},".format("Without Noise",np.mean(this_regret)))
                original_regrets_by_capacities.append(this_regret)
            this_noisy_Y = noisy_Ys[n]
            noisy_Y_sets = [np.array(this_noisy_Y[i*48:((i+1)*48)]) for i in range(len(Y_train_sets))]
            this_regret, __, predicted_solutions, solutions = scikit_get_regret(None, X_train_sets, Y_train_sets,
                                                                           weights_train_sets, opt_params=opt_param, pred_Ys = noisy_Y_sets,
                                                                           pool=mypool)

            print("Iter: {}, regret: {},".format(n,np.mean(this_regret)))
            regrets_by_loads[li].append(this_regret)

    fig_icon, ax_icon = plt.subplots()
    for ci,c in enumerate(loads):
        this_original_regret = np.array(original_regrets_by_capacities[ci])
        this_noisy_regret = np.array(regrets_by_loads[ci])


        ax_icon.scatter(c, np.mean(this_original_regret), c='b', label='original regret')
        ax_icon.scatter(c, np.mean(this_noisy_regret), c='r', label='rand_noisy_regrets')

    ax_icon.set_title('Noise redistribution mean regret(N={}, M={})'.format(n_iter,m))
    ax_icon.set_xlabel("Capacities")
    ax_icon.set_ylabel("Regret")
    ax_icon.legend(labels=['orig,redist'])
    ax_icon.set_xticks(loads)
    fig_icon.savefig("schedm{}_redist_error.pdf".format(m))
    fig_icon.savefig("schedm{}_redist_error.png".format(m))


    sorted_pred_regression = [x for _, x in sorted(zip(Y_train, pred))]
    # sorted_pred_regression_norm = [x for _, x in sorted(zip(Y_train, pred_norm))]


    # max_noisy_norm = sum(abs(noisy_Y))
    # pred_norm = [float(i) / max_noisy_norm for i in noisy_Y]
    # x = [x for x in range(len(Y_train))]
    # sorted_pred_regression_noisy = [x for _, x in sorted(zip(Y_train, noisy_Y))]
    # sorted_pred_regression_noisy_norm = [x for _, x in sorted(zip(Y_train, pred_norm))]
    # # plt.scatter([5,10],[20,10])


    # max_norm = sum(abs(Y_train.flatten()))
    # norm_Y = [float(i) / max_norm for i in Y_train.flatten()]
    #
    # fig_icon, ax_icon = plt.subplots()
    # ax_icon.scatter(x, sorted(norm_Y), c='r', label='Samples')
    #
    # ax_icon.set_title('Icon Test Samples (Sorted)')
    # ax_icon.set_xlabel("Sample No")
    # ax_icon.set_ylabel("Energy Price")
    # ax_icon.legend()




    # fig_icon_density, ax_icon_density = plt.subplots()
    #
    # sns.kdeplot(np.array(norm_Y), ax=ax_icon_density)
    # norm_Y_mean = np.mean(np.array(norm_Y))
    # norm_Y_median = np.median(np.array(norm_Y))
    # ax_icon_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5, label='mean')
    # ax_icon_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label='median')
    # ax_icon_density.legend()
    #
    #
    # fig_reg, ax_reg = plt.subplots(2,1)
    # ax_reg[0].scatter(x, sorted_pred_regression_norm, c='b', label='regression')
    # ax_reg[0].scatter(x, sorted(norm_Y), c='r', label='real')
    # ax_reg[0].set_title('Regression_and_Real_Data (Sorted)')
    # ax_reg[0].legend()
    #
    # ax_reg[1].scatter(x, sorted_pred_regression_noisy_norm, c='b', label='regression')
    # ax_reg[1].scatter(x, sorted(norm_Y), c='r', label='real')
    # ax_reg[1].set_title('Regression_and_Real_Data Noisy (Sorted)')
    # ax_reg[1].legend()
    #
    #
    #
    # fig_reg_density, ax_reg_density = plt.subplots()
    #
    # sns.kdeplot(np.array(sorted_pred_regression_norm), ax=ax_reg_density)
    # norm_Y_mean = np.mean(np.array(sorted_pred_regression_norm))
    # norm_Y_median = np.median(np.array(sorted_pred_regression_norm))
    # ax_reg_density.axvline(x=norm_Y_mean, c='k', ls='-', lw=2.5, label='mean')
    # ax_reg_density.axvline(x=norm_Y_median, c='orange', ls='--', lw=2.5, label='median')
    # ax_reg_density.legend()
    #
    #
    # fig1, axs1 = plt.subplots(4, 1)
    # axs1[0].scatter(x, sorted_pred_regression_norm, c='b', label='regression')
    # axs1[0].scatter(x, sorted(norm_Y), c='r', label='real')
    # axs1[0].set_title('Regression and Real Data (Sorted)')
    # axs1[0].legend()
    #
    # axs1[1].scatter(x, sorted_pred_regression_norm, c='b', label='Regression')
    # n = 100
    # moving_avg = np.cumsum(sorted_pred_regression_norm, dtype=float)
    # moving_avg[n:] = moving_avg[n:] - moving_avg[:-n]
    # moving_avg = moving_avg[n - 1:] / n
    # axs1[1].scatter(x[n - 1:], moving_avg, c='r', label='Moving Avg')
    #
    # axs1[1].set_title('Regression (Sorted, Moving Average)')
    # axs1[1].legend()





    plt.show()
if __name__ == "__main__":
    # for kfold in range(5):
    #     get_bias_var(kfold=kfold, is_shuffle=True)
    # for kfold in range(1):
    #     get_bias_var(kfold=kfold, is_shuffle=True)
    # capacities = None
    # # capacities = [12]
    #
    # bias_variance_sets(capacities, kfold=1, is_shuffle=True)


    # plot_impactful_sets_dnl(capacity=12, kfold=2, is_shuffle=True)
    # gen_icon_regret(instance = 1)
    # read_icon_regret()

    # gen_MSE_regret()
    # read_MSE_regret()

    # average_problem_set(kfold = 0, is_shuffle = True)
    mse_regression_tests(kfold =0, is_shuffle = True,c=12)
    mse_regression_tests(kfold=0, is_shuffle=True, c=196)
    # ppo_different_problems(is_shuffle=True, kfold=0)
    # ppo_different_problems_spo(is_shuffle=True, kfold=0)
    # c=12
    # regression_random_mse_test_knap(c=c)
    # regression_random_mse_test_sched()