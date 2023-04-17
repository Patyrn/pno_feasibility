# from IconEasySolver import compute_profit_scheduling_easy, compute_optimal_average_value_icon_easy, \
#     compute_sampled_alpha_profit_icon_easy

import numpy as np
import torch

from IconEasySolver import compute_optimal_average_value_icon_energy, \
    compute_icon_energy_single_benchmark, compute_profit_icon_pred, compute_sampled_alpha_icon_energy
from KnapsackSolver import compute_optimal_average_value_knapsack, \
    compute_profit_knapsack, compute_profit_knapsack_single_benchmark, compute_profit_knapsack_pred, \
    compute_sampled_alpha_profit_knapsack
from Params import KNAPSACK, ICON_SCHEDULING_EASY


# def get_optimization_objective_for_samples(benchmark_X, benchmark_Y, pred_Y, benchmark_weights, opt_params,
#                                            sample_space,
#                                            mpPool=None):
#     """
#     Computes regrets of a single benchmark given a range of alpha samples
#     :param benchmark_X:
#     :param benchmark_Y:
#     :param benchmark_weights:
#     :param capacities:
#     :param alphas: vector of all alpha
#     :param const: regression constant
#     :param sample_space: range of alphas
#     :param k: indicator of the current alpha
#     :return:sampled_profits(nparray), sampled_predicted_profits(nparray)
#     """
#
#     sampled_profits = []
#     sampled_predicted_profits = []
#     # for tmp_alpha in sample_space:
#
#     # Compute predicted cost F_k for each alpha_k in searching space for the current k
#     # serial
#
#     for tmp_alpha in sample_space:
#         profit, predicted_profit = compute_objective_value_single_benchmarks(tmp_alpha, benchmark_X,
#                                                                              benchmark_Y,
#                                                                              benchmark_weights, pred_Y,
#
#                                                                              opt_params)
#         sampled_profits.append(profit)
#         sampled_predicted_profits.append(predicted_profit)
#
#     # parallel
#
#     # mypool = mp.Pool(processes=min(8, mp.cpu_count()))
#     # mypool = mpPool
#     #
#     # map_func = partial(compute_objective_value_single_benchmarks, benchmark_X=benchmark_X, Y=benchmark_Y,
#     #                    weights=benchmark_weights, C_k=C_k, opt_params=opt_params, k=k)
#     # results = mypool.map(map_func, sample_space)
#     #
#     # for profit, predicted_profit in results:
#     #     sampled_profits.append(profit)
#     #     sampled_predicted_profits.append(predicted_profit)
#
#     sampled_profits = np.array(sampled_profits)
#     sampled_predicted_profits = np.array(sampled_predicted_profits)
#     return sampled_profits, sampled_predicted_profits


def compute_objective_value_single_benchmarks(pred_Y, Y, weights,
                                              opt_params):
    # Compute predicted cost F_k for each alpha_k in searching space for the current k

    sampled_profits = -1
    sampled_predicted_profits = -1
    solver = opt_params.get('solver')
    if solver == KNAPSACK:
        capacity = opt_params.get('capacity')
        sampled_profits, sampled_predicted_profits, __ = compute_profit_knapsack_single_benchmark(pred_Y, Y, weights,
                                                                                                  capacity)
    elif solver == ICON_SCHEDULING_EASY:
        sampled_profits, sampled_predicted_profits, __ = compute_icon_energy_single_benchmark(pred_Y, Y, opt_params)

    return sampled_profits, sampled_predicted_profits


#
def get_optimization_objective_for_samples(benchmark_x, benchmark_y, weights, sample_params,
                                           param_ind,
                                           model, layer_no, bias):
    sampled_profits = -1
    sampled_predicted_profits = -1
    solver = model.opt_params.get('solver')
    pred_ys = get_pred_ys(benchmark_x, benchmark_y, model, layer_no, param_ind, sample_params, bias)
    if solver == KNAPSACK:
        sampled_profits, sampled_predicted_profits, decisions, pred_ys = compute_sampled_alpha_profit_knapsack(
            pred_ys, benchmark_y, weights, model.opt_params.get("capacity"))
    elif solver == ICON_SCHEDULING_EASY:
        sampled_profits, sampled_predicted_profits, decisions, pred_ys = compute_sampled_alpha_icon_energy(pred_ys, benchmark_y, model.opt_params)

    return sampled_profits, sampled_predicted_profits, decisions, pred_ys


def get_optimization_objective(Y, weights, opt_params, pred_Y=None):
    solver = opt_params.get('solver')
    if solver == KNAPSACK:
        return compute_profit_knapsack_pred(pred_Y, Y, weights, opt_params)
    elif solver == ICON_SCHEDULING_EASY:
        return compute_profit_icon_pred(pred_Y, Y, opt_params)
    else:
        print('error')


def get_optimal_average_objective_value(Y, weights, opt_params):
    solver = opt_params.get('solver')
    if solver == KNAPSACK:
        return compute_optimal_average_value_knapsack(Y, weights, opt_params)
    elif solver == ICON_SCHEDULING_EASY:
        return compute_optimal_average_value_icon_energy(Y, opt_params)
    else:
        print('error')


def get_pred_ys(benchmark_x, benchmark_y, model, layer_no, param_ind, sample_params, bias):
    layer = model.get_layer(layer_no)
    benchmark_size = len(benchmark_y)
    sample_length = len(sample_params)
    pred_ys = np.zeros((benchmark_size, sample_length))
    for i, param in enumerate(sample_params):
        with torch.no_grad():
            if bias:
                layer.bias[param_ind] = torch.from_numpy(np.array(param)).float()
            else:
                # print("weight before: {} model: {}".format(layer.weight[param_ind], model.get_layer(1).weight[param_ind]))
                layer.weight[param_ind] = torch.from_numpy(np.array(param)).float()
                # print("weight after: {} model: {}".format(layer.weight[param_ind], model.get_layer(1).weight[param_ind]))
        # current_weight = temp_model.fc1.weight.detach()
        # current_weight = param
        pred_y = model.forward(torch.from_numpy(benchmark_x).float()).detach().numpy()
        pred_ys[:, i] = pred_y.T
    return pred_ys
