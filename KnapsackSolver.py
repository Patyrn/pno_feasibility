from __future__ import print_function

import numpy as np
import torch
from ortools.algorithms import pywrapknapsack_solver

from Params import KNAPSACK



def knapsack_solver(values, weights, capacities):
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
            KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
    values = [0 if i < 0 else int(i*10000) for i in values]
    # values = values*100
    # print(values)
    # print(weights)
    # print(capacities)
    # print(values)

    solver.Init(values, weights, capacities)
    computed_value = solver.Solve()
    packed_items = []
    packed_weights = []
    total_weight = 0

    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            if values[i] > 0:
                packed_items.append(i)
                packed_weights.append(weights[0][i])
                total_weight += weights[0][i]
    return computed_value, packed_items

    # print('Total weight:', total_weight)
    # print('Packed items:', packed_items)
    # print('Packed_weights:', packed_weights)


def calculate_profit_from_items(items, values):
    total_value = 0
    for i in items:
        total_value = total_value + values[i]
    return total_value


def compute_profit_knapsack(X, Y, weights, model, opt_params):
    """
    Computes the average profit of a benchmark set
    :param const:
    :param weights:
    :param alphas: parameters of the model
    :param train_X: test set features
    :param train_Y: test set profits
    :param capacities: capacity of the problem
    :return: average_profit:
    """
    capacity = opt_params.get('capacity')
    alphas = model_params.get('alphas')
    const = model_params.get('const')

    profits = np.zeros(len(Y))
    index = range(len(Y))
    for i, benchmark_X, benchmark_Y, benchmark_weights in zip(index, X, Y, weights):
        profit = compute_profit_static_alphas(benchmark_X=benchmark_X, benchmark_Y=benchmark_Y,
                                                  weights=benchmark_weights, capacity=capacity, alphas=alphas,
                                                  const=const)
        profits[i] = profit

    return profits

def compute_profit_knapsack_pred(pred_Y,Y,weights,opt_params):
    capacity = opt_params.get('capacity')

    profits = np.zeros(len(Y))
    solutions = []
    index = range(len(Y))
    for i, benchmark_pred_Y, benchmark_Y, benchmark_weights in zip(index, pred_Y, Y, weights):
        profit,items = compute_profit_pred_y(benchmark_pred_Y=benchmark_pred_Y, benchmark_Y=benchmark_Y,
                                              weights=benchmark_weights, capacity=capacity)
        profits[i] = profit
        solutions.append(items)

    return profits,solutions

def compute_profit_pred_y(benchmark_pred_Y, benchmark_Y, weights, capacity):
    profit_alpha_k, __, items = compute_profit_knapsack_single_benchmark(benchmark_pred_Y, benchmark_Y, weights, capacity)
    return profit_alpha_k, items

def compute_profit_static_alphas(benchmark_X, benchmark_Y, weights, capacity, alphas, const):
    F_k = compute_C_k(benchmark_X, alphas, const, isSampling=False)
    profit_alpha_k, __, __ = compute_profit_knapsack_single_benchmark(F_k, benchmark_Y, weights, capacity)

    return profit_alpha_k


def compute_optimal_average_value_knapsack(Y, weights, opt_params):
    capacity = opt_params.get('capacity')
    objective_values = np.zeros(len(Y))
    solutions = []
    index =  range(len(Y))
    for i, benchmark_Y, benchmark_weights in zip(index, Y, weights):
        benchmark_Y = benchmark_Y.reshape(-1).tolist()
        benchmark_weights = benchmark_weights.reshape(-1).tolist()

        optimal_objective_value, predicted_opt_items = knapsack_solver(benchmark_Y, [benchmark_weights],
                                                                       capacity)
        solutions.append(predicted_opt_items)
        objective_values[i] = calculate_profit_from_items(predicted_opt_items, benchmark_Y)

        # solution = solveKnapsackProblem(benchmark_Y, [benchmark_weights], capacity, warmstart=None)
        # predicted_opt_items = np.asarray(solution['assignments'])
        # objective_values[i] = np.sum(benchmark_Y * predicted_opt_items)
    return objective_values, solutions


def compute_profit_knapsack_single_benchmark(F_k, Y, weights, capacities):
    """
    Computes the knapsack regret for a single benchmark. Returns both regret with respected predicted values and true values.
    Regret is defined as the loss of choosing a particular item set over the optimal item set.
    Item set choice is determined accordingly to the predicted values. However from there we have two regret calculations

    Predicted_regret: we use the item set calculated using predicted values, but when calculating cost we still use predicted item values. We propose this regret is convex therefore we can use it to find transition points.
    Regret(True regret): we use the item set calculated using predicted values, but when calculating cost we use true values. This regret is used to evaluate our algorithm.

    :param F_k: Predicted values for a alpha_k value
    :param Y: True values
    :param weights: weights of the items
    :param capacities: capacity of the knapsack problem
    :return: regret(int), predicted_regret(int)
    """
    F_k = (F_k.reshape(-1)).tolist()
    Y = Y.reshape(-1).tolist()
    weights = weights.reshape(-1).tolist()

    __, predicted_opt_items = knapsack_solver(F_k, [weights], capacities)
    predicted_profit = calculate_profit_from_items(predicted_opt_items, F_k)
    profit = calculate_profit_from_items(predicted_opt_items, Y)

    # solution = solveKnapsackProblem(F_k, [weights], capacities, warmstart=None)
    # # print(solution['assignments'])
    # # print(np.asarray(solution['assignments']))
    # predicted_opt_items = np.asarray(solution['assignments'])
    # #
    # predicted_profit = np.sum(F_k * predicted_opt_items)
    # profit = np.sum(Y * predicted_opt_items)
    return profit, predicted_profit, predicted_opt_items


def compute_sampled_alpha_profit_knapsack(pred_ys, benchmark_y, weights, capacity):
    """
    Computes regrets of a single benchmark given a range of alpha samples
    :param benchmark_X:
    :param benchmark_Y:
    :param benchmark_weights:
    :param capacities:
    :param alphas: vector of all alpha
    :param const: regression constant
    :param sample_space: range of alphas
    :param k: indicator of the current alpha
    :return:sampled_profits(nparray), sampled_predicted_profits(nparray)
    """
    benchmark_size = pred_ys.shape[0]
    sample_length = pred_ys.shape[1]
    POVS = np.zeros(sample_length)
    TOVS = np.zeros(sample_length)
    for i, pred_y in enumerate(pred_ys.T):
        # tov, pov, predicted_opt_items = compute_profit_knapsack_single_benchmark(pred_y, benchmark_y,
        #                                                                          weights,
        #                                                                          capacities=[10])

        tov, pov, predicted_opt_items = compute_profit_knapsack_single_benchmark(pred_y,
                                                             Y=benchmark_y,
                                                             weights=weights,
                                                             capacities=capacity)

        POVS[i] = pov
        TOVS[i] = tov
    return TOVS, POVS, predicted_opt_items, pred_ys


def get_opt_params_knapsack(capacity=24):
    params = {'capacity': [capacity],
              'solver': KNAPSACK}
    return params


if __name__ == '__main__':
    print("main")
