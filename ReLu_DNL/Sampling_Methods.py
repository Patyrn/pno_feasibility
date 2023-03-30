import copy
import math
import time

import numpy as np

from ReLu_DNL.LinearFunction import LinearFunction

from Solver import get_optimization_objective_for_samples

from Utils import Interval, create_transition_points_from_intervals, TransitionPoint

import heapq


"""
This file contains the Sampler object and functions related to compute optimization probelem of benchmarks
"""

DIVIDE_AND_CONQUER = 'DIVIDE_AND_CONQUER'
DIVIDE_AND_CONQUER_MAX = 'DIVIDE_AND_CONQUER_MAX'
DIVIDE_AND_CONQUER_GREEDY = 'DIVIDE_AND_CONQUER_GREEDY'
DIVIDE_AND_CONQUER_GREEDY_MERGED = 'DIVIDE_AND_CONQUER_GREEDY_MERGED'

EXHAUSTIVE = 'EXHAUSTIVE'
EXHAUSTIVE_MAX = 'EXHAUSTIVE_MAX'
SAMPLE_RANGE_MULTIPLIER = 3

MID_TRANSITION_POINT_SELECTION = 'MID_POINT'
EDGE_TRANSITION_POINT_SELECTION = 'EDGE'

RUN_TIME_CONSTANT = 3


class runtime_param_ind(object):
    def __init__(self, runtime, layer_no, param_ind, bias = None, update_count = 0):
        self.runtime = runtime
        self.layer_no = layer_no
        self.param_ind = param_ind
        self.update_count = update_count
        if bias is None:
            self.bias = False
        else:
            self.bias = bias

    def increase_count(self):
        self.update_count += 1
    def __eq__(self, other):
        return self.runtime == other.runtime

    def __lt__(self, other):
        return self.runtime < other.runtime


class Sampler:
    """
    Sampler class: This class/object is used for deciding sampling space and finding transition points.
    """

    def __init__(self, max_step_size_magnitude=0, min_step_size_magnitude=-1, step_size_divider=2,
                 sampling_method="DIVIDE_AND_CONQUER", transition_point_selection=MID_TRANSITION_POINT_SELECTION,
                 opt_params=None, model = None):
        self.max_step_size_magnitude = max_step_size_magnitude
        self.min_step_size_magnitude = min_step_size_magnitude
        self.step_size_divider = step_size_divider
        self.sampling_method = sampling_method
        self.transition_point_selection = transition_point_selection
        self.opt_params = opt_params
        self.time_in_loop = 0

        array = [runtime_param_ind(runtime=3, layer_no=layer_no, param_ind=(i, j)) for
                 layer_no in range(len(model.layers)) for j in
                 range(model.get_layer(layer_no).in_features) for i in
                 range(model.get_layer(layer_no).out_features)]

        biases = [runtime_param_ind(runtime=3, layer_no=layer_no, param_ind=j, bias=True) for
                 layer_no in range(len(model.layers)) for j in
                 range(len(model.get_layer(layer_no).bias))]
        array.extend(biases)
        heapq.heapify(array)
        self.greedy_run_times = array

    def divide_and_conquer_search(self, benchmark_x, benchmark_y, weights, param_ind, model, layer_no=None, profit=-20,
                                  bias=False):
        """
              This methods approaches the transition point search from an exhaustive approach. The sample size is bounded
              by max_step_size_magnitude. Step size is determined by min_step_size_magnitude. Each sample point is used
              for profit calculations. And then these profits and sample points are used to extract transition points.


              :param alphas: parameters of the model
              :param k: the index of the parameter that is optimized
              :param const: constant of the model
              :param train_X: test set features
              :param train_Y: test set profits
              :param train_weights: test set weights
              :param capacities: capacity of the problem

              :return: item_sets(list): list of item sets, transition_points: a list of transition points,
              predicted_profits(list): predicted profits of the sampling points
              profits(list): true profits of the sampling points
              sample_space(list): all sample points
              IMPORTANT: The lists are 2d lists. Assume a list(M,N) where M is the iterations and N is the sample points.
              To get the final results you would look for list(m,:).
              """
        # print(param_ind)
        copy_time = time.time()
        temp_model = copy.deepcopy(model)
        copy_finish = time.time() - copy_time
        sampling_time = time.time()
        if bias:
            current_param = temp_model.get_layer(layer_no).bias[param_ind].detach().numpy()
        else:
            current_param = temp_model.get_layer(layer_no).weight[param_ind].detach().numpy()
        # print("dnc param: {}".format(current_param))
        start_ind = current_param - abs(current_param) * 10
        end_ind = current_param + abs(current_param) * 10

        M = math.ceil(math.log(10 ** (self.max_step_size_magnitude - self.min_step_size_magnitude),
                               self.step_size_divider)) + 1

        # Initialize first sample range parameters
        # print("before initialize current_param: {}".format(current_param))
        if current_param != 0:
            # print("min_step_size value should be", abs(current_param * (10 ** self.min_step_size_magnitude)))
            min_step_size = abs(current_param * (10 ** self.min_step_size_magnitude))
            # print("min_step_size value should is", min_step_size)
            sample_range = abs(current_param * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
            # print('in if, min_step_size: {}, sample_range: {}'.format(min_step_size, sample_range))
        else:
            sample_range = abs(1 * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
            min_step_size = 1 * (10 ** self.min_step_size_magnitude)
            # print('in else, min_step_size: {}, sample_range: {}'.format(min_step_size, sample_range))
        step_size = sample_range / self.step_size_divider
        interval_mid_point = current_param
        interval_start = interval_mid_point - sample_range / 2
        interval_end = interval_mid_point + sample_range / 2

        # Initialize lists
        sample_spaces = [[] for i in range(M)]
        POVS = [[] for i in range(M)]
        TOVS = [[] for i in range(M)]
        transition_intervals = [[] for i in range(M)]
        intervals = [[] for i in range(M + 1)]

        intervals[0] = [Interval(TransitionPoint(interval_start, 0, 0), TransitionPoint(interval_end, 0, 0))]
        iteration = 0
        final_intervals = []
        sampling_time = time.time()
        for i in range(M):
            for interval_no, interval in enumerate(intervals[i]):
                iteration += 1
                start_index = interval.starting_point.x
                end_index = interval.ending_point.x
                sample_size = self.step_size_divider + 1
                # print("M: {}, Interval: {} Start: {}, End {}".format(M, i, start_index, end_index))
                step_size = abs(end_index - start_index) / self.step_size_divider
                if step_size > min_step_size:
                    sample_space = [start_index + step_size * j for j in range(sample_size)]
                    # if round(end_index, 3) > round(sample_space[-1], 3):
                    #     sample_space[-1] = end_index
                    sample_spaces[i].extend(sample_space)
                    # compute the profits of the sample points
                    if i == 0:
                        benchmark_TOVS, benchmark_POVS, __, pred_ys = get_optimization_objective_for_samples(
                            benchmark_x,
                            benchmark_y,
                            weights,
                            sample_space,
                            param_ind,
                            temp_model, layer_no, bias)
                    else:
                        # cache previous sample solutions
                        benchmark_TOVS, benchmark_POVS, __, pred_ys = get_optimization_objective_for_samples(
                            benchmark_x,
                            benchmark_y,
                            weights,
                            sample_space[1:-1],
                            param_ind,
                            temp_model, layer_no, bias)

                        benchmark_TOVS = np.hstack(
                            (interval.starting_point.true_profit, benchmark_TOVS, interval.ending_point.true_profit))

                        benchmark_POVS = np.hstack((interval.starting_point.predicted_profit, benchmark_POVS,
                                                    interval.ending_point.predicted_profit))

                    this_transition_intervals, dead_intervals = find_transition_intervals(sample_space, benchmark_POVS,
                                                                                          benchmark_TOVS)
                    transition_intervals[i].extend(this_transition_intervals)
                    final_intervals.extend(dead_intervals)
                    intervals[i + 1] = transition_intervals[i]
                    POVS[i].extend(benchmark_POVS)
                    TOVS[i].extend(benchmark_TOVS)
                else:
                    final_intervals.append(interval)
                    # decision_policies.append(predicted_opt_items)

        transition_points = create_transition_points_from_intervals(final_intervals,
                                                                    selection_method=self.transition_point_selection)
        # # Debugging plots
        # print("NUmber of transition points:", len(transition_points))
        # debug_number_of_samples = 10**(self.max_step_size_magnitude - self.min_step_size_magnitude)
        # debug_step_size = (interval_end - interval_start) / debug_number_of_samples
        # debug_samples = [interval_start + x*debug_step_size for x in range(debug_number_of_samples)]
        # debug_benchmark_TOVS, debug_benchmark_POVS, __, debug_pred_ys = get_optimization_objective_for_samples(
        #                     benchmark_x,
        #                     benchmark_y,
        #                     weights,
        #                     debug_samples,
        #                     param_ind,
        #                     temp_model, layer_no)
        #
        # plt.scatter(debug_samples, debug_benchmark_TOVS)
        # plt.scatter(current_param, transition_points[0].true_profit, c='r')
        # for transition_point in transition_points:
        #     plt.scatter(transition_point.x, transition_point.true_profit)
        # plt.show()
        #     plt.subplot(M, 2, 2 * i + 1)
        #     plt.xlim([interval_start, interval_end])
        #     plt.scatter(sample_spaces[i], POVS[i])
        #     # for transition_point in transition_points:
        #     #     plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
        #     #                 transition_point.starting_point.predicted_profit, c='r')
        #     plt.title('POV')
        #     plt.subplot(M, 2, (i + 1) * 2)
        #     plt.xlim([interval_start, interval_end])
        #     plt.title('TOV')
        #     plt.scatter(sample_spaces[i], TOVS[i])
        #     for transition_point in transition_points:
        #         plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
        #                     transition_point.starting_point.true_profit, c='r')
        #     plt.subplot(3, 2, 3)
        #     plt.title('Derivatives')
        #     shifted_POVS = np.roll(benchmark_POVS, 1)
        #     der = (benchmark_POVS[1:99] - shifted_POVS[1:99]) / (sample_params[1] - sample_params[2])
        #     plt.scatter(sample_params[1:99], der)
        #     shifted_der = np.roll(der, 1)
        #     d2 = (der[2:98] - shifted_der[2:98]) / (sample_params[1] - sample_params[2])
        #
        #     plt.subplot(3, 2, 4)
        #     plt.title('Second Derivatives')
        #     plt.scatter(sample_params[2:98], d2)
        #     # for i in range(96):
        #     #     print("decision policy({}): {}, der2: {}".format(i, decision_policies[i + 2], np.round(d2[i], 3)))
        #
        #     plt.subplot(3, 2, 5)
        #     plt.title('Individual Item Predictions')
        #     for i in range(50):
        #         plt.plot(sample_params, pred_ys[i, :])
        #     plt.subplot(3, 2, 6)
        #     plt.title("MSE")
        #     mse = []
        #     for i in range(100):
        #         mse.append(np.mean((benchmark_y - pred_ys[:, i]) ** 2))
        #     plt.scatter(sample_params, mse)
        #
        #     plt.figure()
        #     for i in range(100):
        #         plt.title("pred_values vs real values")
        #         plt.subplot(10,10,i+1)
        #         plt.scatter(np.array([i for i in range(50)]), pred_ys[:,i], c='r')
        #         plt.scatter(np.array([i for i in range(50)]), benchmark_y, c='b')
        # plt.show()

        sampling_time = time.time() - sampling_time
        # print("sampling_end_time:", round(sampling_time, 6), "copy time:", (copy_finish), "iteration: ", iteration)

        return transition_points, final_intervals, sample_spaces[-1], POVS[-1], TOVS[-1]

    def divide_and_conquer_greedy_search(self, benchmark_x, benchmark_y, weights, param, model, layer_no=None,
                                         profit=-20, bias=False):
        """
        This methods approaches the transition point search from a divide and conquer approach. Same as the vanilla divide and conquer method,
        but stops at the first transition point which improves the current profit


        :param alphas: parameters of the model
        :param k: the index of the parameter that is optimized
        :param const: constant of the model
        :param train_X: test set features
        :param train_Y: test set profits
        :param train_weights: test set weights
        :param capacities: capacity of the problem

        :return: item_sets(list): list of item sets, transition_points: a list of transition points,
        predicted_profits(list): predicted profits of the sampling points
        profits(list): true profits of the sampling points
        sample_space(list): all sample points
        IMPORTANT: The lists are 2d lists. Assume a list(M,N) where M is the iterations and N is the sample points.
        To get the final results you would look for list(m,:).
        """
        """
               This methods approaches the transition point search from an exhaustive approach. The sample size is bounded
               by max_step_size_magnitude. Step size is determined by min_step_size_magnitude. Each sample point is used
               for profit calculations. And then these profits and sample points are used to extract transition points.


               :param alphas: parameters of the model
               :param k: the index of the parameter that is optimized
               :param const: constant of the model
               :param train_X: test set features
               :param train_Y: test set profits
               :param train_weights: test set weights
               :param capacities: capacity of the problem

               :return: item_sets(list): list of item sets, transition_points: a list of transition points,
               predicted_profits(list): predicted profits of the sampling points
               profits(list): true profits of the sampling points
               sample_space(list): all sample points
               IMPORTANT: The lists are 2d lists. Assume a list(M,N) where M is the iterations and N is the sample points.
               To get the final results you would look for list(m,:).
               """
        # print(param_ind)
        copy_time = time.time()
        temp_model = copy.deepcopy(model)
        copy_finish = time.time() - copy_time
        sampling_time = time.time()
        # if self.greedy_run_times is None:
        #     # first_layer_run_time_array = np.zeros(
        #     #     (temp_model.get_layer(1).out_features, temp_model.get_layer(1).in_features))
        #     # second_layer_run_time_array = np.zeros((temp_model.get_layer(2).out_features,
        #     #                                         temp_model.get_layer(2).in_features))
        #     # self.greedy_run_times = [first_layer_run_time_array, second_layer_run_time_array]
        #     array = [runtime_param_ind(runtime=0, layer_no=layer_no, param_ind=(i, j)) for
        #              layer_no in [1, 2] for i in
        #              range(temp_model.get_layer(layer_no).in_features) for j in range(temp_model.get_layer(layer_no).out_features)]
        #     heapq.heapify(array)
        #     self.greedy_run_times = array

        if bias:
            current_param = temp_model.get_layer(layer_no).bias[param.param_ind].detach().numpy()
        else:
            current_param = temp_model.get_layer(layer_no).weight[param.param_ind].detach().numpy()
        # print("dnc param: {}".format(current_param))
        start_ind = current_param - abs(current_param) * 10
        end_ind = current_param + abs(current_param) * 10

        M = math.ceil(math.log(10 ** (self.max_step_size_magnitude - self.min_step_size_magnitude),
                               self.step_size_divider)) + 1

        # Initialize first sample range parameters
        # print("before initialize current_param: {}, type: {}".format(current_param,current_param.shape))

        if current_param != 0:
            # print("min_step_size value should be", abs(current_param * (10 ** self.min_step_size_magnitude)))
            min_step_size = abs(current_param * (10 ** self.min_step_size_magnitude))
            # print("min_step_size value should is", min_step_size)
            sample_range = abs(current_param * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
            # print('in if, min_step_size: {}, sample_range: {}'.format(min_step_size, sample_range))
        else:
            sample_range = abs(1 * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
            min_step_size = 1 * (10 ** self.min_step_size_magnitude)
            # print('in else, min_step_size: {}, sample_range: {}'.format(min_step_size, sample_range))
        interval_mid_point = current_param
        interval_start = interval_mid_point - sample_range / 2
        interval_end = interval_mid_point + sample_range / 2
        # Initialize lists
        sample_spaces = [[] for i in range(M)]
        POVS = [[] for i in range(M)]
        TOVS = [[] for i in range(M)]
        transition_intervals = [[] for i in range(M)]
        intervals = [[] for i in range(M + 1)]

        intervals[0] = [Interval(TransitionPoint(interval_start, 0, 0), TransitionPoint(interval_end, 0, 0))]
        final_intervals = []
        iteration = 0
        greedy_sampling_time = time.time()
        for i in range(M):
            for interval_no, interval in enumerate(intervals[i]):

                iteration += 1
                start_index = interval.starting_point.x

                end_index = interval.ending_point.x
                sample_size = self.step_size_divider + 1
                # print("M: {}, Interval: {} Start: {}, End {}".format(M, i, start_index, end_index))
                step_size = abs(end_index - start_index) / self.step_size_divider

                if step_size > min_step_size:
                    # print('start index:', start_index, "step size:", step_size)
                    sample_space = [start_index + step_size * j for j in range(sample_size)]
                    # print("sample space", sample_space)
                    # if round(end_index, 3) > round(sample_space[-1], 3):
                    #     sample_space[-1] = end_index
                    sample_spaces[i].extend(sample_space)
                    # compute the profits of the sample points
                    if i == 0:
                        benchmark_TOVS, benchmark_POVS, __, pred_ys = get_optimization_objective_for_samples(
                            benchmark_x,
                            benchmark_y,
                            weights,
                            sample_space,
                            param.param_ind,
                            temp_model, layer_no, bias)
                        # initialize profit to the profit with the current value, may cache it to increase speed
                        profit = benchmark_TOVS[1]
                        original_param = TransitionPoint(x=sample_space[1], true_profit=profit,
                                                         predicted_profit=benchmark_POVS[1])
                    else:
                        # cache previous sample solutions
                        benchmark_TOVS, benchmark_POVS, __, pred_ys = get_optimization_objective_for_samples(
                            benchmark_x,
                            benchmark_y,
                            weights,
                            sample_space[1:-1],
                            param.param_ind,
                            temp_model, layer_no, bias)

                        benchmark_TOVS = np.hstack(
                            (interval.starting_point.true_profit, benchmark_TOVS, interval.ending_point.true_profit))

                        benchmark_POVS = np.hstack((interval.starting_point.predicted_profit, benchmark_POVS,
                                                    interval.ending_point.predicted_profit))

                    if max(benchmark_TOVS) > profit:
                        index = np.argmax(benchmark_TOVS)
                        transition_point = TransitionPoint(x=sample_space[index], true_profit=benchmark_TOVS[index],
                                                           predicted_profit=benchmark_POVS[index])
                        transition_points = [transition_point]
                        greedy_sampling_end_time = time.time() - greedy_sampling_time
                        # print("GOOD POINT greedy_sampling_end_time:", round(greedy_sampling_end_time, 6), "iteration: ", iteration)
                        sampling_time = time.time() - sampling_time
                        # print("sampling_end_time:", round(sampling_time, 6), "copy time:", copy_finish, "iteration: ",
                        #       iteration)
                        return transition_points, final_intervals, sample_spaces[-1], POVS[-1], TOVS[-1], greedy_sampling_end_time

                    greedy_sampling_end_time = time.time() - greedy_sampling_time
                    if greedy_sampling_end_time > param.runtime * RUN_TIME_CONSTANT:
                        transition_points = [original_param]
                        # print("TIMEOUT greedy_sampling_end_time:", round(greedy_sampling_end_time, 6), "iteration: ", iteration)
                        sampling_time = time.time() - sampling_time
                        # print("sampling_end_time:", round(sampling_time, 6), "copy time:", (copy_finish),
                        # "iteration: ", iteration)
                        return transition_points, final_intervals, sample_spaces[-1], POVS[-1], TOVS[-1], greedy_sampling_end_time

                    this_transition_intervals, dead_intervals = find_transition_intervals(sample_space, benchmark_POVS,
                                                                                          benchmark_TOVS)
                    transition_intervals[i].extend(this_transition_intervals)
                    intervals[i + 1] = transition_intervals[i]
                    POVS[i].extend(benchmark_POVS)
                    TOVS[i].extend(benchmark_TOVS)
                else:
                    final_intervals.append(interval)
                    greedy_sampling_end_time = time.time() - greedy_sampling_time

                    if greedy_sampling_end_time > param.runtime * RUN_TIME_CONSTANT:
                        transition_points = [original_param]
                        # print("TIMEOUT greedy_sampling_end_time:", round(greedy_sampling_end_time, 6), "iteration: ", iteration)
                        sampling_time = time.time() - sampling_time
                        # print("sampling_end_time:", round(sampling_time, 6), "copy time:", (copy_finish), "iteration: ",
                        #       iteration)
                        return transition_points, final_intervals, sample_spaces[-1], POVS[-1], TOVS[-1], greedy_sampling_end_time
                    # decision_policies.append(predicted_opt_items)
        greedy_sampling_end_time = time.time() - greedy_sampling_time
                # print("greedy_sampling_end_time:", round(greedy_sampling_end_time, 6), "iteration: ", iteration)
        transition_points = [original_param]
        # # Debugging plots
        # print("NUmber of transition points:", len(transition_points))
        # debug_number_of_samples = 10**(self.max_step_size_magnitude - self.min_step_size_magnitude)
        # debug_step_size = (interval_end - interval_start) / debug_number_of_samples
        # debug_samples = [interval_start + x*debug_step_size for x in range(debug_number_of_samples)]
        # debug_benchmark_TOVS, debug_benchmark_POVS, __, debug_pred_ys = get_optimization_objective_for_samples(
        #                     benchmark_x,
        #                     benchmark_y,
        #                     weights,
        #                     debug_samples,
        #                     param_ind,
        #                     temp_model, layer_no)
        #
        # plt.scatter(debug_samples, debug_benchmark_TOVS)
        # plt.scatter(current_param, transition_points[0].true_profit, c='r')
        # for transition_point in transition_points:
        #     plt.scatter(transition_point.x, transition_point.true_profit)
        # plt.show()
        #     plt.subplot(M, 2, 2 * i + 1)
        #     plt.xlim([interval_start, interval_end])
        #     plt.scatter(sample_spaces[i], POVS[i])
        #     # for transition_point in transition_points:
        #     #     plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
        #     #                 transition_point.starting_point.predicted_profit, c='r')
        #     plt.title('POV')
        #     plt.subplot(M, 2, (i + 1) * 2)
        #     plt.xlim([interval_start, interval_end])
        #     plt.title('TOV')
        #     plt.scatter(sample_spaces[i], TOVS[i])
        #     for transition_point in transition_points:
        #         plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
        #                     transition_point.starting_point.true_profit, c='r')
        #     plt.subplot(3, 2, 3)
        #     plt.title('Derivatives')
        #     shifted_POVS = np.roll(benchmark_POVS, 1)
        #     der = (benchmark_POVS[1:99] - shifted_POVS[1:99]) / (sample_params[1] - sample_params[2])
        #     plt.scatter(sample_params[1:99], der)
        #     shifted_der = np.roll(der, 1)
        #     d2 = (der[2:98] - shifted_der[2:98]) / (sample_params[1] - sample_params[2])
        #
        #     plt.subplot(3, 2, 4)
        #     plt.title('Second Derivatives')
        #     plt.scatter(sample_params[2:98], d2)
        #     # for i in range(96):
        #     #     print("decision policy({}): {}, der2: {}".format(i, decision_policies[i + 2], np.round(d2[i], 3)))
        #
        #     plt.subplot(3, 2, 5)
        #     plt.title('Individual Item Predictions')
        #     for i in range(50):
        #         plt.plot(sample_params, pred_ys[i, :])
        #     plt.subplot(3, 2, 6)
        #     plt.title("MSE")
        #     mse = []
        #     for i in range(100):
        #         mse.append(np.mean((benchmark_y - pred_ys[:, i]) ** 2))
        #     plt.scatter(sample_params, mse)
        #
        #     plt.figure()
        #     for i in range(100):
        #         plt.title("pred_values vs real values")
        #         plt.subplot(10,10,i+1)
        #         plt.scatter(np.array([i for i in range(50)]), pred_ys[:,i], c='r')
        #         plt.scatter(np.array([i for i in range(50)]), benchmark_y, c='b')
        # plt.show()
        sampling_time = time.time() - sampling_time
        # print("step_size:", round(sampling_time, 6), "copy time:", (copy_finish), "iteration: ", iteration)
        return transition_points, final_intervals, sample_spaces[-1], POVS[-1], TOVS[-1], greedy_sampling_end_time

    def update_greedy_runtime(self, runtime, param):
        param_runtime = runtime_param_ind(runtime, param.layer_no, param.param_ind, param.bias, update_count = param.update_count)
        heapq.heappush(self.greedy_run_times, param_runtime)

    def divide_and_conquer_greedy_search_merged(self, benchmarks_x, benchmarks_y, weights, param, model, layer_no=None,
                                         profit=-20, bias=False):
        """
             This methods approaches the transition point search from a divide and conquer approach. Same as the vanilla divide and conquer method,
             but stops at the first transition point which improves the current profit


             :param alphas: parameters of the model
             :param k: the index of the parameter that is optimized
             :param const: constant of the model
             :param train_X: test set features
             :param train_Y: test set profits
             :param train_weights: test set weights
             :param capacities: capacity of the problem

             :return: item_sets(list): list of item sets, transition_points: a list of transition points,
             predicted_profits(list): predicted profits of the sampling points
             profits(list): true profits of the sampling points
             sample_space(list): all sample points
             IMPORTANT: The lists are 2d lists. Assume a list(M,N) where M is the iterations and N is the sample points.
             To get the final results you would look for list(m,:).
             """
        """
               This methods approaches the transition point search from an exhaustive approach. The sample size is bounded
               by max_step_size_magnitude. Step size is determined by min_step_size_magnitude. Each sample point is used
               for profit calculations. And then these profits and sample points are used to extract transition points.
    
    
               :param alphas: parameters of the model
               :param k: the index of the parameter that is optimized
               :param const: constant of the model
               :param train_X: test set features
               :param train_Y: test set profits
               :param train_weights: test set weights
               :param capacities: capacity of the problem
    
               :return: item_sets(list): list of item sets, transition_points: a list of transition points,
               predicted_profits(list): predicted profits of the sampling points
               profits(list): true profits of the sampling points
               sample_space(list): all sample points
               IMPORTANT: The lists are 2d lists. Assume a list(M,N) where M is the iterations and N is the sample points.
               To get the final results you would look for list(m,:).
               """
        # print(param_ind)
        copy_time = time.time()
        temp_model = copy.deepcopy(model)
        copy_finish = time.time() - copy_time
        sampling_time = time.time()
        # if self.greedy_run_times is None:
        #     # first_layer_run_time_array = np.zeros(
        #     #     (temp_model.get_layer(1).out_features, temp_model.get_layer(1).in_features))
        #     # second_layer_run_time_array = np.zeros((temp_model.get_layer(2).out_features,
        #     #                                         temp_model.get_layer(2).in_features))
        #     # self.greedy_run_times = [first_layer_run_time_array, second_layer_run_time_array]
        #     array = [runtime_param_ind(runtime=0, layer_no=layer_no, param_ind=(i, j)) for
        #              layer_no in [1, 2] for i in
        #              range(temp_model.get_layer(layer_no).in_features) for j in range(temp_model.get_layer(layer_no).out_features)]
        #     heapq.heapify(array)
        #     self.greedy_run_times = array

        if bias:
            current_param = temp_model.get_layer(layer_no).bias[param.param_ind].detach().numpy()
        else:
            current_param = temp_model.get_layer(layer_no).weight[param.param_ind].detach().numpy()
        # print("dnc param: {}".format(current_param))
        start_ind = current_param - abs(current_param) * 10
        end_ind = current_param + abs(current_param) * 10

        M = math.ceil(math.log(10 ** (self.max_step_size_magnitude - self.min_step_size_magnitude),
                               self.step_size_divider)) + 1

        # Initialize first sample range parameters
        # print("before initialize current_param: {}, type: {}".format(current_param,current_param.shape))

        if current_param != 0:
            # print("min_step_size value should be", abs(current_param * (10 ** self.min_step_size_magnitude)))
            min_step_size = abs(current_param * (10 ** self.min_step_size_magnitude))
            # print("min_step_size value should is", min_step_size)
            sample_range = abs(current_param * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
            # print('in if, min_step_size: {}, sample_range: {}'.format(min_step_size, sample_range))
        else:
            sample_range = abs(1 * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
            min_step_size = 1 * (10 ** self.min_step_size_magnitude)
            # print('in else, min_step_size: {}, sample_range: {}'.format(min_step_size, sample_range))
        interval_mid_point = current_param
        interval_start = interval_mid_point - sample_range / 2
        interval_end = interval_mid_point + sample_range / 2
        # Initialize lists
        sample_spaces = [[] for i in range(M)]
        POVS = [[] for i in range(M)]
        TOVS = [[] for i in range(M)]
        transition_intervals = [[] for i in range(M)]
        intervals = [[] for i in range(M + 1)]

        intervals[0] = [Interval(TransitionPoint(interval_start, 0, 0), TransitionPoint(interval_end, 0, 0))]
        final_intervals = []
        iteration = 0
        greedy_sampling_time = time.time()
        for i in range(M):
            for interval_no, interval in enumerate(intervals[i]):

                iteration += 1
                start_index = interval.starting_point.x

                end_index = interval.ending_point.x
                sample_size = self.step_size_divider + 1
                # print("M: {}, Interval: {} Start: {}, End {}".format(M, i, start_index, end_index))
                step_size = abs(end_index - start_index) / self.step_size_divider

                if step_size > min_step_size:
                    # print('start index:', start_index, "step size:", step_size)
                    sample_space = [start_index + step_size * j for j in range(sample_size)]
                    # print("sample space", sample_space)
                    # if round(end_index, 3) > round(sample_space[-1], 3):
                    #     sample_space[-1] = end_index
                    sample_spaces[i].extend(sample_space)
                    # compute the profits of the sample points
                    if i == 0:
                        benchmark_TOVS, benchmark_POVS, __, pred_ys = get_optimization_objective_for_samples(
                            benchmark_x,
                            benchmark_y,
                            weights,
                            sample_space,
                            param.param_ind,
                            temp_model, layer_no, bias)
                        # initialize profit to the profit with the current value, may cache it to increase speed
                        profit = benchmark_TOVS[1]
                        original_param = TransitionPoint(x=sample_space[1], true_profit=profit,
                                                         predicted_profit=benchmark_POVS[1])
                    else:
                        # cache previous sample solutions
                        benchmark_TOVS, benchmark_POVS, __, pred_ys = get_optimization_objective_for_samples(
                            benchmark_x,
                            benchmark_y,
                            weights,
                            sample_space[1:-1],
                            param.param_ind,
                            temp_model, layer_no, bias)

                        benchmark_TOVS = np.hstack(
                            (interval.starting_point.true_profit, benchmark_TOVS, interval.ending_point.true_profit))

                        benchmark_POVS = np.hstack((interval.starting_point.predicted_profit, benchmark_POVS,
                                                    interval.ending_point.predicted_profit))

                    if max(benchmark_TOVS) > profit:
                        index = np.argmax(benchmark_TOVS)
                        transition_point = TransitionPoint(x=sample_space[index], true_profit=benchmark_TOVS[index],
                                                           predicted_profit=benchmark_POVS[index])
                        transition_points = [transition_point]
                        greedy_sampling_end_time = time.time() - greedy_sampling_time
                        # print("GOOD POINT greedy_sampling_end_time:", round(greedy_sampling_end_time, 6), "iteration: ", iteration)
                        sampling_time = time.time() - sampling_time
                        # print("sampling_end_time:", round(sampling_time, 6), "copy time:", copy_finish, "iteration: ",
                        #       iteration)
                        return transition_points, final_intervals, sample_spaces[-1], POVS[-1], TOVS[
                            -1], greedy_sampling_end_time

                    greedy_sampling_end_time = time.time() - greedy_sampling_time
                    if greedy_sampling_end_time > param.runtime * RUN_TIME_CONSTANT:
                        transition_points = [original_param]
                        # print("TIMEOUT greedy_sampling_end_time:", round(greedy_sampling_end_time, 6), "iteration: ", iteration)
                        sampling_time = time.time() - sampling_time
                        # print("sampling_end_time:", round(sampling_time, 6), "copy time:", (copy_finish),
                        # "iteration: ", iteration)
                        return transition_points, final_intervals, sample_spaces[-1], POVS[-1], TOVS[
                            -1], greedy_sampling_end_time

                    this_transition_intervals, dead_intervals = find_transition_intervals(sample_space, benchmark_POVS,
                                                                                          benchmark_TOVS)
                    transition_intervals[i].extend(this_transition_intervals)
                    intervals[i + 1] = transition_intervals[i]
                    POVS[i].extend(benchmark_POVS)
                    TOVS[i].extend(benchmark_TOVS)
                else:
                    final_intervals.append(interval)
                    greedy_sampling_end_time = time.time() - greedy_sampling_time

                    if greedy_sampling_end_time > param.runtime * RUN_TIME_CONSTANT:
                        transition_points = [original_param]
                        # print("TIMEOUT greedy_sampling_end_time:", round(greedy_sampling_end_time, 6), "iteration: ", iteration)
                        sampling_time = time.time() - sampling_time
                        # print("sampling_end_time:", round(sampling_time, 6), "copy time:", (copy_finish), "iteration: ",
                        #       iteration)
                        return transition_points, final_intervals, sample_spaces[-1], POVS[-1], TOVS[
                            -1], greedy_sampling_end_time
                    # decision_policies.append(predicted_opt_items)
        greedy_sampling_end_time = time.time() - greedy_sampling_time
        # print("greedy_sampling_end_time:", round(greedy_sampling_end_time, 6), "iteration: ", iteration)
        transition_points = [original_param]
        # # Debugging plots
        # print("NUmber of transition points:", len(transition_points))
        # debug_number_of_samples = 10**(self.max_step_size_magnitude - self.min_step_size_magnitude)
        # debug_step_size = (interval_end - interval_start) / debug_number_of_samples
        # debug_samples = [interval_start + x*debug_step_size for x in range(debug_number_of_samples)]
        # debug_benchmark_TOVS, debug_benchmark_POVS, __, debug_pred_ys = get_optimization_objective_for_samples(
        #                     benchmark_x,
        #                     benchmark_y,
        #                     weights,
        #                     debug_samples,
        #                     param_ind,
        #                     temp_model, layer_no)
        #
        # plt.scatter(debug_samples, debug_benchmark_TOVS)
        # plt.scatter(current_param, transition_points[0].true_profit, c='r')
        # for transition_point in transition_points:
        #     plt.scatter(transition_point.x, transition_point.true_profit)
        # plt.show()
        #     plt.subplot(M, 2, 2 * i + 1)
        #     plt.xlim([interval_start, interval_end])
        #     plt.scatter(sample_spaces[i], POVS[i])
        #     # for transition_point in transition_points:
        #     #     plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
        #     #                 transition_point.starting_point.predicted_profit, c='r')
        #     plt.title('POV')
        #     plt.subplot(M, 2, (i + 1) * 2)
        #     plt.xlim([interval_start, interval_end])
        #     plt.title('TOV')
        #     plt.scatter(sample_spaces[i], TOVS[i])
        #     for transition_point in transition_points:
        #         plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
        #                     transition_point.starting_point.true_profit, c='r')
        #     plt.subplot(3, 2, 3)
        #     plt.title('Derivatives')
        #     shifted_POVS = np.roll(benchmark_POVS, 1)
        #     der = (benchmark_POVS[1:99] - shifted_POVS[1:99]) / (sample_params[1] - sample_params[2])
        #     plt.scatter(sample_params[1:99], der)
        #     shifted_der = np.roll(der, 1)
        #     d2 = (der[2:98] - shifted_der[2:98]) / (sample_params[1] - sample_params[2])
        #
        #     plt.subplot(3, 2, 4)
        #     plt.title('Second Derivatives')
        #     plt.scatter(sample_params[2:98], d2)
        #     # for i in range(96):
        #     #     print("decision policy({}): {}, der2: {}".format(i, decision_policies[i + 2], np.round(d2[i], 3)))
        #
        #     plt.subplot(3, 2, 5)
        #     plt.title('Individual Item Predictions')
        #     for i in range(50):
        #         plt.plot(sample_params, pred_ys[i, :])
        #     plt.subplot(3, 2, 6)
        #     plt.title("MSE")
        #     mse = []
        #     for i in range(100):
        #         mse.append(np.mean((benchmark_y - pred_ys[:, i]) ** 2))
        #     plt.scatter(sample_params, mse)
        #
        #     plt.figure()
        #     for i in range(100):
        #         plt.title("pred_values vs real values")
        #         plt.subplot(10,10,i+1)
        #         plt.scatter(np.array([i for i in range(50)]), pred_ys[:,i], c='r')
        #         plt.scatter(np.array([i for i in range(50)]), benchmark_y, c='b')
        # plt.show()
        sampling_time = time.time() - sampling_time
        # print("step_size:", round(sampling_time, 6), "copy time:", (copy_finish), "iteration: ", iteration)
        return transition_points, final_intervals, sample_spaces[-1], POVS[-1], TOVS[-1], greedy_sampling_end_time

    def exhaustive_search(self, benchmark_x, benchmark_y, weights, param_ind, model, layer=None):
        """
              This methods approaches the transition point search from an exhaustive approach. The sample size is bounded
              by max_step_size_magnitude. Step size is determined by min_step_size_magnitude. Each sample point is used
              for profit calculations. And then these profits and sample points are used to extract transition points.


              :param alphas: parameters of the model
              :param k: the index of the parameter that is optimized
              :param const: constant of the model
              :param train_X: test set features
              :param train_Y: test set profits
              :param train_weights: test set weights
              :param capacities: capacity of the problem

              :return: item_sets(list): list of item sets, transition_points: a list of transition points,
              predicted_profits(list): predicted profits of the sampling points
              profits(list): true profits of the sampling points
              sample_space(list): all sample points
              IMPORTANT: The lists are 2d lists. Assume a list(M,N) where M is the iterations and N is the sample points.
              To get the final results you would look for list(m,:).
              """
        current_param = layer.weight[param_ind].detach().numpy()
        # print("exh param: {}".format(current_param))
        if current_param != 0:
            min_step_size = abs(current_param * (10 ** self.min_step_size_magnitude))
            sample_range = abs(current_param * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
        else:
            sample_range = abs(1 * SAMPLE_RANGE_MULTIPLIER * (10 ** self.max_step_size_magnitude))
            min_step_size = 1 * (10 ** self.min_step_size_magnitude)
        step_size = abs(current_param * (10 ** self.min_step_size_magnitude))
        sample_numbers = round(sample_range / step_size)
        interval_mid_point = current_param
        interval_start = interval_mid_point - sample_range / 2
        interval_end = interval_mid_point + sample_range / 2

        sample_params = np.array([x for x in np.linspace(interval_start, interval_end, sample_numbers, endpoint=True)])

        # sample_params = np.array([x for x in np.linspace(start_ind, end_ind, 100)])

        TOVS, POVS, __, pred_ys = get_optimization_objective_for_samples(benchmark_x, benchmark_y, weights,
                                                                         sample_params,
                                                                         param_ind,
                                                                         model, layer)

        transition_intervals = find_transition_intervals_old(alpha_samples=sample_params, predicted_profits=POVS,
                                                             profits=TOVS)

        transition_points = create_transition_points_from_intervals(transition_intervals,
                                                                    selection_method=self.transition_point_selection)
        # decision_policies.append(predicted_opt_items)
        # plt.figure()
        # plt.subplot(3, 2, 1)
        # plt.scatter(sample_params, POVS)
        # for interval in transition_intervals:
        #     plt.scatter((interval.starting_point.x + interval.ending_point.x) * 0.5,
        #                 interval.starting_point.predicted_profit, c='r')
        # plt.title('POV')
        # plt.subplot(3, 2, 2)
        # plt.title('TOV')
        # plt.scatter(sample_params, TOVS)
        # for interval in transition_intervals:
        #     plt.scatter((interval.starting_point.x + interval.ending_point.x) * 0.5,
        #                 interval.starting_point.true_profit, c='r')
        # plt.subplot(3, 2, 3)
        # plt.title('Derivatives')
        # shifted_POVS = np.roll(POVS, 1)
        # der = (POVS[1:99] - shifted_POVS[1:99]) / (sample_params[1] - sample_params[2])
        # plt.scatter(sample_params[1:99], der)
        # shifted_der = np.roll(der, 1)
        # d2 = (der[2:98] - shifted_der[2:98]) / (sample_params[1] - sample_params[2])
        #
        # plt.subplot(3, 2, 4)
        # plt.title('Second Derivatives')
        # plt.scatter(sample_params[2:98], d2)
        # # for i in range(96):
        # #     print("decision policy({}): {}, der2: {}".format(i, decision_policies[i + 2], np.round(d2[i], 3)))
        #
        # plt.subplot(3, 2, 5)
        # plt.title('Individual Item Predictions')
        # for i in range(50):
        #     plt.plot(sample_params, pred_ys[i, :])
        # plt.subplot(3, 2, 6)
        # plt.title("MSE")
        # mse = []
        # for i in range(100):
        #     mse.append(np.mean((benchmark_y - pred_ys[:, i]) ** 2))
        # plt.scatter(sample_params, mse)

        # plt.figure()
        # for i in range(100):
        #     plt.title("pred_values vs real values")
        #     plt.subplot(10,10,i+1)
        #     plt.scatter(np.array([i for i in range(50)]), pred_ys[:,i], c='r')
        #     plt.scatter(np.array([i for i in range(50)]), benchmark_y, c='b')
        # plt.show()
        return transition_points, transition_intervals, sample_params, POVS, TOVS

    def get_transition_points(self, model, layer, benchmark_x, benchmark_y, weights, param_ind, bias, profit):
        """
        This is a wrapper function, calls the related functions depending of the model.

        DIVIDE AND CONQUER: USES DIVIDE AND CONQUER search for finding transition points, returns all transition points.
        DIVIDE AND CONQUER MAX: USES DIVIDE AND CONQUER search for finding transition points, returns the transition point with the best profit.
        EXHAUSTIVE:USES EXHAUSTIVE search for finding transition points.
        EXHAUSTIVE MAX: USES EXHAUSTIVE search for finding transition points, returns the transition point with the best profit.

        :param alphas: parameters of the model
        :param k: the index of the parameter that is optimized
        :param const: constant of the model
        :param train_X: test set features
        :param train_Y: test set profits
        :param train_weights: test set weights
        :param capacities: capacity of the problem

        :return:
        """
        if self.sampling_method == DIVIDE_AND_CONQUER:
            return self.divide_and_conquer_search(benchmark_x=benchmark_x, benchmark_y=benchmark_y, weights=weights,
                                                  param_ind=param_ind,
                                                  model=model, layer_no=layer, bias=bias)

        if self.sampling_method == DIVIDE_AND_CONQUER_GREEDY:
            return self.divide_and_conquer_greedy_search(benchmark_x=benchmark_x, benchmark_y=benchmark_y,
                                                         weights=weights,
                                                         param=param_ind,
                                                         model=model, layer_no=layer, bias=bias, profit=profit)

        if self.sampling_method == DIVIDE_AND_CONQUER_GREEDY_MERGED:
            return self.divide_and_conquer_greedy_search_merged(benchmarks_x=benchmarks_x, benchmarks_y=benchmarks_y,
                                                         weights=weights,
                                                         param=param_ind,
                                                         model=model, layer_no=layer, bias=bias, profit=profit)

        if self.sampling_method == EXHAUSTIVE:
            return self.exhaustive_search(benchmark_x=benchmark_x, benchmark_y=benchmark_y, weights=weights,
                                          param_ind=param_ind,
                                          model=model, layer_no=layer)


def find_best_transition_point(transition_points, alpha):
    """
    find the transition point with the best profit.
    :param transition_points: a list of transition points
    :param alpha: parameters of the model
    :return: best_point: transition point with the best profit.
    """
    max_profit = alpha.true_profit
    best_point = alpha
    for transition_point in transition_points[len(transition_points) - 1]:
        if transition_point.true_profit > max_profit:
            best_point = transition_point

    return best_point


def find_transition_intervals_old(alpha_samples, predicted_profits, profits):
    """
    given alpha samples and related predicted profits, tries to find transition intervals. We do it by building linear
    function parameters for each interval, and comparing intervals.
    :param profits: profits of the sampled alphas
    :param alpha_samples: a set of alpha samples
    :param predicted_profits: predicted costs of the sampled alphas
    :return: transition_points(
    list): a list of transition points.
    """
    transition_intervals = []
    is_prev_point_transition = False
    i = 0
    while i < (len(alpha_samples) - 2):
        sample_point_1 = TransitionPoint((alpha_samples[i]), predicted_profits[i], profits[i])
        sample_point_2 = TransitionPoint((alpha_samples[i + 1]), predicted_profits[i + 1], profits[i + 1])
        sample_point_3 = TransitionPoint((alpha_samples[i + 2]), predicted_profits[i + 2], profits[i + 2])

        lin_func_1 = LinearFunction(sample_point_1, sample_point_2)
        lin_func_2 = LinearFunction(sample_point_2, sample_point_3)
        if not lin_func_1.is_same(lin_func_2):
            if not is_prev_point_transition:
                first_transition_interval = Interval(sample_point_1,
                                                     sample_point_2)
            second_transition_interval = Interval(sample_point_2,
                                                  sample_point_3)
            if not is_prev_point_transition:
                transition_intervals.extend([first_transition_interval, second_transition_interval])
            else:
                transition_intervals.extend([second_transition_interval])
            is_prev_point_transition = True
        else:
            is_prev_point_transition = False
        i = i + 1

    return transition_intervals


def find_transition_intervals(alpha_samples, predicted_profits, profits):
    """
    given alpha samples and related predicted profits, tries to find transition intervals. We do it by building linear
    function parameters for each interval, and comparing intervals.
    :param profits: profits of the sampled alphas
    :param alpha_samples: a set of alpha samples
    :param predicted_profits: predicted costs of the sampled alphas
    :return: transition_points(
    list): a list of transition points.
    """
    transition_intervals = []
    dead_intervals = []
    # is_prev_point_transition = True
    i = 0

    sample_point_1 = TransitionPoint((alpha_samples[i]), predicted_profits[i], profits[i])
    sample_point_2 = TransitionPoint((alpha_samples[i + 1]), predicted_profits[i + 1], profits[i + 1])
    sample_point_3 = TransitionPoint((alpha_samples[i + 2]), predicted_profits[i + 2], profits[i + 2])

    lin_func_1 = LinearFunction(sample_point_1, sample_point_2)
    lin_func_2 = LinearFunction(sample_point_2, sample_point_3)

    if lin_func_1.is_same(lin_func_2):
        dead_intervals.append(Interval(sample_point_1, sample_point_3))
    else:
        transition_intervals.append(Interval(sample_point_1, sample_point_2))
        transition_intervals.append(Interval(sample_point_2, sample_point_3))

    return transition_intervals, dead_intervals

    # while i < (len(alpha_samples) - 2):
    #     sample_point_1 = TransitionPoint((alpha_samples[i]), predicted_profits[i], profits[i])
    #     sample_point_2 = TransitionPoint((alpha_samples[i + 1]), predicted_profits[i + 1], profits[i + 1])
    #     sample_point_3 = TransitionPoint((alpha_samples[i + 2]), predicted_profits[i + 2], profits[i + 2])
    #
    #     lin_func_1 = LinearFunction(sample_point_1, sample_point_2)
    #     lin_func_2 = LinearFunction(sample_point_2, sample_point_3)
    #     if not lin_func_1.is_same(lin_func_2):
    #         if not is_prev_point_transition:
    #             first_transition_interval = Interval(sample_point_1,
    #                                                  sample_point_2)
    #         second_transition_interval = Interval(sample_point_2,
    #                                               sample_point_3)
    #         if not is_prev_point_transition:
    #             transition_intervals.extend([first_transition_interval, second_transition_interval])
    #         else:
    #             transition_intervals.extend([second_transition_interval])
    #         is_prev_point_transition = True
    #     else:
    #         is_prev_point_transition = False
    #     i = i + 1
    #
    # return transition_intervals
