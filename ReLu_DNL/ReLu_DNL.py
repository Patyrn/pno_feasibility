import copy
import heapq
import os
import statistics
import time

import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp

from functools import partial
from operator import attrgetter
from sklearn.utils import shuffle
from sklearn import preprocessing

from ReLu_DNL import Sampling_Methods
from ReLu_DNL.Sampling_Methods import DIVIDE_AND_CONQUER_GREEDY
# from dnl import Sampling_Methods
# from dnl.DnlNeuralLayer import get_dnl_exact_grad
# from dnl.EnergyDataUtil import get_energy_data
# from dnl.KnapsackSolver import get_opt_params_knapsack
# from dnl.PredictPlusOptimize import PredictPlusOptModel
# from dnl.Sampling_Methods import DIVIDE_AND_CONQUER_GREEDY
# from dnl.Solver import get_optimization_objective, get_optimal_average_objective_value
# from dnl.Utils import get_train_test_split


from Solver import get_optimization_objective, get_optimal_average_objective_value
from Utils import TransitionPoint, get_mini_batches


def get_benchmarks_from_input_tensor(x, benchmark_size):
    x = x.T.detach().numpy()
    batch_size = int(x.shape[1] / benchmark_size)
    benchmarks_X = [x[:, i * benchmark_size:(i + 1) * benchmark_size] for i in range(batch_size)]
    return benchmarks_X


def get_benchmarks_from_input_numpy(x, benchmark_size):
    x = x.reshape(1, -1)
    batch_size = int(x.shape[1] / benchmark_size)
    benchmarks_X = [x[:, i * benchmark_size:(i + 1) * benchmark_size] for i in range(batch_size)]
    return benchmarks_X


def get_and_clean_transition_points(benchmark_X, benchmark_Y, benchmark_weights, model, layer_no,
                                    param_ind, sampler, profit,
                                    bias=False):
    # print("Inside get and clean, starting get, thread{}",format(os.getpid()))

    dnc_points, dnc_intervals, __, dnc_POVS, dnc_TOVS, run_time = sampler.get_transition_points(
        model,
        layer_no,
        benchmark_X,
        benchmark_Y,
        benchmark_weights,
        param_ind,
        bias, profit)
    # print("number of transition points", len(dnc_points))
    # print("Inside get and clean, finished get, starting clean, thread{}",format(os.getpid()))
    best_transition_point_set_benchmark = clean_transition_points(dnc_points, profit)
    # print("Inside get and clean, finished clean, thread{}",format(os.getpid()))
    return best_transition_point_set_benchmark, run_time


def clean_transition_points(transition_points, profit):
    cleaner_transition_points = set()
    # print('inside clean thread{}'.format(os.getpid()))
    for transition_point in transition_points:
        # print("transition_point", transition_point.true_profit)
        # dont clean right now
        # if transition_point.true_profit > profit:

        # print("clean:", transition_point.x)
        cleaner_transition_points.add(transition_point.x)
        # print("clean done")
        # print('error at clean')
    # print('finished clean thread{}'.format(os.getpid()))
    return cleaner_transition_points


def find_the_best_transition_point_benchmarks(train_X, train_Y, model, layer_no, param_ind, transition_point_list,
                                              opt_params,
                                              train_weights, profit, bias, pool=None):
    # print('inside find_the_best_transition_point_benchmarks')
    if bias:
        original_param = TransitionPoint(x=copy.deepcopy(model.get_layer(layer_no).bias[param_ind].detach().numpy()),
                                         true_profit=profit)
    else:
        original_param = TransitionPoint(x=copy.deepcopy(model.get_layer(layer_no).weight[param_ind].detach().numpy()),
                                         true_profit=profit)
    best_average_profit = profit
    if not (len(transition_point_list) == 1):
        if pool is not None:
            map_func = partial(find_the_best_transition_point_benchmarks_worker, train_X=train_X,
                               train_Y=train_Y,
                               train_weights=train_weights,
                               model=model, layer_no=layer_no,
                               opt_params=opt_params,
                               param_ind=param_ind, bias=bias)
            results = pool.map(map_func, transition_point_list)
            # list_param = [original_param]
            # list_param.extend(results)
            results.extend([original_param])

            # print('x', [transition_point.x for transition_point in results], ' objective_value' ,
            # [transition_point.true_profit for transition_point in results])
            # print('im comparing transition points')
            best_transition_point = max(results, key=attrgetter('true_profit'))
            # if original_param.x != best_transition_point_candidate.x:
            #     best_transition_point = copy.deepcopy(best_transition_point_candidate)
            # else:
            #     best_transition_point = copy.deepcopy(original_param)


        else:
            best_transition_point = copy.deepcopy(original_param)
            for transition_point_x in transition_point_list:
                transition_point = find_the_best_transition_point_benchmarks_worker(transition_point_x, train_X=train_X,
                                                                                    train_Y=train_Y,
                                                                                    train_weights=train_weights,
                                                                                    model=model, layer_no=layer_no,
                                                                                    opt_params=opt_params,
                                                                                    param_ind=param_ind, bias=bias)

                if transition_point.true_profit > best_average_profit:
                    best_average_profit = transition_point.true_profit
                    best_transition_point = copy.deepcopy(transition_point)
    else:
        best_transition_point = original_param
    return best_transition_point


def find_the_best_transition_point_benchmarks_worker(transition_point_x, train_X, train_Y, train_weights, model,
                                                     layer_no,
                                                     opt_params, param_ind, bias):
    temp_model = copy.deepcopy(model)
    layer = temp_model.get_layer(layer_no)

    # print("Before weight: {}".format(temp_model.get_layer(layer_no).weight.data[param_ind]))
    if bias:
        with torch.no_grad():
            layer.bias.data[param_ind] = transition_point_x
    else:
        with torch.no_grad():
            layer.weight.data[param_ind] = transition_point_x
    # print("After weight: {}".format(temp_model.get_layer(layer_no).weight.data[param_ind]))
    pred_Ys = []
    for x in train_X:
        pred_Ys.append(temp_model.forward(x).detach().numpy().flatten())
    optimization_objective, __ = get_optimization_objective(pred_Y=pred_Ys, Y=train_Y,
                                                            weights=train_weights, opt_params=opt_params,
                                                            )
    average_profit = np.median(optimization_objective)
    L2_loss = model.L2_lambda * (model.get_L2_loss(layer_no, param_ind, bias) + transition_point_x ** 2)
    # print("L2 loss: {}, regret with L2 Loss: {}".format(L2_loss, average_profit - L2_loss))
    # print('k: ' + str(k) + ' transition_point: ' + str(transition_point_x) + ' profit: ' + str(average_profit))

    return TransitionPoint(transition_point_x, true_profit=average_profit - L2_loss)


def find_the_best_transition_point_benchmarks_merged(benchmark_X, benchmark_Y, benchmark_weights, model, layer_no,
                                                     param_ind, sampler, profit,
                                                     bias=False, pool=None):
    # print("Inside get and clean, starting get, thread{}",format(os.getpid()))

    dnc_points, dnc_intervals, __, dnc_POVS, dnc_TOVS, run_time = sampler.get_transition_points(
        model,
        layer_no,
        benchmark_X,
        benchmark_Y,
        benchmark_weights,
        param_ind,
        bias, profit, pool)
    best_transition_point_set_benchmark = dnc_points[0]
    return best_transition_point_set_benchmark, run_time


class my_abs(nn.Module):
    def forward(self, x):
        # unfortunately we don't have automatic broadcasting yet
        return torch.abs(x)


class relu_ppo(nn.Module):
    def __init__(self, batch_size=16, dnl_batch_size=16, layer_params=None, benchmark_size=48, max_epoch=10,
                 learning_rate=1e-2,
                 dnl_epoch=10,
                 opt_params=5,
                 dnl_learning_rate=0.1, leaky_slope=0.2, sampling_method=Sampling_Methods.DIVIDE_AND_CONQUER,
                 max_step_size_magnitude=1,
                 min_step_size_magnitude=-1, dropout_percentage=10,
                 transition_point_selection=Sampling_Methods.MID_TRANSITION_POINT_SELECTION,
                 verbose=False, is_Val=True, run_time_limit=10000,
                 is_parallel=False, pool=None, is_update_bias=True, L2_lambda=0.001, path=None):
        """
        :param layer_params: neural net layer configuration
        :param benchmark_size: size of the benchmark
        :param max_epoch: Max epoch numbers
        :param learning_rate: learning_rate for nn regression
        :param opt_params: optimization parameters a.k.a constraints
        :param dnl_learning_rate: learning rate for dnl
        :param pool: pool if parallel computing
        :param params_per_epoch_divider: number of param updated -> max_params / divider
        """
        super().__init__()
        self.dnl_batch_size = dnl_batch_size
        self.batch_size = batch_size
        self.epoch_number = max_epoch
        self.learning_rate = learning_rate
        self.benchmark_size = benchmark_size
        self.opt_params = opt_params
        self.dnl_learning_rate = dnl_learning_rate
        self.scaler = None
        self.leaky_slope = leaky_slope

        if layer_params is None:
            self.layer_params = [1]
        else:
            self.layer_params = layer_params
        self.init_layers(self.layer_params)
        self.pool = pool
        self.is_parallel = is_parallel
        self.dnl_epoch = dnl_epoch
        self.sampling_method = sampling_method
        self.max_step_size_magnitude = max_step_size_magnitude
        self.min_step_size_magnitude = min_step_size_magnitude
        self.transition_point_selection = transition_point_selection
        self.L2_lambda = L2_lambda
        self.dropout_percentage = dropout_percentage
        self.number_of_parameters_to_update = 0
        if path is not None:
            self.path = path
        else:
            self.path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

        self.test_run_time = 0
        self.verbose = verbose
        self.is_update_bias = is_update_bias
        self.is_val = is_Val
        self.sampling_time = 0
        self.compare_time = 0

        self.run_time_limit = run_time_limit
        self.training_obj_value = []
        self.test_regrets = []
        self.test_objs = []
        self.test_MSEs = []
        self.val_regrets = []
        self.val_objs = []
        self.epochs = []
        self.sub_epochs = []
        self.run_time = []
        self.test_pred_y = []
        self.test_biases = []
        self.test_vars = []

        self.dnc_sampler = Sampling_Methods.Sampler(sampling_method=sampling_method,
                                                    max_step_size_magnitude=max_step_size_magnitude,
                                                    min_step_size_magnitude=min_step_size_magnitude,
                                                    transition_point_selection=transition_point_selection, model=self)
        self.profiler_sampler = Sampling_Methods.Sampler(sampling_method=DIVIDE_AND_CONQUER_GREEDY,
                                                    max_step_size_magnitude=max_step_size_magnitude,
                                                    min_step_size_magnitude=min_step_size_magnitude,
                                                    transition_point_selection=transition_point_selection, model=self)

    def init_layers(self, layer_params):
        self.layers = nn.ModuleList()
        for i, layer_param in enumerate(layer_params[:-1]):
            self.layers.append(nn.Linear(in_features=layer_params[i], out_features=layer_params[i + 1]))

        # self.relu = nn.LeakyReLU(negative_slope=self.leaky_slope)
        self.abs = my_abs()
        self.relu = nn.LeakyReLU(negative_slope=self.leaky_slope)

    def get_layer(self, layer_no):
        return self.layers[layer_no]

    def forward(self, x):
        if self.scaler is not None:
            x = torch.from_numpy(self.scaler.transform(x)).float()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = x.float()
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            # x = self.abs(layer(x))
        x = self.layers[-1](x)

        # linear_out = self.fc1(x)  # shape is (batches, features)
        # linear_shape = linear_out.shape
        # first_slice = linear_out[:, 0:int(linear_shape[1]/2)]
        # second_slice = linear_out[:, int(linear_shape[1]/2):]
        # tuple_of_activated_parts = (
        #     self.relu(first_slice),
        #     self.abs(second_slice)
        # )
        # out = torch.cat(tuple_of_activated_parts, dim=1)
        # x = self.fc2(out)

        return x

    def fit_nn(self, x, y, max_epochs=None):
        if max_epochs is None:
            max_epochs = 10

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        if self.scaler == None:
            self.scaler = preprocessing.StandardScaler().fit(x)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        for t in range(max_epochs):
            permutation = torch.randperm(x.size()[0])
            for i in range(0, x.size()[0], self.batch_size):

                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = x[indices], y[indices]

                y_pred = self.forward(batch_x)
                loss = torch.sqrt(criterion(batch_y, y_pred))

                # Use autograd to compute the backward pass.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % (64 * 200) == 0:
                    print('e:{} i:{}/{} loss:{}'.format(t, i, x.size()[0], loss.item()))
                # # Update weights using gradient descent
                # with torch.no_grad():
                #     self.w1 -= self.learning_rate * self.w1.grad
                #     self.w2 -= self.learning_rate * self.w2.grad
                #
                #     # Manually zero the gradients after updating weights
                #     self.w1.grad.zero_()
                #     self.w2.grad.zero_()
                #
            # if t % 100 == 99:
            #     # print('y',y,'y_pred',y_pred)
            #     print(t, loss.item())
        torch.save(self, os.path.join(self.path, "regression.pth"))

    def get_L2_loss(self, custom_layer_no=None, custom_param_ind=None, bias=False):
        weight_sum = 0
        for layer_no in range(len(self.layers)):
            for param_ind in [(i, j) for i in range(self.get_layer(layer_no).weight.size()[0]) for j in
                              range(self.get_layer(layer_no).weight.size()[1])]:
                if not bias and (
                        layer_no == custom_layer_no and custom_param_ind is not None and custom_param_ind == param_ind):
                    weight_sum += self.get_layer(layer_no).weight.data[param_ind].detach().numpy().astype(float) ** 2

            for param_ind in range(self.get_layer(layer_no).bias.size()[0]):
                if bias and (
                        layer_no == custom_layer_no and custom_param_ind is not None and custom_param_ind == param_ind):
                    weight_sum += self.get_layer(layer_no).bias.data[param_ind].detach().numpy().astype(float) ** 2
        return weight_sum

    def fit_dnl(self, train_X, train_Y, train_weights, val_X, val_Y, val_weights, benchmark_size, print_test=False,
                test_X=None, test_Y=None,
                test_weights=None, core_number=8, sampling_method=None, max_step_size_magnitude=None,
                min_step_size_magnitude=None,
                transition_point_selection=None, test_X_MSE=None, test_Y_MSE=None, run_time_batch_size=10):

        if sampling_method is None:
            sampling_method = self.sampling_method
        if max_step_size_magnitude is None:
            max_step_size_magnitude = self.max_step_size_magnitude
        if min_step_size_magnitude is None:
            min_step_size_magnitude = self.min_step_size_magnitude
        if transition_point_selection is None:
            transition_point_selection = self.transition_point_selection

        # dnc_sampler = Sampling_Methods.Sampler(sampling_method=sampling_method,
        #                                        max_step_size_magnitude=max_step_size_magnitude,
        #                                        min_step_size_magnitude=min_step_size_magnitude,
        #                                        transition_point_selection=transition_point_selection, model = self)

        best_val = float("inf")

        # pred_Y = self.forward(train_X)
        # profit = np.median(get_optimization_objective(pred_Y=pred_Y, Y=train_Y, weights=train_weights,
        #                                               opt_params=self.opt_params))

        # test_regret = np.median(self.get_regret(test_X, test_Y, test_weights))
        # val_regret = np.median(self.get_regret(val_X, val_Y, val_weights))
        # self.test_regrets.append(test_regret)

        # self.val_regrets.append(val_regret)

        start_time = time.time()

        EPOCH = 0

        benchmark_numbers = int(len(train_Y) / benchmark_size)
        number_of_batches = int(benchmark_numbers / self.batch_size)

        if self.is_parallel:
            mypool = mp.Pool(processes=min(8, core_number))
        else:
            mypool = None
        profit = self.get_objective_value(train_X, train_Y, train_weights, self.opt_params, pool=mypool)
        prev_profit = profit

        if self.is_val:
            val_regret, _, val_obj, _ = self.get_regret(val_X, val_Y, val_weights, pool=mypool)

        else:
            val_regret = -1
            val_obj = -1

        if print_test:
            test_regret, _, test_obj, _ = self.get_regret(test_X, test_Y, test_weights, pool=mypool)
        else:
            test_regret = -1
            test_obj = -1
            #
        if test_X_MSE is not None:
            test_pred_y = self.forward(
                torch.from_numpy(test_X_MSE)).detach().numpy()
            if test_Y_MSE is not None:
                test_MSE = self.get_MSE(test_X_MSE, test_Y_MSE)
            else:
                test_MSE = -1
        else:
            test_pred_y = 0
            test_MSE = 0

        self.test_regrets.append(np.median(test_regret))
        self.test_objs.append(test_obj)

        self.test_pred_y.append(test_pred_y)
        self.test_MSEs.append(test_MSE)

        self.val_regrets.append(np.median(val_regret))
        self.val_objs.append(val_obj)

        self.training_obj_value.append(profit)
        self.run_time.append(0)
        self.epochs.append(0)
        self.sub_epochs.append(0)

        is_break = False
        print("original objective value: " + str(profit))
        print("------------------------")

        if self.dnl_batch_size == -1:
            dnl_batch_size = len(train_Y)
        else:
            dnl_batch_size = self.dnl_batch_size

        mini_batches_X, mini_batches_Y, mini_batches_weights = get_mini_batches(X=train_X, Y=train_Y,
                                                                                weights=train_weights,
                                                                                size=dnl_batch_size)

        print("BEFORE:", "objective value:", profit, 'val regret',
              self.val_regrets[-1], 'test regret', self.test_regrets[-1], flush=True)
        self.initiate_greedy_runtimes(self.profiler_sampler, train_X[0:run_time_batch_size], train_Y[0:run_time_batch_size],
                                      train_weights[0:run_time_batch_size], profit, mypool)

        sub_epoch = 0
        if test_X_MSE is not None:
            test_pred_y = self.forward(
                torch.from_numpy(test_X_MSE)).detach().numpy()
            self.test_pred_y.append(test_pred_y)
        while (EPOCH < self.dnl_epoch) and self.run_time[-1] < self.run_time_limit:
            mini_batches_X, mini_batches_Y, mini_batches_weights = shuffle(mini_batches_X, mini_batches_Y,
                                                                           mini_batches_weights)
            # UNCOMMENT BELOW FOR RANDOM SHUFFLE BATCHES
            # permutations = torch.randperm(number_of_benchmarks).tolist()
            # bELOW IS FOR ORDERED BATCHES
            permutations = [x for x in range(number_of_batches)]
            print("-------------------------")
            print("EPOCH: {}".format(EPOCH))
            for mini_batch_X, mini_batch_Y, mini_batch_weights in zip(mini_batches_X, mini_batches_Y,
                                                                      mini_batches_weights):
                train_X = mini_batch_X
                train_Y = mini_batch_Y
                train_weights = mini_batch_weights

                # pred_Y = self.forward(train_X)
                # profit = np.median(get_optimization_objective(pred_Y=pred_Y, Y=train_Y, weights=train_weights,
                #                                               opt_params=self.opt_params))
                print("-----------------------")

                layer_order = [2, 1]
                bias = True
                # if dnc_sampler.greedy_run_times is not None and len(dnc_sampler.greedy_run_times) > 10:
                # print(len(dnc_sampler.greedy_run_times))
                # number_of_parameters_to_update = int(len(dnc_sampler.greedy_run_times) / 5)
                number_of_parameters_to_update = int(
                    len(self.dnc_sampler.greedy_run_times) * (100-self.dropout_percentage) / 100)
                self.number_of_parameters_to_update = number_of_parameters_to_update
                print("Sub Epoch: {} Number of Updates: {}/{}".format(sub_epoch,self.number_of_parameters_to_update,len(self.dnc_sampler.greedy_run_times)))
                # number_of_parameters_to_update = 10
                param_list = [heapq.heappop(self.dnc_sampler.greedy_run_times) for i in
                              range(number_of_parameters_to_update)]
                for param in param_list:
                    param.increase_count()
                    # print("layer: {}, ind: {}, count: {}, run_time: {}".format(param.layer_no, param.param_ind, param.update_count, param.runtime))
                self.update_weights_heap(train_X, train_Y, train_weights, self.dnc_sampler, mypool, bias, profit,
                                         param_list)
                # else:
                #     for layer_no in layer_order:
                #         # print("layer: {}".format(layer_no))
                #         best_transition_points_set = set()
                #         # weights
                #         self.update_weights(train_X, train_Y, train_weights, layer_no, dnc_sampler, mypool, bias, profit)
                #
                #         # test_regret = self.get_regret(test_X, test_Y, test_weights, self.opt_params, pool=mypool)
                #         # manual_profit = self.get_objective_value(train_X, train_Y, train_weights, self.opt_params,
                #         #                                          pool=mypool)
                #
                #         if self.is_update_bias:
                #             bias = True
                #             self.update_bias(train_X, train_Y, train_weights, layer_no, dnc_sampler, mypool, bias, profit)

                test_run_time = time.time()
                if self.is_val:
                    # print('val')
                    val_regret, _, val_obj, _ = self.get_regret(val_X, val_Y, val_weights, pool=mypool)
                    val_regret = np.median(val_regret)
                    self.val_regrets.append(val_regret)
                    self.val_objs.append(np.median(val_obj))
                    if val_regret < best_val:
                        best_model_state = copy.deepcopy(self)

                if print_test:
                    # print('test')
                    test_regret, _, test_obj, _ = self.get_regret(test_X, test_Y, test_weights, pool=mypool)
                    self.test_regrets.append(np.median(test_regret))
                    self.test_objs.append(np.median(test_obj))
                    # train_regret = np.median(self.get_regret(train_X, train_Y, train_weights, pool=mypool))
                    # self.training_obj_value.append(train_regret)

                self.test_run_time = self.test_run_time + time.time() - test_run_time
                sub_epoch += 1
                if test_X_MSE is not None:
                    test_pred_y = self.forward(
                        torch.from_numpy(test_X_MSE)).detach().numpy()
                    self.test_pred_y.append(test_pred_y)
                    if test_Y_MSE is not None:
                        test_MSE = self.get_MSE(test_X_MSE, test_Y_MSE)
                        self.test_MSEs.append(test_MSE)

                self.sub_epochs.append(sub_epoch)
                self.epochs.append(EPOCH)
                self.run_time.append((time.time() - start_time - self.test_run_time))
                print("EPOCH:", EPOCH, "sub epoch:", sub_epoch, "objective value:", profit, 'val regret',
                      self.val_regrets[-1], 'test regret', self.test_regrets[-1], "run time:", self.run_time[-1],
                      flush=True)
                print("Sampling Time: {}, Compare Time: {}, Test Time{}".format(self.sampling_time, self.compare_time,
                                                                                self.test_run_time))
                print("**************************************************************************************")
                if self.run_time[-1] > self.run_time_limit:
                    is_break = True
                    break
                if is_break:
                    break
            EPOCH += 1
        print("EPOCH:", EPOCH, "objective value:", profit, 'val regret', self.val_regrets[-1], 'test regret',
              self.test_regrets[-1])
        print("Sampling Time: {}, Compare Time: {}".format(self.sampling_time, self.compare_time))
        print('Training finished ')
        print("-----------------------")
        self.number_of_epochs = EPOCH
        torch.save(best_model_state, os.path.join(self.path, "dnl22.pth"))
        if self.is_parallel:
            mypool.close()

    def initiate_greedy_runtimes(self, sampler, batch_X, batch_Y, batch_weights, profit, mypool):
        param_list = [heapq.heappop(sampler.greedy_run_times) for i in range(len(sampler.greedy_run_times))]
        for param in param_list:
            layer_no = param.layer_no
            # print("entering get and clean")
            # Weights
            best_transition_points_set, average_run_time = self.get_and_clean_profiler_wrap(layer_no=layer_no,
                                                                                            param_ind=param,
                                                                                            profit=profit,
                                                                                            train_X=batch_X,
                                                                                            train_Y=batch_Y,
                                                                                            train_weights=batch_weights,
                                                                                            bias=param.bias,
                                                                                            mypool=mypool)
            sampler.update_greedy_runtime(average_run_time * 5, param)
        self.dnc_sampler.greedy_run_times = copy.deepcopy(self.profiler_sampler.greedy_run_times)
    def get_MSE(self, x, y):
        y = torch.from_numpy(y).float()
        y_pred = self.forward(x)
        loss = (y - y_pred).detach().numpy() ** 2
        return np.median(loss)
        # if t % 100 == 99:
        #     # print('y',y,'y_pred',y_pred)
        #     print(t, loss.item())

    def get_regret(self, X, Y, weights, opt_params=None, pool=None):
        if opt_params is None:
            opt_params = self.opt_params
        pred_Ys = []
        for x in X:
            pred_Ys.append(self.forward(x).detach().numpy().flatten())

        if pool is None:
            objective_values_predicted_items, predicted_sols = get_optimization_objective(Y=Y, pred_Y=pred_Ys,
                                                                                          weights=weights,
                                                                                          opt_params=opt_params,
                                                                                          )
            optimal_objective_values, sols = get_optimal_average_objective_value(Y=Y, weights=weights,
                                                                                 opt_params=self.opt_params,
                                                                                 )

            regret = np.median(optimal_objective_values - objective_values_predicted_items)
        else:
            map_func = partial(get_regret_worker, opt_params=self.opt_params)
            iter = zip(Y, pred_Ys, weights)
            objective_values = pool.starmap(map_func, iter)
            # objective_values_predicted_items, optimal_objective_values = zip(*objective_values)
            # optimal_objective_values = np.concatenate(optimal_objective_values)
            # objective_values_predicted_items = np.concatenate(objective_values_predicted_items)
            # # regret = np.median(
            # #     optimal_objective_values - objective_values_predicted_items)
            #
            # regret = optimal_objective_values - objective_values_predicted_items

            objective_values_predicted_items, predicted_solutions, optimal_objective_values, solutions = zip(
                *objective_values)
            # predicted_solutions = [solution[0] for solution in predicted_solutions]
            # solutions = [solution[0] for solution in solutions]

            optimal_objective_values = np.concatenate(optimal_objective_values)
            objective_values_predicted_items = np.concatenate(objective_values_predicted_items)
            regret = optimal_objective_values - objective_values_predicted_items

        return regret, optimal_objective_values, list(predicted_solutions), list(solutions)
        # return regret, optimal_objective_values

    def get_objective_value(self, X, Y, weights, opt_params, pool=None):
        pred_Ys = []
        for x in X:
            pred_Ys.append(self.forward(x).detach().numpy().flatten())

        if pool is None:
            objective_values_predicted_items = get_optimization_objective(Y=Y, pred_Y=pred_Ys,
                                                                          weights=weights,
                                                                          opt_params=opt_params,
                                                                          )
            optimal_average_objective_value = get_optimal_average_objective_value(Y=Y, weights=weights,
                                                                                  opt_params=self.opt_params,
                                                                                  )
            regret = np.median(optimal_average_objective_value - objective_values_predicted_items)
        else:
            map_func = partial(get_regret_worker, opt_params=self.opt_params)
            iter = zip(Y, pred_Ys, weights)
            objective_values = pool.starmap(map_func, iter)
            objective_values_predicted_items, _, optimal_objective_values, _ = zip(*objective_values)
            regret = np.median(
                np.concatenate(optimal_objective_values) - np.concatenate(objective_values_predicted_items))
            objective_values_predicted_items = np.concatenate(objective_values_predicted_items)
        return np.median(objective_values_predicted_items)

    def get_and_clean_profiler_wrap(self, layer_no,
                                    param_ind,
                                    profit, train_X, train_Y,
                                    train_weights, bias, mypool):

        map_func = partial(get_and_clean_transition_points, model=self,
                           layer_no=layer_no, sampler=self.profiler_sampler,
                           param_ind=param_ind,
                           profit=profit, bias=bias)

        iter = [[benchmark_X, benchmark_Y, benchmark_weights] for
                benchmark_X, benchmark_Y, benchmark_weights in
                zip(train_X, train_Y, train_weights)]
        # print(layer_no)
        best_transition_points_set, run_times = zip(*mypool.starmap(map_func, iter))
        best_transition_points_set = set().union(*best_transition_points_set)
        return best_transition_points_set, statistics.median(run_times)

    def update_weights_heap(self, train_X, train_Y, train_weights, dnc_sampler, mypool, bias, profit, param_list):
        for param in param_list:
            if self.is_parallel:
                layer_no = param.layer_no
                # print("entering get and clean")
                # Weights
                start_time = time.time()
                # best_transition_points_set, average_run_time = self.get_and_clean_profiler_wrap(layer_no=layer_no, sampler=dnc_sampler,
                #                                                               param_ind=param,
                #                                                               profit=profit, train_X=train_X,
                #                                                               train_Y=train_Y,
                #                                                               train_weights=train_weights, bias=param.bias,
                #                                                               mypool=mypool)
                # dnc_sampler.update_greedy_runtime(average_run_time, param)
                #
                # self.sampling_time += time.time() - start_time
                # start_time = time.time()
                # # print("finished get and clean, entering find the best")
                # benchmark_best_transition_point = find_the_best_transition_point_benchmarks(train_X,
                #                                                                             train_Y,
                #                                                                             model=self,
                #                                                                             layer_no=layer_no,
                #                                                                             train_weights=train_weights,
                #                                                                             opt_params=self.opt_params,
                #                                                                             transition_point_list=list(
                #                                                                                 best_transition_points_set),
                #                                                                             profit=profit,
                #                                                                             param_ind=param.param_ind,
                #                                                                             pool=mypool, bias=param.bias)

                benchmark_best_transition_point,average_run_time = find_the_best_transition_point_benchmarks_merged(train_X,
                                                                                                   train_Y,
                                                                                                   train_weights,
                                                                                                   model=self,
                                                                                                   layer_no=layer_no,
                                                                                                   sampler=dnc_sampler,
                                                                                                   param_ind=param,
                                                                                                   profit=profit,
                                                                                                   bias=param.bias,
                                                                                                   pool=mypool)

                dnc_sampler.update_greedy_runtime(average_run_time, param)
                self.compare_time += time.time() - start_time

                prev_profit = profit
                profit = benchmark_best_transition_point.true_profit

                # print('best transition point profit: ', benchmark_best_transition_point.true_profit)
                # print("before:",self.weight.data[param_ind], "transition point:", benchmark_best_transition_point.x)
                # print("gradient: {}".format(gradient))
                if param.bias:
                    with torch.no_grad():
                        gradient = benchmark_best_transition_point.x - self.get_layer(layer_no).bias.data[
                            param.param_ind].detach().numpy().astype(float)
                        if gradient != 0:
                            self.get_layer(layer_no).bias.data[param.param_ind] = self.get_layer(layer_no).bias.data[
                                                                                      param.param_ind] + self.dnl_learning_rate * gradient

                else:
                    with torch.no_grad():
                        gradient = benchmark_best_transition_point.x - self.get_layer(layer_no).weight.data[
                            param.param_ind].detach().numpy().astype(float)
                        if gradient != 0:
                            self.get_layer(layer_no).weight.data[param.param_ind] = \
                            self.get_layer(layer_no).weight.data[
                                param.param_ind] + self.dnl_learning_rate * gradient

                    # print("after:",self.weight.data[param_ind])
                    # print("gradient: {}".format(gradient))

    def update_weights(self, train_X, train_Y, train_weights, layer_no, dnc_sampler, mypool, bias, profit):

        # greedy update best parameters

        for param_ind in [(i, j) for i in range(self.get_layer(layer_no).weight.size()[0]) for j in
                          range(self.get_layer(layer_no).weight.size()[1])]:
            # print("param: {}".format(param_ind))

            if self.is_parallel:
                # print("entering get and clean")
                # Weights
                start_time = time.time()
                best_transition_points_set = self.get_and_clean_profiler_wrap(layer_no=layer_no, sampler=dnc_sampler,
                                                                              param_ind=param_ind,
                                                                              profit=profit, train_X=train_X,
                                                                              train_Y=train_Y,
                                                                              train_weights=train_weights, bias=bias,
                                                                              mypool=mypool)

                self.sampling_time += time.time() - start_time
                start_time = time.time()
                # print("finished get and clean, entering find the best")
                benchmark_best_transition_point = find_the_best_transition_point_benchmarks(train_X,
                                                                                            train_Y,
                                                                                            model=self,
                                                                                            layer_no=layer_no,
                                                                                            train_weights=train_weights,
                                                                                            opt_params=self.opt_params,
                                                                                            transition_point_list=list(
                                                                                                best_transition_points_set),
                                                                                            profit=profit,
                                                                                            param_ind=param_ind,
                                                                                            pool=mypool, bias=bias)
                self.compare_time += time.time() - start_time
                # Biases

                # print("finished find the best")
            else:
                start_time = time.time()
                for benchmark_X, benchmark_Y, benchmark_weights in zip(train_X, train_Y, train_weights):
                    # Check for efficiency copying. Find a way to impelemnt divide and conquer without changing original layers

                    dnc_start_time = time.time()

                    best_transition_points_set = get_and_clean_transition_points(benchmark_X=benchmark_X,
                                                                                 benchmark_Y=benchmark_Y,
                                                                                 benchmark_weights=benchmark_weights,
                                                                                 model=self,
                                                                                 layer_no=layer_no, sampler=dnc_sampler,
                                                                                 param_ind=param_ind,
                                                                                 profit=profit, train_X=train_X,
                                                                                 train_Y=train_Y,
                                                                                 train_weights=train_weights, bias=bias)

                    benchmark_best_transition_point = find_the_best_transition_point_benchmarks(train_X,
                                                                                                train_Y,
                                                                                                model=self,
                                                                                                layer_no=layer_no,
                                                                                                train_weights=train_weights,
                                                                                                opt_params=self.opt_params,
                                                                                                transition_point_list=list(
                                                                                                    best_transition_points_set),
                                                                                                profit=profit,
                                                                                                param_ind=param_ind,
                                                                                                pool=mypool, bias=bias)

                    self.compare_time += time.time() - start_time

            # If not doing coordinate descent add below to return layer to original
            # with torch.no_grad():
            #     current_temp_layer.weight[param_ind] = current_original_layer.weight[param_ind]
            # prev_manual_profit = self.get_objective_value(train_X, train_Y, train_weights, self.opt_params,
            #                                               pool=mypool)

            # print('manual before, : manual: {}'.format(prev_manual_profit))
            prev_profit = profit
            profit = benchmark_best_transition_point.true_profit
            gradient = benchmark_best_transition_point.x - self.get_layer(layer_no).weight.data[
                param_ind].detach().numpy().astype(float)
            if gradient != 0:
                # print('best transition point profit: ', benchmark_best_transition_point.true_profit)
                # print("before:",self.weight.data[param_ind], "transition point:", benchmark_best_transition_point.x)
                # print("gradient: {}".format(gradient))
                with torch.no_grad():
                    self.get_layer(layer_no).weight.data[param_ind] = self.get_layer(layer_no).weight.data[
                                                                          param_ind] + self.dnl_learning_rate * gradient
                # print("after:",self.weight.data[param_ind])
                # print("gradient: {}".format(gradient))

    def update_bias(self, train_X, train_Y, train_weights, layer_no, dnc_sampler, mypool, bias, profit):
        for param_ind in range(self.get_layer(layer_no).bias.size()[0]):
            # print("param: {}".format(param_ind))
            if self.is_parallel:
                # print("entering get and clean")
                # Weights
                map_func = partial(get_and_clean_transition_points, model=self,
                                   layer_no=layer_no, sampler=dnc_sampler,
                                   param_ind=param_ind,
                                   profit=profit, train_X=train_X, train_Y=train_Y,
                                   train_weights=train_weights, bias=bias)
                iter = [[benchmark_X, benchmark_Y, benchmark_weights] for
                        benchmark_X, benchmark_Y, benchmark_weights in
                        zip(train_X, train_Y, train_weights)]
                best_transition_points_set = mypool.starmap(map_func, iter)
                best_transition_points_set = set().union(*best_transition_points_set)

                # print("finished get and clean, entering find the best")
                original_param = copy.deepcopy(
                    self.get_layer(layer_no).bias[param_ind].detach().numpy().astype(float))
                benchmark_best_transition_point = find_the_best_transition_point_benchmarks(train_X,
                                                                                            train_Y,
                                                                                            model=self,
                                                                                            layer_no=layer_no,
                                                                                            train_weights=train_weights,
                                                                                            opt_params=self.opt_params,
                                                                                            transition_point_list=list(
                                                                                                best_transition_points_set),
                                                                                            profit=profit,
                                                                                            param_ind=param_ind,
                                                                                            pool=mypool,

                                                                                            bias=bias)

                # Biases

                # print("finished find the best")
            else:
                for benchmark_X, benchmark_Y, benchmark_weights in zip(train_X, train_Y, train_weights):
                    # Check for efficiency copying. Find a way to impelemnt divide and conquer without changing original layers

                    dnc_start_time = time.time()
                    dnc_points, dnc_intervals, __, dnc_POVS, dnc_TOVS = dnc_sampler.get_transition_points(
                        self,
                        layer_no,
                        benchmark_X,
                        benchmark_Y,
                        benchmark_weights,
                        param_ind)
                    dnc_end_time = time.time()
                    # print("DNC finished in {}s".format(dnc_end_time - dnc_start_time))

                    best_transition_point_set_benchmark = clean_transition_points(dnc_points, profit)

                    best_transition_points_set = best_transition_points_set.union(
                        best_transition_point_set_benchmark)
                original_param = copy.deepcopy(self.weight[param_ind])
                benchmark_best_transition_point = find_the_best_transition_point_benchmarks(train_X,
                                                                                            train_Y,
                                                                                            model=self,
                                                                                            layer_no=layer_no,
                                                                                            train_weights=train_weights,
                                                                                            opt_params=self.opt_params,
                                                                                            transition_point_list=list(
                                                                                                best_transition_points_set),
                                                                                            profit=profit,
                                                                                            param_ind=param_ind,
                                                                                            original_param=original_param)

            # If not doing coordinate descent add below to return layer to original
            # with torch.no_grad():
            #     current_temp_layer.weight[param_ind] = current_original_layer.weight[param_ind]

            # print('manual before, : manual: {}'.format(prev_manual_profit))
            prev_profit = profit
            profit = benchmark_best_transition_point.true_profit
            gradient = benchmark_best_transition_point.x - self.get_layer(layer_no).bias.data[
                param_ind].detach().numpy().astype(float)

            if gradient != 0:
                # print('best transition point profit: ', benchmark_best_transition_point.true_profit)
                # print("before:",self.weight.data[param_ind], "transition point:", benchmark_best_transition_point.x)
                # print("gradient: {}".format(gradient))
                with torch.no_grad():
                    self.get_layer(layer_no).bias.data[param_ind] = self.get_layer(layer_no).bias.data[
                                                                        param_ind] + self.dnl_learning_rate * gradient
                # print("after:",self.weight.data[param_ind])

    def get_MSE(self, X, Y):
        predicted_values = self.forward(X).detach().numpy()
        MSE = np.median((Y - predicted_values) ** 2)
        return MSE

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def print(self):
        first_line = ['Method', 'Max Step Size Order', 'Min Step Size Order', "Layer params", "params_per_epoch",
                      "leaky_slope", 'Run Time Limit', 'Epoch Limit',
                      'Mini Batch Size', 'Learning rate', 'Parallelism', "Bias", "L2 Lambda"]
        second_line = [self.sampling_method, self.max_step_size_magnitude, self.min_step_size_magnitude,
                       self.layer_params, self.number_of_parameters_to_update, self.leaky_slope,
                       self.run_time_limit, self.dnl_epoch, self.dnl_batch_size, self.dnl_learning_rate,
                       self.is_parallel, self.is_update_bias, self.L2_lambda]
        third_line = ['epochs', 'sub epochs', 'run time', 'val obj', 'val regret', 'test obj', 'test regret',
                      'regret ratio%', 'test_MSE']
        new_test_predy = [[] for i in range(self.test_pred_y[0].shape[0])]
        ratios = [self.test_regrets[i] * 100 / self.test_objs[i] for i in range(len(self.test_regrets))]
        for i in range(len(self.test_pred_y)):
            for j, predictions in enumerate(new_test_predy):
                new_test_predy[j].append(float(self.test_pred_y[i][j]))

        significant_number = 3
        rest = round_list([self.epochs, self.sub_epochs, self.run_time, self.val_objs, self.val_regrets, self.test_objs,
                           self.test_regrets, ratios, self.test_MSEs
                           ], significant_number)
        # rest.extend(new_test_predy)
        rest = np.array(rest).T.tolist()
        print = []
        print.append(first_line)
        print.append(second_line)
        print.append(third_line)
        print.extend(rest)
        return print

    def get_file_name(self, file_type='.csv'):
        file_name = str(self.sampling_method) + '-' + str(self.max_step_size_magnitude) + str(
            self.min_step_size_magnitude) + file_type
        return file_name


def round_list(arrs, sig_number):
    rounded_list = [[round(elem, sig_number) for elem in arr] for arr in arrs]
    return rounded_list


def get_regret_worker(Y, pred_Ys, weights, opt_params):
    average_objective_value_with_predicted_items, predicted_sols = get_optimization_objective(Y=[Y], pred_Y=[pred_Ys],
                                                                                              weights=[weights],
                                                                                              opt_params=opt_params,
                                                                                              )
    optimal_average_objective_value, sols = get_optimal_average_objective_value(Y=[Y], weights=[weights],
                                                                                opt_params=opt_params,
                                                                                )
    return average_objective_value_with_predicted_items, predicted_sols[0], optimal_average_objective_value, sols[0]
