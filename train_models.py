import os
import random

import numpy as np

from EnergyDataUtil import get_energy_data
from KnapsackSolver import get_opt_params_knapsack
from ReLu_DNL.ReLu_DNL import relu_ppo
from ReLu_DNL.Sampling_Methods import DIVIDE_AND_CONQUER, DIVIDE_AND_CONQUER_GREEDY, DIVIDE_AND_CONQUER_GREEDY_MERGED
from SPO.ICON_Load1SPO import SPO_load1
from dnl.Utils import get_train_test_split_SPO
from SPO.Weighted_knapsack_spo import weighted_knapsack_SPO
from Utils import get_train_test_split
import multiprocessing as mp




def train_relu(file_name_prefix='noprefix', file_folder='', max_step_size_magnitude=0,
                       min_step_size_magnitude=-1,
                       layer_params=None, dropout_percentage=10, path= None,
                       step_size_divider=10, opt_params=None,
                       generate_weight=True, unit_weight=True, is_shuffle=False, print_test=True,
                       test_boolean=None, core_number=7, time_limit=12000, regression_epoch=50, dnl_epoch=3,
                       mini_batch_size=32, dnl_batch_size=None, verbose=False,
                       kfold=0, learning_rate=0.01, dnl_learning_rate=0.3, dataset=None,
                       noise_level=0, is_update_bias=False, L2_Lambda=0.001, is_save=False):
        random.seed(42)
        NUMBER_OF_RANDOM_TESTS = 1
        random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
        # random_seeds = [42 for p in range(NUMBER_OF_RANDOM_TESTS)]
        global divide_conquer_greedy_time, divide_greedy_profit, divide_profit, divide_conquer_time
        if test_boolean is None:
            test_boolean = [0, 1]
        NUMBER_OF_MODELS = 2
        baseline_regression = 2

        training_obj_values_per_epoch = [[[] for n in range(NUMBER_OF_MODELS)] for j in range(NUMBER_OF_RANDOM_TESTS)]
        training_obj_values = np.zeros((NUMBER_OF_MODELS + 1, NUMBER_OF_RANDOM_TESTS))
        epochs = np.zeros((NUMBER_OF_MODELS, NUMBER_OF_RANDOM_TESTS))
        regrets = np.zeros((NUMBER_OF_MODELS + 1, NUMBER_OF_RANDOM_TESTS))
        run_times = np.zeros((NUMBER_OF_MODELS, NUMBER_OF_RANDOM_TESTS))
        test_MSES = np.zeros((NUMBER_OF_MODELS + 1, NUMBER_OF_RANDOM_TESTS))
        training_MSES = np.zeros((NUMBER_OF_MODELS, NUMBER_OF_RANDOM_TESTS))  # might not use

        model_method_names = ['Relu Divide and Conquer',
                              'Relu Divide and Conquer Select Greedy'
                              ]
        if dataset is None:
            dataset = get_energy_data('energy_data.txt', generate_weight=generate_weight, unit_weight=unit_weight,
                                      kfold=kfold, noise_level=noise_level)

        # combine weights with X first
        # may need to split weights
        for random_test_index, random_seed in zip(range(NUMBER_OF_RANDOM_TESTS), random_seeds):
            train_set, test_set = get_train_test_split(dataset, random_seed=random_seed, is_shuffle=is_shuffle)

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

            # benchmark_number = 1
            train_X = benchmarks_train_X
            train_Y = benchmarks_train_Y
            train_weights = benchmarks_weights_train

            val_X = benchmarks_val_X
            val_Y = benchmarks_val_Y
            val_weights = benchmarks_weights_val

            test_X = test_set.get('benchmarks_X')
            test_Y = test_set.get('benchmarks_Y')
            test_weights = test_set.get('benchmarks_weights')
            #
            test_MSE_X = test_set.get('X').T
            test_MSE_Y = test_set.get('Y').T

            number_of_features = X_train[0].shape[0]
            print('layer_params', layer_params)
            layer_params_this = [number_of_features] + layer_params
            if dnl_batch_size is None:
                dnl_batch_size = int(len(benchmarks_train_X))
            mypool = mp.Pool(processes=8)



            baseline_model = relu_ppo(batch_size=mini_batch_size, max_step_size_magnitude=max_step_size_magnitude,
                                      min_step_size_magnitude=min_step_size_magnitude,
                                      dropout_percentage=dropout_percentage,
                                      layer_params=layer_params_this, dnl_epoch=dnl_epoch,
                                      opt_params=opt_params, dnl_batch_size=dnl_batch_size,
                                      dnl_learning_rate=dnl_learning_rate,
                                      is_parallel=True, is_update_bias=is_update_bias, L2_lambda=L2_Lambda,
                                      sampling_method=DIVIDE_AND_CONQUER, run_time_limit=time_limit, path=path)
            baseline_model.fit_nn(X_train, Y_train, max_epochs=regression_epoch)
            baseline_test_regret,_, baseline_test_obj,_ = baseline_model.get_regret(test_X, test_Y, test_weights,
                                                                                pool=mypool)

            baseline_test_regret = np.median(baseline_test_regret)
            baseline_test_obj = np.median(baseline_test_obj)

            mypool.close()

            print('baseline test regret:', baseline_test_regret)

            models = []

            models.append(relu_ppo(batch_size=mini_batch_size, max_step_size_magnitude=max_step_size_magnitude,
                                   min_step_size_magnitude=min_step_size_magnitude,
                                   dropout_percentage=dropout_percentage,
                                   layer_params=layer_params_this, dnl_epoch=dnl_epoch,
                                   opt_params=opt_params, dnl_batch_size=dnl_batch_size,
                                   dnl_learning_rate=dnl_learning_rate,
                                   is_parallel=True, is_update_bias=is_update_bias, L2_lambda=L2_Lambda,
                                   sampling_method=DIVIDE_AND_CONQUER, run_time_limit=time_limit))
            models.append(relu_ppo(batch_size=mini_batch_size, max_step_size_magnitude=max_step_size_magnitude,
                                   min_step_size_magnitude=min_step_size_magnitude,
                                   dropout_percentage=dropout_percentage,
                                   layer_params=layer_params_this, dnl_epoch=dnl_epoch,
                                   opt_params=opt_params, dnl_batch_size=dnl_batch_size,
                                   dnl_learning_rate=dnl_learning_rate,
                                   is_parallel=True, is_update_bias=is_update_bias, L2_lambda=L2_Lambda,
                                   sampling_method=DIVIDE_AND_CONQUER_GREEDY_MERGED, run_time_limit=time_limit))


            # initialize models


            for i, model in enumerate(models):
                if test_boolean[i] == True:
                    model.fit_nn(X_train, Y_train, max_epochs=regression_epoch)
                    MSE = model.get_MSE(test_MSE_X, test_MSE_Y)


            for model, i in zip(models, range(NUMBER_OF_MODELS)):
                if test_boolean[i] == True:
                    print("Starting", model_method_names[i], 'Model')
                    model.fit_dnl(train_X, train_Y, train_weights, val_X, val_Y, val_weights,
                                  benchmark_size=48, test_X=test_X, test_Y=test_Y,
                                  test_weights=test_weights, test_X_MSE=test_MSE_X, test_Y_MSE=test_MSE_Y,
                                  print_test=True)
                    # model.fit_dnl(train_X=train_X, train_Y=train_Y,
                    #                          train_weights=train_weights, val_X=val_X, val_Y=val_Y, val_weights=val_weights,
                    #                          test_X=test_X, test_Y=test_Y,
                    #                          test_weights=test_weights, print_test=print_test, core_number=core_number)
                    print(model_method_names[i], "Running Time:", str(model.run_time[-1]) + "s\n")
                    print(model_method_names[i], "Test Running Time:", str(model.test_run_time) + "s\n")
                    file_name = file_name_prefix
                    model.get_MSE(test_MSE_X, test_MSE_Y)
                    if is_save:
                        print('Saving model {}'.format(model_method_names[i]))
                        f_name = "regression_{}_c{}f{}l{}.pth".format(opt_params.get('solver'),opt_params.get('capacity')[0], kfold,"01")
                        # model.save_regression_model(f_name)
                        f_name = "dnl_noregress_{}_c{}f{}l{}.pth".format(opt_params.get('solver'),opt_params.get('capacity')[0], kfold,"01")
                        model.save_dnl_model(f_name)
            print("----RESULTS----")

            for model, i in zip(models, range(NUMBER_OF_MODELS)):

                if test_boolean[i] == True:
                    print(model_method_names[i], 'Objective Value:', model.training_obj_value[-1], "Running Time:",
                          str(model.run_time[-1]) + "s\n")

                    run_times[i, random_test_index] = model.run_time[-1]
                    training_obj_values_per_epoch[random_test_index][i].extend(model.training_obj_value)
                    training_obj_values[i, random_test_index] = model.training_obj_value[-1]
                    epochs[i, random_test_index] = model.number_of_epochs
                    print(model_method_names[i], 'Objective Value:', model.training_obj_value[-1], "Running Time:",
                          str(model.run_time[-1]) + "s\n")

                    training_obj_values[baseline_regression, random_test_index] = model.training_obj_value[0]


            print("----END----")

            # Tests
            baseline_mse = baseline_model.get_MSE(test_MSE_X, test_MSE_Y)

            regrets[baseline_regression, random_test_index] = baseline_test_regret
            test_MSES[baseline_regression, random_test_index] = baseline_mse



            # print('printing regret baseline = ' + str(baseline_test_regret) + ', printing MSE baseline = ' + str(
            #     baseline_mse))

        #     for model, i in zip(models, range(NUMBER_OF_MODELS)):
        #         if test_boolean[i] == True:
        #             test_regret, _, test_obj,_ = model.get_regret(test_X, test_Y, test_weights, pool=mypool)
        #             test_MSE = model.get_MSE(test_MSE_X, test_MSE_Y)
        #             print(model_method_names[i], 'Regret:', test_regret, "MSE:",
        #                   str(test_MSE) + "s\n")
        #             regrets[i, random_test_index] = test_regret
        #             test_MSES[i, random_test_index] = test_MSE
        # print('printing regret baseline = ' + str(
        #     np.mean(regrets[baseline_regression, :])) + ', printing MSE baseline = ' + str(
        #     np.mean(test_MSES[baseline_regression, :])))
        # for model, i in zip(models, range(NUMBER_OF_MODELS)):
        #     if test_boolean[i] == True:
        #         print(model_method_names[i], 'Regret:', np.mean(regrets[i, :]), "MSE:",
        #               str(np.mean(test_MSES[i, :])) + "s", "Running time", np.mean(run_times[i, :]))

    #
    # def test_intopt(instance_number=1, is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=1,
    #              dest_folder="Tests/icon/intopt/", time_limit=12000, epoch_limit=1):
    #     random.seed(42)
    #     random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    #     random_seed = random_seeds[0]
    #     for kfold in kfolds:
    #         dataset = dnl_energydata.get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True, kfold=kfold)
    #         opt_params = get_icon_instance_params(instance_number)
    #
    #         train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle)
    #         file_name_suffix = 'intoptl' + str(instance_number) + 'k' + str(kfold) + '.csv'
    #         intopt_icon_run(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, opt_params=opt_params,
    #                   instance_number=instance_number, file_name_suffix=file_name_suffix, dest_folder=dest_folder,
    #                   time_limit=time_limit, epoch_limit=epoch_limit)


def train_relu_dnl_knapsack(max_step_size_magnitude=0, min_step_size_magnitude=-1, capacities=None,
                           dnl_epoch=10, regression_epoch=30,
                           kfolds=None,
                           test_boolean=None, core_number=8, is_shuffle=True, layer_params=None,
                           dropout_percentage=10,
                           learning_rate=0.1, dnl_learning_rate=1,
                           dnl_batch_size=None, mini_batch_size=32, n_iter=5,
                            is_save=False):
    # dataset = np.load('Data.npz')
    dataset = None
    if kfolds is None:
        kfolds = [0]
    if capacities is None:
        capacities = [12]

    for capacity in capacities:
            for kfold in kfolds:
                opt_params = get_opt_params_knapsack(capacity=capacity)
                train_relu(dataset=dataset, kfold=kfold,
                               max_step_size_magnitude=max_step_size_magnitude,
                               min_step_size_magnitude=min_step_size_magnitude, layer_params=layer_params,
                               dropout_percentage=dropout_percentage,
                               opt_params=opt_params, dnl_epoch=dnl_epoch, regression_epoch=regression_epoch,
                               is_shuffle=is_shuffle,
                               generate_weight=True, unit_weight=False, core_number=core_number,
                               test_boolean=test_boolean, time_limit=3000,
                               learning_rate=learning_rate, dnl_learning_rate=dnl_learning_rate,
                               dnl_batch_size=dnl_batch_size, mini_batch_size=mini_batch_size, is_save=is_save)





def train_spo_knapsack(capacities=None, is_shuffle=False, layer_params = None, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=5,
                      dest_folder="Tests/icon/Easy/kfolds/spo/", noise_level=0):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    if capacities is None:
        capacities = [12]
    for capacity in capacities:
        for kfold in kfolds:
            dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False, kfold=kfold,
                                      noise_level=noise_level)
            train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle)
            layer_params_str = layer_params[0]
            file_name_prefix = 'Iconknap_c' + str(capacity) + "k" + str(kfold) + "_SPO_l" + "".join(
                ["0" + str(x) for x in layer_params_str]) + ".csv"

            # print("baseline_regret", baseline_regret)
            f_name = "spo_{}_c{}f{}l{}.pth".format("knap",capacity, kfold,"01")
            weighted_knapsack_SPO(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, capacity=capacity,
                                   layer_params=layer_params, dest_folder=dest_folder, save_model=True, f_name=f_name)

def train_spo_icon(capacities=None, is_shuffle=False, layer_params = None, NUMBER_OF_RANDOM_TESTS=1, kfolds=[0], n_iter=5,
                      dest_folder="Tests/icon/Easy/kfolds/spo/", noise_level=0):
    random.seed(42)
    random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
    random_seed = random_seeds[0]
    if capacities is None:
        capacities = [12]
    for capacity in capacities:
        for kfold in kfolds:
            dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True, kfold=kfold,
                                      noise_level=noise_level)
            train_set_SPO, test_set_SPO = get_train_test_split_SPO(dataset, random_seed=random_seed, is_shuffle=is_shuffle)
            layer_params_str = layer_params[0]
            file_name_prefix = 'Iconknap_c' + str(capacity) + "k" + str(kfold) + "_SPO_l" + "".join(
                ["0" + str(x) for x in layer_params_str]) + ".csv"

            # print("baseline_regret", baseline_regret)
            f_name = "spo_{}_l{}f{}l{}.pth".format("icon",capacity, kfold,"01")
            SPO_load1(n_iter=n_iter, train_set=train_set_SPO, test_set=test_set_SPO, capacity=capacity,
                                   layer_params=layer_params, dest_folder=dest_folder, save_model=True, f_name=f_name)

if __name__ == "__main__":
    # capacities = [220]
    capacities  = [12,48,196,220]
    layer_params= [1]
    dropout_percentage = 0
    kfolds = [0]
    test_boolean = [0, 1]
    is_shuffle = True
    # train_relu_dnl_knapsack(max_step_size_magnitude=0, min_step_size_magnitude=-1, capacities=capacities, dnl_epoch=10, layer_params=layer_params,
    #                            dropout_percentage=dropout_percentage,is_shuffle = is_shuffle,
    #                            regression_epoch=0, core_number=8, learning_rate=0.01, dnl_learning_rate=0.1, mini_batch_size=32,
    #                            n_iter=1, is_save=True, kfolds=kfolds, dnl_batch_size=-1, test_boolean=test_boolean)
    layer_params = [[1]]
    train_spo_knapsack (capacities=capacities,layer_params=layer_params, is_shuffle = is_shuffle, kfolds=kfolds, n_iter=1)
    # noise_test_incremental(kfold=0, capacities=capacities, n_iter=100, noise_start=0, noise_end=1000, noise_step=10)
