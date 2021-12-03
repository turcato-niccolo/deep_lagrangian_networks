"""
Utilities Source File

Author: Niccolò Turcato (niccolo.turcato@studenti.unipd.it)
"""
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm

import pytorchtools
from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
import robust_fl_with_gps.Project_Utils as Project_FL_Utils
from pytorchtools import EarlyStopping


def unpack_dataset_joint_variables(dataset, n_dof):
    """
        Unpacks a dataset in the format with examples in rows with joint positions, velocities and accelerations in
        columns (in that order)

        Returns matrices q, qv, qa; containing rows of examples for joint positions, velocities and accelerations
    """
    q = dataset[:, 0:n_dof]  # joint positions
    qv = dataset[:, n_dof:n_dof * 2]  # joint velocities
    qa = dataset[:, n_dof * 2:]  # joint accelerations

    return q, qv, qa


def convert_predictions_to_dataset(prediction, features_name, joint_index_list):
    output_labels = []
    for feat_name in features_name:
        output_labels += [feat_name + '_' + str(joint + 1) for joint in joint_index_list]
    predictions_pd = pd.DataFrame(prediction, columns=output_labels)

    return predictions_pd


def nMSE(y, y_hat):
    num_sample = y.size
    return np.sum((y - y_hat) ** 2) / num_sample / np.var(y)


def MSE(y, y_hat):
    num_sample = y.size
    return np.sum((y - y_hat) ** 2) / num_sample


def k_fold_cross_val_model_selection(num_dof, optimizer_lambda, dataset, targets, hyperparameters_list, k_folds=5,
                                     flg_cuda=True, X_val=None, Y_val=None, early_stopping=False, patience=50):
    """
        Perfroms k-fold cross validation for model selection
    """

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=69)

    # Start print

    results_nmse_models = []

    for model_idx, hyperparameters in enumerate(hyperparameters_list):
        print('\n\n--------------------------------')
        print(f'{model_idx}: training with hyperparamers: {hyperparameters}', flush=True)
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            # Print
            print(f'FOLD {fold}', flush=True)
            print('--------------------------------', flush=True)

            # Sample elements randomly from a given list of ids, no replacement.
            train_data = dataset[train_ids]
            train_targets = targets[train_ids]

            val_data = dataset[val_ids]
            val_targets = targets[val_ids]

            # Init the neural network
            delan_model = DeepLagrangianNetwork(num_dof, **hyperparameters)
            delan_model = delan_model.cuda(torch.device('cuda:0')) if flg_cuda else delan_model.cpu()
            optimizer = optimizer_lambda(delan_model.parameters(), hyperparameters["learning_rate"],
                                         hyperparameters["weight_decay"], True)
            if early_stopping:
                earlystop = EarlyStopping(patience, False,
                                          path='data/model_selection/checkpoint_mod{}.pt'.format(model_idx))
                delan_model.train_model(train_data, train_targets, optimizer, save_model=False,
                                        early_stopping=earlystop, X_val=X_val, Y_val=Y_val)
            else:
                delan_model.train_model(train_data, train_targets, optimizer, save_model=False)

            # Process is complete.
            print('Training process has finished.', flush=True)

            # Print about testing
            print('Starting testing', flush=True)

            # Evaluationfor this fold
            with torch.no_grad():
                est_val_targets = delan_model.evaluate(val_data)
            nMse = nMSE(val_targets, est_val_targets)
            results[fold] = nMse

        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum_nmse = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value}')
            sum_nmse += value
        avg = sum_nmse / len(results.items())
        print(f'Average: {avg}')
        results_nmse_models.append(avg)

    return np.argmax(np.array(results_nmse_models))


def data_efficiency_test(hyperparams, X_tr, Y_tr, X_test, Y_test, step_portion=0.05, k_repetition=5, flg_cuda=True):
    """
    Function that performs data efficiency test, given a step portion, it performs training ok k models on portions of
    increasing size from the minimum portion (=step portion) size to the full training set size.
    [Example: step=0.05 (=5%) -> 5%, 10%, 15%, ..., 100% (20 sizes)


    :param hyperparams: hyperparameters of the model,
                        - Minibatch size must be set such that min train set size >= minibatch size
                        - Max epoch set to the value the user wants the last model(s) that uses the full train set to
                            have, the smaller train set size model have max epoch set such that optimization updates are
                            constant
    :param X_tr: training set features to be sampled (n_samples x (3*#dof))
    :param Y_tr: training set targets to be sampled (n_samples x dof)

    :param X_test: test set features used to evaluete models performance (n_samples x (3*#dof))
    :param Y_test: test set targets used to evaluete models performance (n_samples x dof)
    :param step_portion: incremental size to use for data eff test
    :param k_repetition: number of model to train and test for each train set size

    :return: a dictionary containing entries of G_MSE and n_MSE of each training set size, for each
    :param flg_cuda: If true uses first cuda device, uses cpu otherwise
    """
    num_dof = Y_tr.shape[1]
    n_steps = math.ceil(1 / step_portion)
    num_data_tr = X_tr.shape[0]

    hyperparams['n_minibatch'] = math.floor(step_portion * num_data_tr)

    total_max_iter = math.floor(num_data_tr / hyperparams['n_minibatch']) * hyperparams['max_epoch']

    training_steps = [int(num_data_tr * (i + 1) * step_portion) for i in range(n_steps)]

    test_results = dict()
    for k in range(k_repetition):
        test_results[k] = dict()
        test_results[k]['G_MSE'] = dict()
        test_results[k]['n_MSE'] = dict()

    test_results['MEAN'] = dict()
    test_results['MEAN']['G_MSE'] = dict()
    test_results['MEAN']['n_MSE'] = dict()

    for step_num_data_tr in training_steps:
        print("Training with {} samples".format(step_num_data_tr))
        num_iterations_per_epoch = step_num_data_tr / hyperparams['n_minibatch']
        max_epoch = math.ceil(total_max_iter / num_iterations_per_epoch)
        hyperparams['max_epoch'] = max_epoch

        g_mse = 0
        n_mse = 0
        for k in range(k_repetition):
            n_mse_curr_tr = 0
            g_mse_curr_tr = 0
            np.random.seed(k + 1)
            training_idx = np.random.choice(num_data_tr, step_num_data_tr, replace=False)

            X = X_tr[training_idx, :]
            Y = Y_tr[training_idx, :]

            delan_model = DeepLagrangianNetwork(num_dof, **hyperparams)
            delan_model = delan_model.cuda(torch.device('cuda:0')) if flg_cuda else delan_model.cpu()
            optimizer = torch.optim.Adam(delan_model.parameters(),
                                         lr=hyperparams["learning_rate"],
                                         weight_decay=hyperparams["weight_decay"],
                                         amsgrad=True)
            flg_save = False

            delan_model.train_model(X, Y, optimizer, save_model=flg_save)

            Y_test_hat = delan_model.evaluate(X_test)

            for i in range(num_dof):
                g_mse_curr_tr += Project_FL_Utils.MSE(Y_test[:, i], Y_test_hat[:, i])
                n_mse_curr_tr += Project_FL_Utils.nMSE(Y_test[:, i], Y_test_hat[:, i])

            n_mse_curr_tr /= num_dof
            n_mse += n_mse_curr_tr  # normalized mse is computed as mean of n_mse between joints

            g_mse += g_mse_curr_tr  # global mse is computed as sum of MSE between joints

            test_results[k]['G_MSE'][step_num_data_tr] = g_mse_curr_tr
            test_results[k]['n_MSE'][step_num_data_tr] = n_mse_curr_tr

        g_mse /= k_repetition
        n_mse /= k_repetition

        print('Global MSE: {}'.format(g_mse))
        print('nomalized MSE: {}'.format(n_mse))

        test_results['MEAN']['G_MSE'][step_num_data_tr] = g_mse
        test_results['MEAN']['n_MSE'][step_num_data_tr] = n_mse

    return test_results


def noising_signals(tau, std_noise, mean_noise):
    """
    Applies random Gaussian Noise to the signal

    :param tau:         Input signal (#samples x dof)
    :param std_noise:   Std dev to use for gaussian noise noise
    :param mean_noise   Means for the Gaussian Noise
    :return: The signal + noise
    """
    noise = np.random.normal(mean_noise, std_noise, tau.shape)

    return tau + noise
