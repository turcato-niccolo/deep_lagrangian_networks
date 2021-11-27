"""
Script for Data efficiency evaluation on Deep Lagrangian Networks - 2dof data

Author: Niccolò Turcato (niccolo.turcato@studenti.unipd.it)
"""
import math
import pickle
from collections import defaultdict

from matplotlib import pyplot as plt

import Utils
import robust_fl_with_gps.Project_Utils as Project_FL_Utils

import argparse
import torch
import numpy as np
import time
import pickle as pkl

import PyQt5

import matplotlib as mp

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory
from pytorchtools import EarlyStopping

data_path = ""
training_file = ""
test_file = ""
training_file_M = ""
test_file_M = ""
saving_path = ""
robot_name = ""
flg_norm = None
flg_train = None
shuffle = 0
N_epoch_print = 0
flg_save = 0
# %%
# Set command line arguments
parser = argparse.ArgumentParser('Data efficiency test on 2DOF data')
parser.add_argument('-robot_name',
                    type=str,
                    default='pendulum_2DOF',
                    help='Name of the robot.')
parser.add_argument('-data_path',
                    type=str,
                    default='robust_fl_with_gps/Real_robots/',
                    help='Path to the folder containing training and test dasets.')
parser.add_argument('-saving_path',
                    type=str,
                    default='data_efficiency/',
                    help='Path to the destination folder for the generated files.')
parser.add_argument('-model_saving_path',
                    type=str,
                    default='data_efficiency/data/',
                    help='Path to the destination folder for the generated files.')
parser.add_argument('-training_file',
                    type=str,
                    default='FE_pendulum_2DOF_sim_tr.pkl',
                    help='Name of the file containing the train dataset.')
parser.add_argument('-test_file',
                    type=str,
                    default='FE_pendulum_2DOF_sim_test.pkl',
                    help='Name of the file containing the test dataset.')
parser.add_argument('-flg_load',
                    type=bool,
                    default=False,
                    help='Flag load model. If True the model loaded from memory, otherwise they are computed.')
parser.add_argument('-flg_save',
                    type=bool,
                    default=True,
                    help='Flag save model. If true, the model parameters are saved in memory.')
parser.add_argument('-flg_train',
                    type=bool,
                    default=True,
                    help='Flag train. If True the model parameters are trained.')
parser.add_argument('-batch_size',
                    type=int,
                    default=512,
                    help='Batch size for the training procedure.')
parser.add_argument('-shuffle',
                    type=bool,
                    default=True,
                    help='Shuffle data before training.')
parser.add_argument('-flg_norm',
                    type=bool,
                    default=False,
                    help='Normalize signal.')
parser.add_argument('-N_epoch',
                    type=int,
                    default=5000,
                    help='Number of Epoch for the training procedure.')
parser.add_argument('-N_epoch_print',
                    type=int,
                    default=1,
                    help='Num epoch between two prints during training.')
parser.add_argument('-flg_cuda',
                    type=bool,
                    default=False,
                    help='Set the device type')
parser.add_argument('-num_data_tr',
                    type=int,
                    default=None,
                    help='Number of data to use in the training.')
parser.add_argument('-num_threads',
                    type=int,
                    default=1,
                    help='Number of computational threads.')
parser.add_argument('-downsampling',
                    type=int,
                    default=10,
                    help='Downsampling.')
locals().update(vars(parser.parse_known_args()[0]))

# %%
# Set flags -- for debug
# flg_train = True
flg_train = False

flg_save = True
# flg_save = False

# flg_load = False
flg_load = True

flg_norm = True

flg_cuda = True
# flg_cuda = True  # Watch this

downsampling = 1
num_threads = 4
N_epoch = 500
norm_coeff = 1
dtype = torch.float64

# Set the paths
print('Setting paths... ', end='')

path_suff = '_data_eff'

# Datasets loading paths
tr_path = data_path + training_file
test_path = data_path + test_file

print('done!')

# Set robot params
print('Setting robot parameters... ', end='')

num_dof = 2
joint_index_list = range(0, num_dof)
robot_structure = [0] * num_dof  # 0 = revolute, 1 = prismatic
joint_names = [str(joint_index) for joint_index in range(1, num_dof + 1)]
features_name_list = [str(i) for i in range(1, num_dof + 1)]
output_feature = 'tau'

print('done!')

# Load datasets
print('Loading datasets... ', end='')

q_names = ['q_' + joint_name for joint_name in joint_names]
dq_names = ['dq_' + joint_name for joint_name in joint_names]
ddq_names = ['ddq_' + joint_name for joint_name in joint_names]
input_features = q_names + dq_names + ddq_names
pos_indices = range(0, num_dof)
acc_indices = range(2 * num_dof, 3 * num_dof)
input_features_joint_list = [input_features] * num_dof

# Read the dataset:
X_tr, Y_tr, active_dims_list, data_frame_tr = Project_FL_Utils.get_data_from_features(tr_path,
                                                                                      input_features,
                                                                                      input_features_joint_list,
                                                                                      output_feature,
                                                                                      num_dof)
X_test, Y_test, _, data_frame_test = Project_FL_Utils.get_data_from_features(test_path,
                                                                             input_features,
                                                                             input_features_joint_list,
                                                                             output_feature,
                                                                             num_dof)

# Training Parameters:
print("\n################################################")
print("Training Deep Lagrangian Networks (DeLaN):")

# Construct Hyperparameters:
hyper = {'n_width': 64,
         'n_depth': 2,
         'diagonal_epsilon': 0.01,
         'activation': 'SoftPlus',
         'b_init': 1.e-4,
         'b_diag_init': 0.001,
         'w_init': 'xavier_normal',
         'gain_hidden': np.sqrt(2.),
         'gain_output': 0.1,
         'learning_rate': 5.e-04,
         'weight_decay': 1.e-5}

max_epoch_last_model = 100

### DATA EFFICIENCY TEST

# num_data_tr = 2400
num_data_tr = X_tr.shape[0]
print('TRAIN: {}'.format(num_data_tr))

step_portion = 0.05
n_steps = 20

k_repetition = 5

hyper['n_minibatch'] = math.floor(step_portion * num_data_tr)

total_max_iter = math.floor(num_data_tr / hyper['n_minibatch']) * max_epoch_last_model

training_steps = [int(num_data_tr * (i + 1) * step_portion) for i in range(n_steps)]

test_results = dict()
for k in range(k_repetition):
    test_results[k] = dict()
    test_results[k]['G_MSE'] = dict()
    test_results[k]['n_MSE'] = dict()

test_results['MEAN'] = dict()
test_results['MEAN']['G_MSE'] = dict()
test_results['MEAN']['n_MSE'] = dict()

epochs=[]

for step_num_data_tr in training_steps:
    num_iterations_per_epoch = step_num_data_tr / hyper['n_minibatch']
    max_epoch = math.ceil(total_max_iter / num_iterations_per_epoch)
    hyper['max_epoch'] = max_epoch
    epochs.append(max_epoch)

    g_mse = 0
    n_mse = 0
    for k in range(k_repetition):
        n_mse_curr_tr = 0
        g_mse_curr_tr = 0
        training_idx = np.random.choice(num_data_tr, step_num_data_tr, replace=False)

        X = X_tr[training_idx, :]
        Y = Y_tr[training_idx, :]

        delan_model = DeepLagrangianNetwork(num_dof, **hyper)
        delan_model = delan_model.cuda(torch.device('cuda:0')) if flg_cuda else delan_model.cpu()
        optimizer = torch.optim.Adam(delan_model.parameters(),
                                     lr=hyper["learning_rate"],
                                     weight_decay=hyper["weight_decay"],
                                     amsgrad=True)
        flg_save = False

        train_loss = delan_model.train_model(X, Y, optimizer, save_model=flg_save)

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

print(test_results)
print("Max epochs per trainin set size :")
print(epochs)

efficiency_results = saving_path + 'data_efficiency_results_pendulum2DOF.pkl'
pickle.dump(test_results, open(efficiency_results, 'wb'))
