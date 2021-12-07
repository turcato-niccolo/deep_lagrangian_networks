"""
Script for Data efficiency evaluation on Deep Lagrangian Networks - 3dof data

Author: Niccolò Turcato (niccolo.turcato@studenti.unipd.it)
"""
import pickle as pkl
import Utils
import robust_fl_with_gps.Project_Utils as Project_FL_Utils

import argparse
import torch
import numpy as np

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
noised_targets_file = ""

path='robust_fl_with_gps/Simulated_robots/SympyBotics_sim/FE_panda/FE_panda_sim_tr.pkl'
# %%
# Set command line arguments
parser = argparse.ArgumentParser('Data efficiency test on 2DOF data')
parser.add_argument('-robot_name',
                    type=str,
                    default='FE_panda3DOF',
                    help='Name of the robot.')
parser.add_argument('-data_path',
                    type=str,
                    default='robust_fl_with_gps/Simulated_robots/SympyBotics_sim/FE_panda/',
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
                    default='FE_panda3DOF_sim_tr.pkl',
                    help='Name of the file containing the train dataset.')
parser.add_argument('-test_file',
                    type=str,
                    default='FE_panda3DOF_sim_test.pkl',
                    help='Name of the file containing the test dataset.')
parser.add_argument('-noised_targets_file',
                    type=str,
                    default='FE_panda3DOF_sim_train_test_targets_noised.pkl',
                    help='Name of the file containing the noised targets.')
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
parser.add_argument('-flg_noise',
                    type=bool,
                    default=False,
                    help='Set the device type')
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

#flg_noise = True
flg_noise = False


flg_cuda = True
# flg_cuda = True  # Watch this

downsampling = 10
num_threads = 4
N_epoch = 500
norm_coeff = 1
dtype = torch.float64

# Set the paths
print('Setting paths... ', end='')

path_suff = ''

# Datasets loading paths
tr_path = data_path + training_file
test_path = data_path + test_file

print('done!')

# Set robot params
print('Setting robot parameters... ', end='')

num_dof = 3
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

if flg_noise:
    noised_targets_file = data_path + noised_targets_file
    Y_tr_noised, Y_test_noised = pkl.load(open(noised_targets_file, 'rb'))
    Y_tr = Y_tr_noised
    Y_test = Y_test_noised
    path_suff += 'noised_'

if downsampling > 1:
    path_suff += "downsampling_" + str(downsampling)
    print('## Downsampling training signals... ', end='')
    X_tr = X_tr[::downsampling]
    Y_tr = Y_tr[::downsampling]
    print('Done!')

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
         'weight_decay': 1.e-5,
         'max_epoch': 150}

# hyper = {"n_width": 128, "n_depth": 2, "diagonal_epsilon": 0.01, "activation": "ReLu", "b_init": 0.0001,
#          "b_diag_init": 0.001, "w_init": "orthogonal", "gain_hidden": 1.4142135623730951, "gain_output": 0.1,
#          "n_minibatch": 512, "learning_rate": 0.01, "weight_decay": 1e-05, "max_epoch": 150}

### DATA EFFICIENCY TEST

# num_data_tr = 2400
num_data_tr = X_tr.shape[0]
print('TRAIN: {}'.format(num_data_tr))

#CALL function
test_results = Utils.data_efficiency_test(hyper, X_tr, Y_tr, X_test, Y_test, k_repetition=10)

print(test_results)


efficiency_results = saving_path + 'orig_model_' + path_suff + 'data_efficiency_results_{}.pkl'.format(robot_name)
pkl.dump(test_results, open(efficiency_results, 'wb'))