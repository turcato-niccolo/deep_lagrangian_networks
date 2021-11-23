"""
Script for training/testing of Deep Lagrangian Network of simulated data of a Panda robot restricted to its first 3
joints and links.

Author: Niccolò Turcato (niccolo.turcato@studenti.unipd.it)
"""
import itertools
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
import json

from pytorchtools import EarlyStopping

try:
    mp.use("Qt5Agg")
    mp.rc('text', usetex=True)
    # mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    mp.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"
except ImportError:
    pass

data_path = ""
training_file = ""
test_file = ""
training_file_M = ""
test_file_M = ""
saving_path = ""
robot_name = ""
flg_norm = None
num_data_tr = None
flg_train = None
shuffle = 0
N_epoch_print = 0
flg_save = 0
# %%
# Set command line arguments
parser = argparse.ArgumentParser('FE panda GIP estimator')
parser.add_argument('-robot_name',
                    type=str,
                    default='FE_panda3DOF_sim_',
                    help='Name of the robot.')
parser.add_argument('-data_path',
                    type=str,
                    default='./robust_fl_with_gps/Simulated_robots/SympyBotics_sim/FE_panda/',
                    help='Path to the folder containing training and test dasets.')
parser.add_argument('-saving_path',
                    type=str,
                    default='./data/Results/DeLan/',
                    help='Path to the destination folder for the generated files.')
parser.add_argument('-model_saving_path',
                    type=str,
                    default='./data/',
                    help='Path to the destination folder for the generated files.')
parser.add_argument('-training_file',
                    type=str,
                    default='FE_panda3DOF_sim_tr.pkl',
                    help='Name of the file containing the train dataset.')
parser.add_argument('-test_file',
                    type=str,
                    default='FE_panda3DOF_sim_test.pkl',
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
flg_train = True

flg_save = True

flg_load = False
# flg_load = True

# flg_cuda = False
flg_cuda = True  # Watch this

downsampling = 1
num_threads = 4
norm_coeff = 1

# Set the paths
print('Setting paths... ', end='')

# Datasets loading paths
tr_path = data_path + training_file
test_path = data_path + test_file

# Set robot params
print('Setting robot parameters... ', end='')

num_dof = 3
joint_index_list = range(0, num_dof)
robot_structure = [0] * num_dof  # 0 = revolute, 1 = prismatic
joint_names = [str(joint_index) for joint_index in range(1, num_dof + 1)]
# features_name_list = [str(i) for i in range(1, num_dof +1)]
output_feature = 'tau'

print('Done!')

# Load datasets
print('Loading datasets... ', end='')

q_names = ['q_' + joint_name for joint_name in joint_names]
dq_names = ['dq_' + joint_name for joint_name in joint_names]
ddq_names = ['ddq_' + joint_name for joint_name in joint_names]
input_features = q_names + dq_names + ddq_names
pos_indices = range(0, num_dof)
acc_indices = range(2 * num_dof, 3 * num_dof)
input_features_joint_list = [input_features] * num_dof

X_tr, Y_tr, active_dims_list, data_frame_tr = Project_FL_Utils.get_data_from_features(tr_path,
                                                                                      input_features,
                                                                                      input_features_joint_list,
                                                                                      output_feature,
                                                                                      num_dof)

X_test, Y_test, active_dims_list, data_frame_test = Project_FL_Utils.get_data_from_features(test_path,
                                                                                            input_features,
                                                                                            input_features_joint_list,
                                                                                            output_feature,
                                                                                            num_dof)
path_suff = 'model_selection_'
if downsampling > 1:
    path_suff+='downsampling'+str(downsampling)+'_'
    print('## Downsampling signals... ', end='')
    X_tr = X_tr[::downsampling]
    Y_tr = Y_tr[::downsampling]
    X_test = X_test[::downsampling]
    Y_test = Y_test[::downsampling]
    print('Done!')

print("\n\n################################################")
print("# Training Samples = {0:05d}".format(int(X_tr.shape[0])))
print("# Test Samples = {0:05d}".format(int(X_test.shape[0])))
print("")

# Training Parameters:
print("\n################################################")
print("Model selection Deep Lagrangian Networks (DeLaN):")

# Construct Hyperparameters:
hyper_base = {'n_width': 64,
         'n_depth': 2,
         'diagonal_epsilon': 0.01,
         'activation': 'SoftPlus',
         'b_init': 1.e-4,
         'b_diag_init': 0.001,
         'w_init': 'orthogonal',
         'gain_hidden': np.sqrt(2.),
         'gain_output': 0.1,
         'n_minibatch': 512,
         'learning_rate': 10.e-03,
         'weight_decay': 1.e-5,
         'max_epoch': 10000}

# Define list of hyperparameters
hyper_list = []
parameters = [
    [64, 128], # n_width
    [2, 3, 4], # n_depth
    ['ReLu', 'SoftPlus', 'Cos'], # activations
    #[1e-3, 5e-3, 1e-2], # b_diag_inits
    ['xavier_normal', 'orthogonal', 'sparse'], # w_init
    #[1e-4, 1e-3, 5e-3, 1e-2, 1e-1] # learning_rates
]

parameters_names = ['n_width', 'n_depth', 'activation', 'w_init']

for combination in itertools.product(*parameters):
    hyper = hyper_base.copy()
    for i, par_name in enumerate(parameters_names):
        hyper[par_name] = combination[i]
    hyper_list.append(hyper)

adam_lambda = lambda model_params, lr, wd, amsgrad: torch.optim.Adam(model_params, lr=lr, weight_decay=wd, amsgrad=amsgrad)


# Splitting test-val dataset
split = 10  # N_train/split samples to val and (Ntrain - N_train/split) to train
val_size = int(X_tr.shape[0] / split)
X_val = X_tr[X_tr.shape[0] - val_size:, :]
Y_val = Y_tr[Y_tr.shape[0] - val_size:, :]
X_tr = X_tr[:X_tr.shape[0] - val_size, :]
Y_tr = Y_tr[:Y_tr.shape[0] - val_size, :]

# If time is strict, consider reducing
downsampling_model_selction = 10
X, Y = X_tr, Y_tr
X = X[::downsampling_model_selction]
Y = Y[::downsampling_model_selction]

patience = int(hyper['max_epoch'] / 400)

early_stopping = True

# Model selection with all training data
idx_best_hyper = Utils.k_fold_cross_val_model_selection(num_dof, adam_lambda, X, Y, hyper_list, flg_cuda=flg_cuda,
                                                        k_folds=2, early_stopping=early_stopping,
                                                        X_val=X_val, Y_val=Y_val)

best_hyper = hyper_list[idx_best_hyper].copy()
best_hyper['max_epoch'] = 20000  # Set the max epoch for the model we decide to train and use

with open('data/selected_model.txt', 'w') as file:
     file.write(json.dumps({'exDict': best_hyper}))

path_suff = 'model_selection_'
if downsampling > 1:
    path_suff+='downsampling'+str(downsampling)+'_'
    print('## Downsampling signals... ', end='')
    X_tr = X_tr[::downsampling]
    Y_tr = Y_tr[::downsampling]
    X_test = X_test[::downsampling]
    Y_test = Y_test[::downsampling]
    print('Done!')

best_hyper['save_file'] = model_saving_path + path_suff + 'delan_panda3DOF_model.torch'

delan_model = DeepLagrangianNetwork(num_dof, **best_hyper)
delan_model = delan_model.cuda(torch.device('cuda:0')) if flg_cuda else delan_model.cpu()
optimizer = torch.optim.Adam(delan_model.parameters(),
                             lr=best_hyper["learning_rate"],
                             weight_decay=best_hyper["weight_decay"],
                             amsgrad=True)

patience = int(hyper['max_epoch'] / 400)

early_stopping = EarlyStopping(patience=patience, verbose=False)

train_loss, val_loss = delan_model.train_model(X_tr, Y_tr, optimizer, save_model=flg_save, early_stopping=early_stopping, X_val=X_val, Y_val=Y_val)

plt.figure()
plt.title('Plot of train-val losses')
plt.plot(train_loss, color='b', label='train loss')
plt.plot(val_loss, color='g', label='val loss')
plt.legend()

X_tr = np.vstack((X_tr, X_val))
Y_tr = np.vstack((Y_tr, Y_val))

delan_model.cpu()

train_qp, train_qv, train_qa = Utils.unpack_dataset_joint_variables(X_tr, num_dof)
test_qp, test_qv, test_qa = Utils.unpack_dataset_joint_variables(X_test, num_dof)

delan_test_tau = delan_model.evaluate(X_test)

q = torch.from_numpy(test_qp).float().to(delan_model.device)
qd = torch.from_numpy(test_qv).float().to(delan_model.device)
qdd = torch.from_numpy(test_qa).float().to(delan_model.device)
zeros = torch.zeros_like(q).float().to(delan_model.device)
#Computing torque decomposition
with torch.no_grad():
    delan_test_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
    delan_test_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_test_g
    delan_test_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_test_g


delan_tr_tau = delan_model.evaluate(X_tr)

# Convert NumPy samples to torch:
q = torch.from_numpy(train_qp).float().to(delan_model.device)
qd = torch.from_numpy(train_qv).float().to(delan_model.device)
qdd = torch.from_numpy(train_qa).float().to(delan_model.device)
zeros = torch.zeros_like(q).float().to(delan_model.device)
#Computing torque decomposition
with torch.no_grad():
    delan_tr_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
    delan_tr_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_tr_g
    delan_tr_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_tr_g


tr_estimates_saving_path = 'data/'+robot_name+path_suff+'DeLaN_train_estimates.pkl'
test_estimates_saving_path = 'data/'+robot_name+path_suff+'DeLaN_test_estimates.pkl'

pd_test_estimates = Utils.convert_predictions_to_dataset(np.hstack([delan_test_tau, delan_test_m, delan_test_c, delan_test_g]),
                                                            ['tau_est', 'm_est', 'c_est', 'g_est'], range(num_dof))

pd_tr_estimates = Utils.convert_predictions_to_dataset(np.hstack([delan_tr_tau, delan_tr_m, delan_tr_c, delan_tr_g]),
                                                            ['tau_est', 'm_est', 'c_est', 'g_est'], range(num_dof))


features = ['tau', 'm', 'c', 'g']
colors_est_plots = ['b', 'g']

for feat in features:
    Project_FL_Utils.print_estimate(data_frame_tr, [pd_tr_estimates],
                                    joint_index_list, flg_print_var=[False],
                                    output_feature=feat, data_noiseless=None, label_prefix='tr\_', color=colors_est_plots)

    Project_FL_Utils.print_estimate(data_frame_test, [pd_test_estimates],
                            joint_index_list, flg_print_var=[False],
                            output_feature=feat, data_noiseless=None, label_prefix='test\_',
                            color=colors_est_plots)

if flg_save:
    print("Saving estimates...")
    pkl.dump(pd_tr_estimates, open(tr_estimates_saving_path, 'wb'))
    pkl.dump(pd_test_estimates, open(test_estimates_saving_path, 'wb'))
    print("Done!")

print("\n################################################\n\n\n")
plt.show()