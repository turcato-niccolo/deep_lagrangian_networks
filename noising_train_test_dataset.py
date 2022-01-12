import argparse

import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import Utils
import robust_fl_with_gps.Project_Utils as Project_FL_Utils

data_path = ""
training_file = ""
test_file = ""
saving_path = ""
robot_name = ""
flg_save = False

parser = argparse.ArgumentParser('DeLaN data noiser')
parser.add_argument('-robot_name',
                    type=str,
                    default='pendulum_2DOF',
                    help='Name of the robot.')
parser.add_argument('-data_path',
                    type=str,
                    default='./robust_fl_with_gps/Real_robots/',
                    help='Path to the folder containing training and test dasets.')
parser.add_argument('-saving_path',
                    type=str,
                    default='./robust_fl_with_gps/Real_robots/',
                    help='Path to the destination folder for the generated files.')
parser.add_argument('-training_file',
                    type=str,
                    default='FE_pendulum_2DOF_sim_tr.pkl',
                    help='Name of the file containing the train dataset.')
parser.add_argument('-test_file',
                    type=str,
                    default='FE_pendulum_2DOF_sim_test.pkl',
                    help='Name of the file containing the test dataset.')
parser.add_argument('-flg_save',
                    type=bool,
                    default=True,
                    help='Flag save model. If true, the model parameters are saved in memory.')
locals().update(vars(parser.parse_known_args()[0]))

print('Setting paths... ', end='')
robot_name = 'FE_panda3DOF'
data_path = './robust_fl_with_gps/Simulated_robots/SympyBotics_sim/FE_panda/'
test_file = 'FE_panda3DOF_sim_test.pkl'
saving_path = data_path
training_file = 'FE_panda3DOF_sim_tr.pkl'

# Datasets loading paths
tr_path = data_path + training_file
test_path = data_path + test_file
path_suff = ''

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
# 2DOF - trainSF = testSF = 5e-2
# 3DOF - trainSF = testSF = 5e-2
train_noise_scale_factor = test_noise_scale_factor = 5e-2

tau_tr_std = [np.std(Y_tr[:, i]) for i in range(num_dof)]
train_noise_std = np.array(tau_tr_std) * train_noise_scale_factor
train_noise_mean = np.zeros_like(train_noise_std)

print('Adding Gaussian Noise (mean: {0}, std: {1}) to training data'.format(train_noise_mean, train_noise_std))

Y_tr_noised = Utils.noising_signals(Y_tr, train_noise_std, train_noise_mean)

tau_test_std = [np.std(Y_test[:, i]) for i in range(num_dof)]
test_noise_std = np.array(tau_test_std) * test_noise_scale_factor
test_noise_mean = np.zeros_like(test_noise_std)

print('Adding Gaussian Noise (mean: {0}, std: {1}) to training data'.format(train_noise_mean, train_noise_std))

Y_test_noised = Utils.noising_signals(Y_test, train_noise_std, train_noise_mean)

for i in range(num_dof):
    plt.figure()
    plt.title('Comparison torque and noised torque of joint {} of Train set'.format(i + 1))
    plt.plot(Y_tr_noised[:, i], label='noised signal')
    plt.plot(Y_tr[:, i], label='signal')
    plt.legend()

for i in range(num_dof):
    plt.figure()
    plt.title('Comparison torque and noised torque of joint {} of Test set'.format(i + 1))
    plt.plot(Y_test_noised[:, i], label='noised signal')
    plt.plot(Y_test[:, i], label='signal')
    plt.legend()

plt.show()

saving_file = saving_path + '{0}_sim_train_test_targets_noised.pkl'.format(robot_name)

pkl.dump([Y_tr_noised, Y_test_noised], open(saving_file, 'wb'))
