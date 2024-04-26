#from __future__ import print_function
import time
import numpy as np
import argparse
#import matplotlib.pyplot as plt
import pandas as pd
from fc_net import *
from solver import Solver

# CHANGE WHEN GOING BACK TO OLD DEVICE!!
from makedata import makedata_double_2
#from keras.utils import np_utils
import os

import csv
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

import csv
filename = "a_to_alpha.csv"
rows = []
with open(filename, 'r') as data:
    for line in csv.reader(data):
        rows.append(line)
rows.remove(['Nonlinearity', 'Normalized A'])
alpha_dict = dict()
for alpha in rows:
    # print(sub[0])
    alpha_dict[float((alpha[0]))] = float(alpha[1])

alpha_dict[0.00]=1e3 # For non-linearity 0





def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', action='store', dest='dataset',
                        choices=['fisher_iris', 'mnist', 'fmnist'], default='mnist',
                        help='Choice of dataset, fisher_iris or mnist')

    parser.add_argument('--alpha_p', '-alpha_p', action='store', dest='alpha_p',
                        default=1.00, type=float,
                        help='Non-linearity value for LTP')
    parser.add_argument('--alpha_d', '-alpha_d', action='store', dest='alpha_d',
                        default=1.00, type=float,
                        help='Non-linearity value for LTD')

    parser.add_argument('--N_msb', '-N_msb', action='store', dest='N_msb',
                        default=5, type=int,
                        help='Number of MSB bits')

    parser.add_argument('--N_lsb', '-N_lsb', action='store', dest='N_lsb',
                        default=5, type=int,
                        help='Number of LSB bits')

    parser.add_argument('--w_lim_msb', '-w_lim_msb', action='store', dest='w_lim_msb',
                        default=5.00, type=float,
                        help='Maximum Weight of MSB')

    parser.add_argument('--threshold1', '-threshold1', action='store', dest='threshold1',
                        default=5e-7, type=float,
                        help='Threshold Value for LSB')

    parser.add_argument('--threshold2', '-threshold2', action='store', dest='threshold2',
                        default=0.5, type=float,
                        help='Threshold Value for LSB')

    parser.add_argument('--InputStep', '-InputStep', action='store', dest='InputStep',
                        default=1.00, type=float,
                        help='Input Step Size')

    parser.add_argument('--batch_size', '-batch_size', action='store', dest='batch_size',
                        default=1000, type=int,
                        help='Batch Size')

    parser.add_argument('--cycle_var', '-cycle_var', action='store', dest='cycle_var',
                        default=False, type=bool,
                        help='Add multiplicative cycle-to-cycle variation')

    parser.add_argument('--cycle_var_add', '-cycle_var_add', action='store', dest='cycle_var_add',
                        default=False, type=bool,
                        help='Add additive cycle-to-cycle variation')

    parser.add_argument('--device_var', '-device_var', action='store', dest='device_var',
                        default=False, type=bool,
                        help='Add multiplicative device-to-device variation')

    parser.add_argument('--device_var_add', '-device_var_add', action='store', dest='device_var_add',
                        default=False, type=bool,
                        help='Add additive device-to-device variation')

    parser.add_argument('--sigma_c2c', '-sigma_c2c', action='store', dest='sigma_c2c',
                        default=0.05, type=float,
                        help='Standard Devivation for cycle-to-cycle variation')

    parser.add_argument('--sigma_d2d', '-sigma_d2d', action='store', dest='sigma_d2d',
                        default=0.05, type=float,
                        help='Standard Devivation for device-to-device variation')

    return parser





def main():

    parser = argparser()
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'mnist':
        # Loading MNIST data
        train_data = pd.read_csv('mnist_train.csv')
        X_train = train_data.iloc[:, 1:].values
        y_train = train_data.iloc[:, 0].values
        X_train = X_train / 255

        test_data = pd.read_csv('mnist_test.csv')
        X_test = test_data.iloc[:, 1:].values
        y_test = test_data.iloc[:, 0].values
        X_test = X_test / 255
        input_size = 784
        hidden_size = 100
        network_dims = [input_size, hidden_size]
        num_classes = 10

        epochs = 30

    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val':  X_test,
        'y_val': y_test
    }

    results_dir = './results_{}'.format(dataset)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    mode = 'nonlinear_double'
    print('#########################################################################')
    print('Training Non-Linear Double Synapse Model')
    print('########################################################################')
    N_msb = args.N_msb
    N_lsb = args.N_lsb
    w_lim_msb = args.w_lim_msb
    pow2N = (2**N_msb)
    w_lim_lsb = w_lim_msb / pow2N  
    w_limit = w_lim_msb + w_lim_lsb
    WeightStep = (2 * w_limit) / (pow2N * pow2N)  
    threshold1 = [args.threshold1]
    threshold2 = [args.threshold2]
    InputStep = args.InputStep
    batch_size = args.batch_size
    cycle_var = args.cycle_var
    cycle_var_add = args.cycle_var_add
    device_var = args.device_var
    device_var_add = args.device_var_add
    sigma_c2c = args.sigma_c2c
    sigma_d2d = args.sigma_d2d

    error_d2d_msb_W1 = np.zeros((network_dims[0], network_dims[1]))
    error_d2d_msb_W2 = np.zeros((network_dims[1], num_classes))

    error_d2d_lsb_W1 = np.zeros((network_dims[0], network_dims[1]))
    error_d2d_lsb_W2 = np.zeros((network_dims[1], num_classes))

    error_d2d_msb_b1 = np.zeros((network_dims[1]))
    error_d2d_msb_b2 = np.zeros((num_classes))

    error_d2d_lsb_b1 = np.zeros((network_dims[1]))
    error_d2d_lsb_b2 = np.zeros((num_classes))

    if device_var is True:

        error_d2d_msb_W1 = np.random.normal(
            0, sigma_d2d, size=(network_dims[0], network_dims[1]))
        error_d2d_msb_W2 = np.random.normal(
            0, sigma_d2d, size=(network_dims[1], num_classes))

        error_d2d_lsb_W1 = np.random.normal(
            0, sigma_d2d, size=(network_dims[0], network_dims[1]))
        error_d2d_lsb_W2 = np.random.normal(
            0, sigma_d2d, size=(network_dims[1], num_classes))

        error_d2d_msb_b1 = np.random.normal(
            0, sigma_d2d, size=(network_dims[1]))
        error_d2d_msb_b2 = np.random.normal(
            0, sigma_d2d, size=(num_classes))

        error_d2d_lsb_b1 = np.random.normal(
            0, sigma_d2d, size=(network_dims[1]))
        error_d2d_lsb_b2 = np.random.normal(
            0, sigma_d2d, size=(num_classes))

    if device_var_add is True:

        error_d2d_msb_W1 = np.random.normal(
            0, sigma_d2d * w_lim_msb, size=(network_dims[0], network_dims[1]))
        error_d2d_msb_W2 = np.random.normal(
            0, sigma_d2d * w_lim_msb, size=(network_dims[1], num_classes))

        error_d2d_lsb_W1 = np.random.normal(
            0, sigma_d2d * w_lim_lsb, size=(network_dims[0], network_dims[1]))
        error_d2d_lsb_W2 = np.random.normal(
            0, sigma_d2d * w_lim_lsb, size=(network_dims[1], num_classes))

        error_d2d_msb_b1 = np.random.normal(
            0, sigma_d2d * w_lim_msb, size=(network_dims[1]))
        error_d2d_msb_b2 = np.random.normal(
            0, sigma_d2d * w_lim_msb, size=(num_classes))

        error_d2d_lsb_b1 = np.random.normal(
            0, sigma_d2d * w_lim_lsb, size=(network_dims[1]))
        error_d2d_lsb_b2 = np.random.normal(
            0, sigma_d2d * w_lim_lsb, size=(num_classes))
    #  0.5- 2.5199, 1 - 1.2517,1.5 - 0.8252,2-0.609 ,2.5 - 0.477, 3 - 0.3868, 3.5 - 0.3206, 4 - 0.2691, 4.5 - 0.2274, 5 - 0.1924, 5.5 - 0.1622, 6 - 0.1356, 6.5 - 0.1117, 7 - 0.0901
    a_P = args.alpha_p
    a_D = args.alpha_d
    # A_P = 0.4835
    # A_D = -0.4879

    A_P = 0.2804
    A_D = -0.35
    if a_P==0:
        A_P=1e3
    if a_D==0:
        A_D=1e3      
    print(A_P, A_D, N_msb, N_lsb, batch_size)
    weights, increasing_weights, decreasing_weights, stddev = makedata_double_2(
        N_msb=N_msb, N_lsb=N_lsb, A_P=A_P, A_D=A_D)
    for i in threshold1:
        for j in threshold2:
            print('Threshold1: {}, Threshold2: {}'.format(i, j))
            q_params = WeightStep, w_lim_msb, w_lim_lsb, InputStep, i, j, weights, increasing_weights, decreasing_weights, mode, cycle_var, sigma_c2c, device_var, sigma_d2d, error_d2d_msb_W1, error_d2d_msb_W2, error_d2d_lsb_W1, error_d2d_lsb_W2, error_d2d_msb_b1, error_d2d_msb_b2, error_d2d_lsb_b1, error_d2d_lsb_b2, cycle_var_add, device_var_add
            print("cycle_var: ", cycle_var, " cycle_var_add: ", cycle_var_add,
                    " device_var: ", device_var, " device_var_add: ", device_var_add, " sigma_c2c: ", sigma_c2c, " sigma_d2d: ", sigma_d2d)

            model_quant = TwoLayerNet(
                network_dims, num_classes, quant_params=q_params)
            solver_quant = Solver(model_quant, data, batch_size=batch_size, print_every=10000,
                                    num_epochs=epochs, optim_config={'learning_rate': 0.001})
            solver_quant.train()
            results = pd.DataFrame(
                {'Train accuracy': solver_quant.train_acc_history,
                    'Test accuracy': solver_quant.val_acc_history
                    })
            file_name = 'results_nld_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(
                mode, N_msb + N_lsb, a_P,  a_D, cycle_var, cycle_var_add, device_var, device_var_add, sigma_c2c, sigma_d2d)
            dest_path = os.path.join(results_dir, file_name)
            results.to_csv(dest_path)
    print(a_P, a_D, A_P, A_D, N_msb, N_lsb)


if "__main__" == __name__:
    main()
