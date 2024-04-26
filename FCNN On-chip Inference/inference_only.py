import time
import numpy as np
import argparse
import pandas as pd
from fc_net import *
from solver import Solver
from math import exp, log, floor, ceil
import os
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


#Helper function to find the nearest valid synaptic weight
def find_nearest_helper(a, ap, ad, g_min, g_max, p_max):
    is_pot = True


    bp = (g_max - g_min)/(1-exp(-1/ap))
    p_analytic = -p_max*ap*log(abs(1 - (a-g_min)/bp))
    min_diff = abs(a - (bp*(1-exp(-ceil(p_analytic)/(ap*p_max))) + g_min))
    p_match = ceil(p_analytic)
    if abs(a - (bp*(1-exp(-floor(p_analytic)/(ap*p_max))) + g_min)) < min_diff:
        min_diff = abs(a - (bp*(1-exp(-floor(p_analytic)/(ap*p_max))) + g_min))
        p_match = floor(p_analytic)


    bd = (g_max - g_min)/(1-exp(-1/ad))
    p_analytic = p_max*(ad*log(abs(1 + (a - g_max)/bd)) + 1)
    if abs(a - (-bd*(1-exp((floor(p_analytic)/p_max - 1)/ad)) + g_max)) < min_diff:
        min_diff = abs(a - (-bd*(1-exp((floor(p_analytic)/p_max - 1)/ad)) + g_max))
        p_match = floor(p_analytic)
        is_pot = False
    if abs(a - (-bd*(1-exp((ceil(p_analytic)/p_max - 1)/ad)) + g_max)) < min_diff:
        min_diff = abs(a - (-bd*(1-exp((ceil(p_analytic)/p_max - 1)/ad)) + g_max))
        p_match = ceil(p_analytic)
        is_pot = False
   
    if p_match < 0:
        p_match = 0
    if p_match > p_max:
        p_match = p_max


    if is_pot is True:
        return (bp*(1-exp(-p_match/(ap*p_max))) + g_min), p_match
    else:
        return (-bd*(1-exp((p_match/p_max - 1)/ad)) + g_max), p_max + p_match
   


#Wrapper for the helper function, works on the full numpy array
def find_nearest_v2(a, ap, ad, g_min, g_max, p_max, dv_set, dv):
    b = np.copy(a)
    num_pulses = np.zeros(shape = a.shape)
    for idx, j in np.ndenumerate(a):
        b[idx], num_pulses[idx] = find_nearest_helper(j, ap, ad, g_min, g_max, p_max)
    if dv_set == 'Multiplicative':
        b*= (1+np.random.normal(loc = 0.0, scale = dv, size = b.shape)) #Multiplicative noise
    elif dv_set == 'Additive':
        b += np.random.normal(loc = 0.0, scale = dv*(g_max), size = b.shape) #Additive noise
    return b, num_pulses  












def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action = 'store', dest = 'dataset',
            choices = ['mnist'], default = 'mnist',
            help = 'Choice of dataset')
    parser.add_argument('--alpha_p', '-ap', action = 'store', dest = 'alpha_p',
            type = float, default = 0.1, help = 'Value of alpha_p, default is 0.1')
    parser.add_argument('--alpha_d', '-ad', action = 'store', dest = 'alpha_d',
            type = float, default = 0, help = 'Value of alpha_d, default is 0')
    parser.add_argument('--nbits', '-nbits', action = 'store', dest = 'num_bits',
            type = float, default = 5, help = 'Number of bits of the synapse, default is 5')
    parser.add_argument('--pulse_energy', '-pulse_energy', action = 'store', dest = 'pulse_energy',
            type = float, default=0,help = 'Energy per synaptic pulse, in J')
    parser.add_argument('--pulse_time', '-pulse_time', action = 'store', dest = 'pulse_time',
            type = float,default=0, help = 'Time per synaptic pulse, in seconds')
    parser.add_argument('--d2d_variations_set', '-d2d_set', action = 'store', dest = 'dv_set',
            choices = ['Additive', 'Multiplicative', 'None'], default = 'None',
            help = 'Additive/Multiplicative device to device variations, default is None')
    parser.add_argument('--d2d_variations_val', '-d2d_val', action = 'store', dest = 'dv', default = 0,
            type = float, help = 'If dv_set is True, specify the standard deviation of noise')
    return parser


def main():


   
    parser = argparser()
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'mnist':
        #Loading MNIST data
        train_data = pd.read_csv('./datasets/mnist_train.csv')
        X_train = train_data.iloc[:, 1:].values
        y_train = train_data.iloc[:, 0].values
        X_train = X_train/255




        test_data = pd.read_csv('./datasets/mnist_test.csv')
        X_test = test_data.iloc[:, 1:].values
        y_test = test_data.iloc[:, 0].values
        X_test = X_test/255


        network_dims = [784, 100]
        num_classes = 10


        epochs = 30
   
    



    data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val':  X_test,
            'y_val': y_test
        }


    results_dir = './experimental_results/'.format(dataset)
    if not os.path.exists(results_dir):
       os.mkdir(results_dir)
       
   
    print('#########################################################################')
    print('Training ideal model')
    print('#########################################################################')
   
    model_noquant = TwoLayerNet(network_dims, num_classes, weight_scale = 0.1)
    solver_noquant = Solver(model_noquant, data, batch_size = 100, print_every = 10000, num_epochs = 10, optim_config={'learning_rate': 1e-1})
    solver_noquant.train()
    results = pd.DataFrame(
        {'Train accuracy':solver_noquant.train_acc_history,
        'Test accuracy':solver_noquant.val_acc_history
        })
   
    w_max = []
    w_min = []
    b_max = []
    b_min = []
    for count, dim in enumerate(zip(model_noquant.network_dims, model_noquant.network_dims[1:])):
        w_max.append(model_noquant.params['W{}'.format(count+1)].max())
        w_min.append(model_noquant.params['W{}'.format(count+1)].min())
        b_max.append(model_noquant.params['b{}'.format(count+1)].max())
        b_min.append(model_noquant.params['b{}'.format(count+1)].min())


    w_max.append(model_noquant.params['W{}'.format(len(model_noquant.network_dims))].max())
    w_min.append(model_noquant.params['W{}'.format(len(model_noquant.network_dims))].min())
    b_max.append(model_noquant.params['b{}'.format(len(model_noquant.network_dims))].max())
    b_min.append(model_noquant.params['b{}'.format(len(model_noquant.network_dims))].min())


    #Set the maximum and minimum mapped synaptic value
    g_max = 1
    g_min = -1


    num_bits = args.num_bits
    alpha_p = args.alpha_p
    alpha_d = args.alpha_d
    pulse_energy = args.pulse_energy
    pulse_time = args.pulse_time
    dv_set = args.dv_set
    dv = args.dv
       
    train_acc = []
    test_acc = []
    a_p = np.sign(alpha_p) * alpha_dict[np.round(np.abs(alpha_p), 2)]
    a_d = np.sign(alpha_d) * alpha_dict[np.round(np.abs(alpha_d), 2)]
    if alpha_p==0:
        a_p=1e3
    if alpha_d==0:
        a_d=1e3 
    # a_p=alpha_p
    # a_d=alpha_d
    #Convert alpha_p to a_p
    # a_p = alpha_to_a[alpha_p]
    # if alpha_d >= 0:
    #     a_d = alpha_to_a[alpha_d]
    # else:
    #     a_d = -alpha_to_a[-alpha_d]  
           
    pulse_max = 2**num_bits - 1
    # pulse_max=50
           
    new_params = {}


    #Resetting the ideal values to real synaptic values, also obtain number of programming pulses required for each array
    num_pulses_arr = []
    for count, dim in enumerate(zip(model_noquant.network_dims, model_noquant.network_dims[1:])):
        new_params['W{}'.format(count+1)], num_pulse_1 = find_nearest_v2(model_noquant.params['W{}'.format(count+1)], a_p, a_d, g_min, g_max, pulse_max, dv_set, dv)
        new_params['b{}'.format(count+1)], num_pulse_2 = find_nearest_v2(model_noquant.params['b{}'.format(count+1)], a_p, a_d, g_min, g_max, pulse_max, dv_set, dv)
        num_pulses_arr.append(num_pulse_1)
        num_pulses_arr.append(num_pulse_2)
               
    new_params['W{}'.format(len(model_noquant.network_dims))], num_pulse_1 = find_nearest_v2(model_noquant.params['W{}'.format(len(model_noquant.network_dims))], a_p, a_d, g_min, g_max, pulse_max, dv_set, dv)
    new_params['b{}'.format(len(model_noquant.network_dims))], num_pulse_2= find_nearest_v2(model_noquant.params['b{}'.format(len(model_noquant.network_dims))], a_p, a_d, g_min, g_max, pulse_max, dv_set, dv)
    num_pulses_arr.append(num_pulse_1)
    num_pulses_arr.append(num_pulse_2)
           
    #Deterimining the max pulse number for calculating time
    max_pulse = np.max([np.max(arr) for arr in num_pulses_arr])
    total_pulses = np.sum([np.sum(arr) for arr in num_pulses_arr])
    total_energy = pulse_energy*total_pulses
    total_time = pulse_time*max_pulse
    total_energy=0
    total_time=0
           


    model_new = TwoLayerNet(network_dims, num_classes, weight_scale = 0.1)
    model_new.params = new_params
    solver_new = Solver(model_new, data, batch_size = 100, print_every = 10000, num_epochs = 10, optim_config={'learning_rate': 1e-1})


    train_acc.append(solver_new.check_accuracy(X_train, y_train, num_samples=1000))
    test_acc.append(solver_new.check_accuracy(X_test, y_test, num_samples=None))


    df = pd.DataFrame({"Train Accuracy" : np.array(train_acc), "Test Accuracy" : np.array(test_acc), "Maximum pulse" : np.array(max_pulse), "Total pulses": total_pulses, "Total energy": total_energy, "Total time" : total_time})
    file_name = 'device_result.csv'
    dest_path = os.path.join(results_dir, file_name)
    df.to_csv(dest_path, index = False)
   


if "__main__" == __name__:
    main()
