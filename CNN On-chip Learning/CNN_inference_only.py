# Purpose: Main script for running the simulator for CNN Inference. It conducts experiments on the VGG7 model with varied configurations, including non-linearity values, bit precision, noise, and variations. It trains and evaluates these models on the CIFAR10 dataset, saving results in .csv files.

import tensorflow as tf
import pandas as pd
import keras
from keras import models
import os
from nonlinearity_mapper import *
from get_dataset import *
from keras.datasets import cifar10
import argparse
import csv




def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha_p', '-ap', action = 'store', dest = 'alpha_p',
            type = float, default = 0, help = 'Value of alpha_p, default is 0')
    parser.add_argument('--alpha_d', '-ad', action = 'store', dest = 'alpha_d',
            type = float, default = 0, help = 'Value of alpha_d, default is 0')
    parser.add_argument('--nbits', '-nbits', action = 'store', dest = 'num_bits',
            type = float, default = 5, help = 'Number of bits of the synapse, default is 5')
    parser.add_argument('--pulse_energy', '-pulse_energy', action = 'store', dest = 'pulse_energy',
            type = float, help = 'Energy per synaptic pulse, in J')
    parser.add_argument('--pulse_time', '-pulse_time', action = 'store', dest = 'pulse_time',
            type = float, help = 'Time per synaptic pulse, in seconds')
    parser.add_argument('--d2d_variations_set', '-d2d_set', action = 'store', dest = 'dv_set',
            choices = ['Additive', 'Multiplicative', 'None'], default = 'None',
            help = 'Additive/Multiplicative device to device variations, default is None')
    parser.add_argument('--d2d_variations_val', '-d2d_val', action = 'store', dest = 'dv', default = 0,
            type = float, help = 'If dv_set is True, specify the standard deviation of noise')
    return parser




parser = argparser()
args = parser.parse_args()


results_dir = './results/experimental_results_CIFAR10_VGG7'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


#IF using VGG7
x_train, x_test, y_train, y_test = build_dataset()


#IF USING VGG7


weights_dir = './pretrained_weights_VGG_CIFAR10/'
model = keras.models.load_model(os.path.join(weights_dir, 'vggmodel.h5'))


#g_max is 1.6 for VGG7 and 10 for VGG16
g_max = 1.6
g_min = -1.6


#Fetching the dictionary for alpha_a to A conversion
alpha_to_a = {}
file_path = './nonlinearitymap.csv'
with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    # Skip the header row
    next(csv_reader)
    # Iterate through each row in the CSV
    for row in csv_reader:
        key = float(row[0]) # Assuming the first column contains keys
        value = float(row[1]) # Assuming the second column contains values
        alpha_to_a[key] = value


num_bits = args.num_bits
alpha_p = args.alpha_p
alpha_d = args.alpha_d
pulse_energy = args.pulse_energy
pulse_time = args.pulse_time
dv_set = args.dv_set
dv = args.dv


train_acc_list = []
test_acc_list = []


model_copy = keras.models.clone_model(model)
opt = tf.keras.optimizers.SGD(lr = 0.01, momentum = 0.9, nesterov=True)


#Below code is for experimental
#Converting alpha to A
a_p = alpha_to_a[alpha_p]
if alpha_d >= 0:
    a_d = alpha_to_a[alpha_d]
else:
    a_d = -alpha_to_a[-alpha_d]
   
pulse_max = 2**num_bits - 1
num_pulses_arr = []
#Converting each convolutional kernel into a real synaptic weight matrix and
# obtain the number of programming pulses required
for idx, w in enumerate(model.trainable_variables):
    w_np = w.numpy()
    w_np_new, num_pulses = find_nearest_v2(w_np, a_p, a_d, g_min, g_max, pulse_max, dv_set, dv)
    num_pulses_arr.append(num_pulses)
    w_np_tensor = tf.convert_to_tensor(w_np_new, dtype = tf.float32)
    model_copy.trainable_variables[idx].assign(w_np_tensor)
model_copy.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
#Evaluate the new weight mapped model on the CIFAR10
_, train_acc = model_copy.evaluate(x_train, y_train)
_, test_acc = model_copy.evaluate(x_test, y_test)
train_acc_list.append(train_acc)
test_acc_list.append(test_acc)


max_pulse = np.max([np.max(arr) for arr in num_pulses_arr])
total_pulses = np.sum([np.sum(arr) for arr in num_pulses_arr])
total_energy = pulse_energy*total_pulses
total_time = pulse_time*max_pulse


df = pd.DataFrame({"Train Accuracy" : np.array(train_acc_list), "Test Accuracy" : np.array(test_acc_list), "Maximum pulse" : np.array(max_pulse), "Total pulses": total_pulses, "Total energy": total_energy, "Total time" : total_time})
file_name = 'device_result.csv'
dest_path = os.path.join(results_dir, file_name)
df.to_csv(dest_path, index = False)
