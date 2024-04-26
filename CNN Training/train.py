from ast import arg
import tensorflow as tf
from utils import *
from layers import Conv2D_quant, Dense_quant
from model_builder import build_model
import ssl
from get_dataset import *
import argparse
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



# Enable eager execution for TensorFlow 2.x
tf.config.run_functions_eagerly(True)

# Command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--refresh',
                    help="Refresh Cycle Frequency", type=int, default=10)
parser.add_argument('-W', '--write_noise',
                    help="Add write noise", action='store_true')
parser.add_argument('-w', '--write_std_dev',
                    help="Write noise std dev", type=float, default=0.06)
parser.add_argument('-D', '--device_variation',
                    help="Add device variations", action='store_true')
parser.add_argument('-d', '--device_std_dev',
                    help="Device Var Std Dev", type=float, default=0.04)
parser.add_argument('-t', '--total_precision',
                    help="Total Bit Width", type=int, default=15)
parser.add_argument('-l', '--lsb_precision',
                    help="LSB_precision", type=int, default=6)
parser.add_argument('-L', '--load_prev',
                    help="Load Previous Weight values", action='store_true')
parser.add_argument(
    '-P', '--prefix', help="Prefix to where the results are stored", type=str, default="")
parser.add_argument('-S', '--dataset',
                    help="which dataset to use", type=str, default="cifar10")
parser.add_argument('--l2', help="L2 regularization strength",
                    type=float, default=0.00)
parser.add_argument('--l1', help="L1 regularization strength",
                    type=float, default=0.00)
parser.add_argument('--alpha_p', help="non-linearity for LTP", type=float, default=1)
parser.add_argument('--alpha_d', help="non-linearity for LTD", type=float, default=1)

args = parser.parse_args()

# Fixing errors which occur on HPC
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Build the neural network model
model = build_model(th=0.1, dv=args.device_variation, std_dev=args.device_std_dev,
                   add_zero_pad=(not args.dataset == "cifar10"))

opt = tf.keras.optimizers.SGD(learning_rate=1)
model.compile(optimizer=opt,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

if "mnist" in args.dataset:
    model.build(input_shape=tf.TensorShape([None, 28, 28, 1]))
else:
    model.build(input_shape=tf.TensorShape([None, 32, 32, 3]))

print(model.summary())

# Build the training and test datasets
dataset, dataset_test = build_dataset(args.dataset)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Define the base path for saving results
base_path = os.path.join(os.path.abspath("./results/"), args.prefix, "res_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.refresh,
                                                                                                                   args.total_precision,
                                                                                                                   args.lsb_precision,
                                                                                                                   args.write_noise,
                                                                                                                   args.write_std_dev,
                                                                                                                   args.device_variation,
                                                                                                                   args.device_std_dev,
                                                                                                                   args.l2,
                                                                                                                   args.l1, args.alpha_p, args.alpha_d))

s = "Running with the Following Parameters:\nRefresh Cycle: {} Total Precision: {} LSB Precision: {}\
    Write Noise: {} STD: {} DeviceVar: {} STD: {}\
    Load Prev: {} Prefix: {} Base Path: {}\
    L2 Reg: {} L1 Reg = {} A_P = {} A_D = {}".format(args.refresh,
                                                     args.total_precision,
                                                     args.lsb_precision,
                                                     args.write_noise,
                                                     args.write_std_dev,
                                                     args.device_variation,
                                                     args.device_std_dev,
                                                     args.load_prev, args.prefix, base_path,
                                                     args.l2, args.l1, args.alpha_p, args.alpha_d)
print(s)



a_P = args.alpha_p
a_D = args.alpha_d
# A_P = 0.4835
# A_D = -0.4879

A_P = tf.math.sign(a_P) * alpha_dict[np.round(np.abs(a_P), 2)]
A_D = tf.math.sign(a_D) * alpha_dict[np.round(np.abs(a_D), 2)]
if a_P==0:
    A_P=1e3
if a_D==0:
    A_D=1e3      
# Training:
acc_hist = fast_backprop(dataset=dataset,
                        dataset_test=dataset_test,
                        epochs=10,
                        model=model,
                        loss_fn=loss_fn,
                        opt=opt,
                        msb=args.total_precision - args.lsb_precision,
                        lsb=args.lsb_precision,
                        write_noise=args.write_noise,
                        std_dev=args.write_std_dev,
                        refresh_freq=args.refresh,
                        load_prev_val=args.load_prev,
                        base_path=base_path,
                        lambda_l2=args.l2,
                        lambda_l1=args.l1, A_P=A_P, A_D=A_D)
