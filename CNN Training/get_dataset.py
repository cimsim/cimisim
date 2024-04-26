# Purpose: Fetches the CIFAR10 dataset and performs the necessary preprocessing steps required for the pre-trained model.
#Called by: CNN_inference_only.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical


def build_dataset():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    input_shape = (32, 32, 3)


    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
    x_train=x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
    x_test=x_test / 255.0
    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)


    return x_train, x_test, y_train, y_test
