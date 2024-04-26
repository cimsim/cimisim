# Import necessary libraries
import numpy as np
import tensorflow as tf
import time
import os
import pandas as pd

# Define a function to compute the sign of a tensor element-wise
s = tf.math.sign

# Define a function for truncation of gradients
def truncate(grad, bit_width):
    """
    Truncate the gradient values to a specified bit width.

    Parameters:
    - grad: Input gradient tensor.
    - bit_width: Number of bits for truncation.

    Returns:
    - Truncated gradient tensor.
    """
    return tf.math.round(grad * (2.0**bit_width)) / (2.0**bit_width)

# Define a function for quantization of gradients
def quantize(grad, bit_width):
    """
    Quantize the gradient values to a specified bit width.

    Parameters:
    - grad: Input gradient tensor.
    - bit_width: Number of bits for quantization.

    Returns:
    - Quantized gradient tensor.
    """
    sign_p = tf.math.sign(grad)
    temp = tf.math.abs(grad)
    answer = tf.floor(temp * (2.0**bit_width)) / (2.0**bit_width)
    return tf.math.multiply(answer, sign_p)

# Define a function for calculating potentiation weight
def potentiation_weight(ini=-1, fin=1, A_P=0.5, N=5, pulse_no=1):
    """
    Calculate the potentiation weight based on parameters.

    Parameters:
    - ini: Initial weight value.
    - fin: Final weight value.
    - A_P: Potentiation constant.
    - N: Number of levels for weight quantization.
    - pulse_no: Pulse number.

    Returns:
    - Potentiation weight tensor.
    """
    Gmax = fin
    Gmin = ini
    n_max = 2**(N)
    n = pulse_no / n_max
    pot_wt = (((Gmax - Gmin) * (1 - tf.math.exp(-n / A_P))) /
           (1 - tf.math.exp(-1 / A_P))) + Gmin
    return pot_wt

# Define a function for calculating depression weight
def depression_weight(ini=-1, fin=1, A_D=-0.5, N=5, pulse_no=1):
    """
    Calculate the depression weight based on parameters.

    Parameters:
    - ini: Initial weight value.
    - fin: Final weight value.
    - A_D: Depression constant.
    - N: Number of levels for weight quantization.
    - pulse_no: Pulse number.

    Returns:
    - Depression weight tensor.
    """
    Gmax = fin
    Gmin = ini
    n_max = 2**(N)
    n = pulse_no / n_max
    dep_wt = -((Gmax - Gmin) * (1 - tf.math.exp((1 - n) / A_D)) /
            (1 - tf.math.exp((1) / A_D))) + Gmax
    return dep_wt

# Define a function for calculating the inverse of potentiation weight
def inv_potentiation_weight(ini=-1, fin=1, A_P=0.5, N=5, G=-1):
    """
    Calculate the inverse of potentiation weight to find pulse number.

    Parameters:
    - ini: Initial weight value.
    - fin: Final weight value.
    - A_P: Potentiation constant.
    - N: Number of levels for weight quantization.
    - G: Weight value.

    Returns:
    - Pulse number tensor.
    """
    Gmax = fin
    Gmin = ini
    n_max = 2**(N)

    B = (Gmax - Gmin) / (1 - tf.math.exp(-1 / A_P))
    n = tf.math.round(-A_P * tf.math.log(1 - (G - Gmin) / B) * n_max)
    return n

# Define a function for calculating the inverse of depression weight
def inv_depression_weight(ini=-1, fin=1, A_D=0.5, N=5, G=-1):
    """
    Calculate the inverse of depression weight to find pulse number.

    Parameters:
    - ini: Initial weight value.
    - fin: Final weight value.
    - A_D: Depression constant.
    - N: Number of levels for weight quantization.
    - G: Weight value.

    Returns:
    - Pulse number tensor.
    """
    Gmax = fin
    Gmin = ini
    n_max = 2**(N)
    
    B = -(Gmax - Gmin) / (1 - tf.math.exp(1 / A_D))
    n = tf.math.round((1 - A_D * tf.math.log(1 - (G - Gmax) / B)) * n_max)
    return n


# Define a function to update weights with various options
def update_fast(grad, trainable_weights, lr, refresh_cycle, lsb_width, msb_width, write_noise, std_dev, w_change=0.0, A_P=100, A_D=-100):
    '''
    Calculates the gradient, and updates the weight values, clips them, and then quantizes them. 

    Parameters:
    - grad: Gradient tensor to be updated.
    - trainable_weights: List of trainable weight tensors.
    - lr: Learning rate for weight updates.
    - refresh_cycle: The cycle for refreshing weights.
    - lsb_width: Width of the least significant bits.
    - msb_width: Width of the most significant bits.
    - write_noise: Boolean flag indicating whether to add write noise.
    - std_dev: Standard deviation for write noise.
    - w_change: Cumulative weight change.
    - A_P: Potentiation constant.
    - A_D: Depression constant.

    Returns:
    - Updated gradient tensor.
    - Cumulative weight change.
    - Number of weight updates.
    '''

    num_updates = 0
    answer = (grad.copy())
    
    for i in range(len(grad)):
        curr_weights = tf.clip_by_value(trainable_weights[i], clip_value_min=-1, clip_value_max=1)
        total_width = msb_width + lsb_width
        
        # Calculate the update_w value with optional write noise
        update_w = lr * grad[i]
        if write_noise:
            update_w += tf.random.normal(update_w.shape, 0, std_dev)

        # Calculate the MSB (Most Significant Bits) weight
        weight_msb_init_int_pot = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=msb_width, G=curr_weights)
        weight_msb_pot = potentiation_weight(ini=-1, fin=1, A_P=A_P, N=msb_width, pulse_no=weight_msb_init_int_pot)
        weight_msb_init_int_dep = inv_depression_weight(ini=-1, fin=1, A_D=A_D, N=msb_width, G=curr_weights)
        weight_msb_dep = depression_weight(ini=-1, fin=1, A_D=A_D, N=msb_width, pulse_no=weight_msb_init_int_dep)
        weight_msb_diff_dep = tf.math.abs(curr_weights - weight_msb_dep)
        weight_msb_diff_pot = tf.math.abs(curr_weights - weight_msb_pot)
        weight_msb_pot = tf.cast(weight_msb_diff_dep > weight_msb_diff_pot, curr_weights.dtype) * weight_msb_pot
        weight_msb_dep = tf.cast(weight_msb_diff_dep < weight_msb_diff_pot, curr_weights.dtype) * weight_msb_dep
        weight_msb = weight_msb_dep + weight_msb_pot
        
        # Calculate the LSB (Least Significant Bits) weight
        weight_lsb = truncate(curr_weights - weight_msb, total_width)
        update_w = tf.clip_by_value(update_w, -2**(-msb_width), 2**(-msb_width))
        w_change += tf.reduce_sum(tf.square(tf.abs(update_w)))
        posup_mat = tf.cast(update_w < 0, weight_lsb.dtype)
        negup_mat = tf.cast(update_w > 0, weight_lsb.dtype)
        weight_lsb_pot_int_init = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=total_width, G=weight_lsb)
        weight_lsb_dep_int_init = inv_depression_weight(ini=-1, fin=1, A_D=A_D, N=total_width, G=weight_lsb)
        weight_lsb_pot_int_init_zeros = posup_mat * weight_lsb_pot_int_init
        weight_lsb_dep_int_init_zeros = negup_mat * weight_lsb_dep_int_init
        weight_lsb_int_init = weight_lsb_pot_int_init_zeros + weight_lsb_dep_int_init_zeros
        weight_lsb = weight_lsb - update_w

        max_lsb = 2**(-msb_width)
        weight_lsb_pot_int = inv_potentiation_weight(ini=-1, fin=1, A_P=A_P, N=total_width, G=weight_lsb)
        weight_lsb_pot = potentiation_weight(ini=-1, fin=1, A_P=A_P, N=total_width, pulse_no=weight_lsb_pot_int)
        weight_lsb_dep_int = inv_depression_weight(ini=-1, fin=1, A_D=A_D, N=total_width, G=weight_lsb)
        weight_lsb_dep = depression_weight(ini=-1, fin=1, A_D=A_D, N=total_width, pulse_no=weight_lsb_dep_int)
        weight_lsb_pot_int_fin = posup_mat * weight_lsb_pot_int
        weight_lsb_dep_int_fin = negup_mat * weight_lsb_dep_int
        weight_lsb_int_fin = weight_lsb_pot_int_fin + weight_lsb_dep_int_fin
        weight_lsb_pot = posup_mat * weight_lsb_pot
        weight_lsb_dep = negup_mat * weight_lsb_dep
        weight_lsb = weight_lsb_pot + weight_lsb_dep
        weight_lsb = tf.clip_by_value(weight_lsb, -max_lsb, max_lsb)
        
        refresh_term = (2**(-total_width)) * (refresh_cycle/2.0) * (tf.math.sign(weight_lsb - (-max_lsb + 2**(-total_width-5))) + tf.math.sign(weight_lsb - (max_lsb - 2**(-total_width-5))))
        num_updates += int(tf.reduce_sum(tf.abs(weight_lsb_int_init - weight_lsb_int_fin))) + int(tf.reduce_sum(tf.abs(tf.sign(refresh_term))))

        # If the w_lsb is close to -max_lsb or +max_lsb then we update it to a lower or higher value.
        answer[i] = -(tf.convert_to_tensor(weight_msb + weight_lsb + refresh_term) - trainable_weights[i])

    return answer, w_change, num_updates

# Define a TensorFlow function for fast backpropagation for a single sample
@tf.function      
def fast_backprop_single_sample(x, y, model, loss_fn, opt, temp, acc, lr, msb, lsb, refresh_cycle, write_noise=False, std_dev=0.02, w_change=0.0, lambda_l2=0.00, lambda_l1=0.00, A_P=100, A_D=-100): 
    '''
    Perform fast backpropagation for a single sample.

    Parameters:
    - x: Input data tensor for the sample.
    - y: Ground truth label tensor for the sample.
    - model: Neural network model.
    - loss_fn: Loss function.
    - opt: Optimizer for weight updates.
    - temp: Cumulative loss.
    - acc: Cumulative accuracy.
    - lr: Learning rate for weight updates.
    - msb: Width of the most significant bits.
    - lsb: Width of the least significant bits.
    - refresh_cycle: The cycle for refreshing weights.
    - write_noise: Boolean flag indicating whether to add write noise.
    - std_dev: Standard deviation for write noise.
    - w_change: Cumulative weight change.
    - lambda_l2: L2 regularization term.
    - lambda_l1: L1 regularization term.
    - A_P: Potentiation constant for weight updates.
    - A_D: Depression constant for weight updates.

    Returns:
    - Cumulative loss.
    - Cumulative accuracy.
    - Cumulative weight change.
    - Number of weight updates.
    '''

    with tf.GradientTape() as tape:
        logits = model(x)  # Forward Pass
        loss = loss_fn(y, logits)  # Getting Loss

        # Adding regularization
        if lambda_l2 != 0.00 or lambda_l1 != 0.00:
            for w in model.trainable_weights:
                loss += lambda_l2 * tf.nn.l2_loss(w)
        
        temp += loss
        y_int64 = (tf.cast(y, tf.int64))
        pred = (tf.equal(tf.math.argmax(logits, 1), y_int64))
        acc = tf.cond(pred, lambda: tf.add(acc, 1), lambda: tf.add(acc, 0))
        gradients = tape.gradient(loss, model.trainable_weights)  # Get gradients
        gradients, w_change, num_updates = update_fast(gradients, model.trainable_weights, lr, refresh_cycle, lsb, msb, write_noise, std_dev, w_change, A_P=A_P, A_D=A_D)
        opt.apply_gradients(zip(gradients, model.trainable_weights))
    return temp, acc, w_change, num_updates

def fast_backprop(dataset=None, dataset_test=None, 
                 epochs=50, model=None, loss_fn=None, opt=None,
                 msb=9, lsb=6, write_noise=False, std_dev=0.02, 
                 refresh_freq=10, load_prev_val=False, base_path='./', 
                 lambda_l2=0.00, lambda_l1=0.00, A_P=100, A_D=-100):
    '''
    Perform fast backpropagation training.

    Parameters:
    - dataset: Training dataset.
    - dataset_test: Test dataset.
    - epochs: Number of training epochs.
    - model: Neural network model.
    - loss_fn: Loss function.
    - opt: Optimizer for weight updates.
    - msb: Width of the most significant bits.
    - lsb: Width of the least significant bits.
    - write_noise: Boolean flag indicating whether to add write noise.
    - std_dev: Standard deviation for write noise.
    - refresh_freq: Frequency of weight refreshing.
    - load_prev_val: Boolean flag indicating whether to load previous training values.
    - base_path: Base directory for saving training results.
    - lambda_l2: L2 regularization term.
    - lambda_l1: L1 regularization term.
    - A_P: Potentiation constant for weight updates.
    - A_D: Depression constant for weight updates.

    Returns:
    - List of training accuracies over epochs.
    '''

    acc_hist = []
    test_acc = []
    w_change_hist = []
    num_updates_hist = []
    base_lr = 1e-3
    w_change = 0.0
    weight_suffix = 'weights'
    accuracy_suffix = 'acc'
    print("A_P, A_D", A_P, A_D)

    # Create the requisite directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, accuracy_suffix), exist_ok=True)
    os.makedirs(os.path.join(base_path, weight_suffix), exist_ok=True)
    
    if load_prev_val:
        # Load previously saved training values
        acc_hist = list(pd.read_csv(os.path.join(base_path, 'acc', 'training_acc.csv')).to_numpy()[:, 1])
        test_acc = list(pd.read_csv(os.path.join(base_path, 'acc', 'test_acc.csv')).to_numpy()[:, 1])
        base_lr = pd.read_csv(os.path.join(base_path, 'acc', 'learning_rate.csv')).to_numpy()[0, 1]
        w_change_hist = list(pd.read_csv(os.path.join(base_path, 'acc', 'weight_change.csv')).to_numpy()[:, 1])
        num_updates_hist = list(pd.read_csv(os.path.join(base_path, 'acc', 'updates.csv')).to_numpy()[:, 1])
        tf.print("Base lr =", base_lr, "acc_hist = ", acc_hist, "test_acc = ", test_acc)

    for epoch in range(epochs):
        print("*" * 25, "Epoch {}".format(epoch), "*" * 25, sep='')
        tm = time.time()
        acc = 0.0
        temp = 0.0
        w_change = 0.0
        iterator = iter(dataset)
        lr = base_lr * (0.985**(tf.math.floor(epoch/1.0)))
        print("Learning rate = ", lr.numpy())

        for step in range(len(dataset)):
            x, y = iterator.get_next()
            refresh_cycle = 1 if step % refresh_freq == 0 else 0;
            temp, acc, w_change, num_updates = fast_backprop_single_sample(x, y, model, loss_fn, opt, temp, acc,
                lr, msb, lsb, refresh_cycle, write_noise, std_dev, w_change,
                lambda_l2=lambda_l2, lambda_l1=lambda_l1, A_P=A_P, A_D=A_D)
            num_updates_hist.append(num_updates)
            pd.DataFrame({'updates': num_updates_hist}).to_csv(os.path.join(base_path, accuracy_suffix, 'updates.csv'))

            if step % 1000 == 999:  # Printing training progress
                tf.print("Time taken: ", time.time() - tm)
                tm = time.time()
                step_float = tf.cast(step, tf.float32)
                tf.print("Step:", step, "Loss:", float(temp / step_float))
                tf.print("Train Accuracy: ", acc * 100.0 / step_float)
                tf.print("Average Weight Change: ", w_change)
                acc_hist.append(acc.numpy() * 100.0 / step_float.numpy())
                w_change_hist.append(w_change.numpy())
                pd.DataFrame({'acc': acc_hist}).to_csv(os.path.join(base_path, accuracy_suffix, 'training_acc.csv'))
                pd.DataFrame({'wc': w_change_hist}).to_csv(os.path.join(base_path, accuracy_suffix, 'weight_change.csv'))
                w_change = 0.0

            if step % 50000 == 49999:  # Evaluate test accuracy
                step_test = 0
                acc_test = 0.0

                for x_test_i, y_test_i in dataset_test:
                    step_test += 1
                    logits = model(x_test_i)
                    y_int64 = (tf.cast(y_test_i, tf.int64))
                    pred = (tf.equal(tf.math.argmax(logits, 1), y_int64))
                    acc_test = tf.cond(pred, lambda: tf.add(acc_test, 1), lambda: tf.add(acc_test, 0))

                tf.print("Test Accuracy: ", acc_test * 100 / tf.cast(step_test, tf.float32))
                test_acc.append(acc_test.numpy() * 100 / tf.cast(step_test, tf.float32).numpy())
                pd.DataFrame({'acc_test': test_acc}).to_csv(os.path.join(base_path, accuracy_suffix, 'test_acc.csv'))

        step_float = tf.cast(step, tf.float32)
        pd.DataFrame({'lr': list([lr.numpy()])}).to_csv(os.path.join(base_path, accuracy_suffix, 'learning_rate.csv'))
        model.save_weights(os.path.join(base_path, weight_suffix, 'weights.h5'))
        tf.print("Weights saved!")

    return acc_hist
