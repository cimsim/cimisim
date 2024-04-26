import numpy as np
import time
import random


"""
This file implements stochastic gradient descent. The function accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. 

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.
"""


def sgd(w, dw, config=None, quant_params=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None:
    config = {}
  config.setdefault("learning_rate", 1e-2)
  if quant_params is not None:
    WeightStep, w_limit_msb, w_limit_lsb, _, _, _, _, increasing_weights, decreasing_weights, mode, cycle_var, sigma_c2c, device_var, sigma_d2d, error_d2d_msb_W1, error_d2d_msb_W2, error_d2d_lsb_W1, error_d2d_msb_W2, error_d2d_msb_b1, error_d2d_msb_b2, error_d2d_lsb_b1, error_d2d_lsb_b2, cycle_var_add, device_var_add = quant_params
    # if cycle_var is True:
    #   stddev_msb, stddev_lsb = stddev

    if mode == 'linear_single':
      w_limit = w_limit_msb + w_limit_lsb
      w += -config["learning_rate"] * dw
      w[w > w_limit] = w_limit
      w[w < -w_limit] = -w_limit
    elif mode == 'nonlinear_single':
      dw, w = _next_weight(w, dw, increasing_weights, decreasing_weights)
      w += -config["learning_rate"] * dw
    elif mode == 'linear_double':
      weightstep_msb, _ = WeightStep
      w_msb, w_lsb = w
      w_lsb += -config["learning_rate"] * dw

      indices_saturated_max = np.where(w_lsb > w_limit_lsb)
      w_lsb[indices_saturated_max] = -w_limit_lsb
      w_msb[indices_saturated_max] += weightstep_msb

      indices_saturated_msb = np.where(w_msb > w_limit_msb)
      w_msb[indices_saturated_msb] = w_limit_msb
      w_lsb[indices_saturated_msb] = w_limit_lsb

      indices_saturated_min = np.where(w_lsb < -w_limit_lsb)
      w_lsb[indices_saturated_min] = w_limit_lsb
      w_msb[indices_saturated_min] -= weightstep_msb

      indices_saturated_msb = np.where(w_msb < -w_limit_msb)
      w_msb[indices_saturated_msb] = -w_limit_msb
      w_lsb[indices_saturated_msb] = -w_limit_lsb

      w = (w_msb, w_lsb)
    else:

      dw, w = _next_weight(w, config["learning_rate"] * dw, increasing_weights,
                           decreasing_weights, w_limit_lsb)
      dw_msb, dw_lsb = dw
      indices_changed_msb = np.where(dw_msb != 0)
      indices_changed_lsb = np.where(dw_lsb != 0)

      w_msb, w_lsb = w
      w_msb += -dw_msb
      w_lsb += -dw_lsb


      w = (w_msb, w_lsb)
      #print("MSB", w_msb)
      #print("LSB", w_lsb)
  else:
    w += -config["learning_rate"] * dw

  return w, config


# Function to add cycle-to-cycle variation to weights
def _cycle_var(w, stddev, indices):
    # Combine indices as tuples
    indices_zipped = zip(*indices)
    w_noise = np.zeros(w.shape)
    for idx in indices_zipped:
        w_noise[idx] = random.normalvariate(0, stddev[w[idx]])
    return w_noise

# Function to find the nearest key in a dictionary
def _find_nearest_key(search_key, d):
    closest_key = min(d.keys(), key=lambda key: abs(key - search_key))
    closest_val = d[closest_key]
    return closest_key, closest_val

# Helper function for mapping a numpy array using a dictionary
def _mapdict_dict(x, d):
    """
    Helper function for _next_weight
    Function to map a numpy array 'x' of any size using a dictionary 'd'
    This iterates through the dictionary for mapping
    """
    x_original = np.copy(x)
    x_new = np.copy(x)
    for old, new in d.items():
        x_new[x == old] = new
    indices_unmapped = zip(*np.where((x_new - x) == 0))

    for idx in indices_unmapped:
        x_original[idx], x_new[idx] = _find_nearest_key(x_new[idx], d)

    return x_new, x_original

# Helper function for mapping a numpy array using a dictionary
def _mapdict_matrix(x, d):
    """
    Helper function for _next_weight
    Function to map a numpy array 'x' of any size using a dictionary 'd'
    This iterates through the matrix for mapping
    """
    
    def _map(data, d):
        """
        Helper function for _mapdict_matrix
        """
        return d[data]

    _map = np.vectorize(_map, otypes=[float])
    return _map(x, d)

# Function to calculate the next weight with non-linear update
def _next_weight(x, dx, increasing_weights, decreasing_weights, w_limit_lsb=None):
    """
    Function to make the non-linear update.
    Note: Is called only by sgd()

    Inputs:
    - x: Current matrix
    - dx: Ideal gradient
    - weights_list: List containing the possible weights
    - increasing_weights
    - decreasing_weights
    - increasing_weights_jump
    - decreasing_weights_jump
    - jump: Weight change when polarity reverses 

    Returns:
    - dx_real: Real gradient the weights are actually updated by
    - update_sign: Sign matrix of the current update
    """

    dx_real = np.zeros(dx.shape)
    update_sign = np.sign(dx)

    # The below function calculates the 1 bit updates. dx_1 is the weight change taken from
    # decreasing weights, where the sign of gradient > 0. Vice versa for dx_2. This method is good
    # when size of matrix is much larger than dictionary
    if type(x) is not tuple:
        if np.size(x) > len(increasing_weights):
            dx_1, x = _mapdict_dict(x, decreasing_weights) * (update_sign > 0)
            dx_2, x = _mapdict_dict(x, increasing_weights) * (update_sign < 0)
            dx_real = dx_1 + dx_2
            dx_real[update_sign == 0] = 0

        # some other method (better when size of matrix is less than dictionary)
        else:
            dx1 = _mapdict_matrix(x, increasing_weights) * (update_sign < 0)
            dx2 = _mapdict_matrix(x, decreasing_weights) * (update_sign > 0)

            dx_real = dx1 + dx2
            dx_real[update_sign == 0] = 0
        dx_real = -dx_real

    else:
        dx_real = np.zeros(dx.shape)
        update_sign = np.sign(dx)

        x_msb, x_lsb = x
        increasing_weights_msb, increasing_weights_lsb = increasing_weights
        decreasing_weights_msb, decreasing_weights_lsb = decreasing_weights
        w_min = -w_limit_lsb
        w_max = w_limit_lsb

        dx_inc_lsb, x_lsb = _mapdict_dict(x_lsb, increasing_weights_lsb)
        indices_saturated = np.where(dx_inc_lsb == 0)

        dx_inc_msb = np.zeros(dx_inc_lsb.shape)
        dx_inc_msb[indices_saturated], x_msb[indices_saturated] = _mapdict_dict(
            x_msb[indices_saturated], increasing_weights_msb)
        indices_saturated_max = np.where((dx_inc_lsb + dx_inc_msb) == 0)

        dx_inc_lsb *= (update_sign < 0)
        dx_inc_msb *= (update_sign < 0)

        dx_dec_lsb, x_lsb = _mapdict_dict(x_lsb, decreasing_weights_lsb)
        indices_saturated = np.where(dx_dec_lsb == 0)

        dx_dec_msb = np.zeros(dx_dec_lsb.shape)
        dx_dec_msb[indices_saturated], x_msb[indices_saturated] = _mapdict_dict(
            x_msb[indices_saturated], decreasing_weights_msb)
        indices_saturated_max = np.where(dx_dec_lsb + dx_dec_msb == 0)

        dx_dec_lsb *= (update_sign > 0)
        dx_dec_msb *= (update_sign > 0)

        dx_real_lsb = dx_inc_lsb + dx_dec_lsb
        dx_real_msb = dx_inc_msb + dx_dec_msb

        dx_real_lsb[update_sign == 0] = 0
        dx_real_msb[update_sign == 0] = 0

        dx_real_lsb = -dx_real_lsb
        dx_real_msb = -dx_real_msb

        dx_real = (dx_real_msb, dx_real_lsb)
        x = (x_msb, x_lsb)

    return dx_real, x
