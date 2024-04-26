import numpy as np

from layers import *
from layer_utils import *

np.random.seed(123)


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with tanh nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.

  The architecure is affine - tanh - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(
      self,
      network=[784, 100],
      num_classes=10,
      weight_scale=1e-3,
      quant_params=None,
      device_var=False,
      cycle_var=False,
      reg=0.0,
  ):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    self.quant_params = quant_params
    self.network_dims = network
    self.num_classes = num_classes

    ############################################################################
    # Initializing the weights and biases of the two-layer net. Weights        #
    # are initialized from a Gaussian centered at 0.0 with                     #
    # standard deviation equal to weight_scale, and biases are                 #
    # initialized to zero. All weights and biases are be stored in the         #
    # dictionary self.params, with first layer weights                         #
    # and biases using the keys 'W1' and 'b1' and second layer                 #
    # weights and biases using the keys 'W2' and 'b2'.                         #
    ############################################################################

    for count, dim in enumerate(zip(self.network_dims, self.network_dims[1:])):
      dim1, dim2 = dim
      self.params['W{}'.format(count + 1)] = np.random.normal(0,
                                                              weight_scale, size=(dim1, dim2))
      self.params['b{}'.format(count + 1)] = np.zeros(dim2)

      #print('Created W{} with shape {}!'.format(count+1, self.params['W{}'.format(count+1)].shape))

    self.params['W{}'.format(len(self.network_dims))] = np.random.normal(
        0, weight_scale, size=(self.network_dims[-1], num_classes))
    #print('Created W{} with shape {}!'.format(len(self.network_dims), self.params['W{}'.format(len(self.network_dims))].shape))
    self.params['b{}'.format(len(self.network_dims))] = np.zeros(num_classes)

    if quant_params is not None:
      WeightStep, w_lim_msb, w_lim_lsb, _, _, _, Weights, increasing_weights, _, mode, cycle_var, sigma_c2c, device_var, sigma_d2d, error_d2d_msb_W1, error_d2d_msb_W2, error_d2d_lsb_W1, error_d2d_lsb_W2, error_d2d_msb_b1, error_d2d_msb_b2, error_d2d_lsb_b1, error_d2d_lsb_b2, cycle_var_add, device_var_add = quant_params
      self.device_var = device_var
      self.device_var = device_var

        #print('Created W{} with shape {}!'.format(len(self.network_dims), self.params['W{}'.format(len(self.network_dims))].shape))
      if mode == 'nonlinear_double':

        increasing_weights_msb, increasing_weights_lsb = increasing_weights
        weights_msb = increasing_weights_msb.keys()
        weights_lsb = increasing_weights_lsb.keys()
        initial_weights_msb = []
        initial_weights_lsb = []
        weight_absmin = (np.abs(Weights).max(), np.abs(Weights).max())
        for msb in weights_msb:
          for lsb in weights_lsb:
            # if (abs(msb + lsb) < np.abs(Weights).max()):
            if (abs(msb + lsb) < 1):
              initial_weights_msb.append(msb)
              initial_weights_lsb.append(lsb)
            if abs(msb + lsb) < abs(sum(weight_absmin)):
              weight_absmin = (msb, lsb)

        if not initial_weights_msb and not initial_weights_lsb:
          for msb in weights_msb:
            for lsb in weights_lsb:
              if (abs(msb + lsb) < np.abs(Weights).max()):
                initial_weights_msb.append(msb)
                initial_weights_lsb.append(lsb)
              if abs(msb + lsb) < abs(sum(weight_absmin)):
                weight_absmin = (msb, lsb)

        initial_weights_msb = np.array(initial_weights_msb)
        initial_weights_lsb = np.array(initial_weights_lsb)

        bias_msb, bias_lsb = weight_absmin

        for count, dim in enumerate(zip(self.network_dims, self.network_dims[1:])):
          dim1, dim2 = dim
          np.random.seed(123)

          w_indices = np.random.randint(
              len(initial_weights_msb), size=(dim1, dim2))
          w_msb = initial_weights_msb[w_indices]
          w_lsb = initial_weights_lsb[w_indices]

          self.params['W{}'.format(count + 1)] = (w_msb, w_lsb)
          b_msb = bias_msb * np.ones(dim2)
          b_lsb = bias_lsb * np.ones(dim2)
          self.params['b{}'.format(count + 1)] = (b_msb, b_lsb)

        Wlast_indices = np.random.randint(
            len(initial_weights_msb), size=(self.network_dims[-1], num_classes))
        Wlast_msb = initial_weights_msb[Wlast_indices]
        Wlast_lsb = initial_weights_lsb[Wlast_indices]
        self.params['W{}'.format(len(self.network_dims))] = (
            Wlast_msb, Wlast_lsb)

        b_last_msb = bias_msb * np.ones(num_classes)
        b_last_lsb = bias_lsb * np.ones(num_classes)
        self.params['b{}'.format(len(self.network_dims))] = (
            b_last_msb, b_last_lsb)
        
        # Apply multiplicative device-to-device variation if enabled

        # if device_var is True:
        #   w1_msb, w1_lsb = self.params['W1']
        #   w2_msb, w2_lsb = self.params['W2']
        #   b1_msb, b1_lsb = self.params['b1']
        #   b2_msb, b2_lsb = self.params['b2']
        #   w1_msb = w1_msb * (1 + error_d2d_msb_W1)
        #   w1_lsb = w1_lsb * (1 + error_d2d_lsb_W1)
        #   w2_msb = w2_msb * (1 + error_d2d_msb_W2)
        #   w2_lsb = w2_lsb * (1 + error_d2d_lsb_W2)

        #   b1_msb = b1_msb * (1 + error_d2d_msb_b1)
        #   b1_lsb = b1_lsb * (1 + error_d2d_lsb_b1)
        #   b2_msb = b2_msb * (1 + error_d2d_msb_b2)
        #   b2_lsb = b2_lsb * (1 + error_d2d_lsb_b2)

        #   w1_msb = np.clip(w1_msb, -w_lim_msb, w_lim_msb)
        #   w2_msb = np.clip(w2_msb, -w_lim_msb, w_lim_msb)
        #   b1_msb = np.clip(b1_msb, -w_lim_msb, w_lim_msb)
        #   b2_msb = np.clip(b2_msb, -w_lim_msb, w_lim_msb)

        #   w1_lsb = np.clip(w1_lsb, -w_lim_lsb, w_lim_lsb)
        #   w2_lsb = np.clip(w2_lsb, -w_lim_lsb, w_lim_lsb)
        #   b1_lsb = np.clip(b1_lsb, -w_lim_lsb, w_lim_lsb)
        #   b2_lsb = np.clip(b2_lsb, -w_lim_lsb, w_lim_lsb)
        #   self.params['W1'] = (w1_msb, w1_lsb)
        #   self.params['W2'] = (w2_msb, w2_lsb)
        #   self.params['b1'] = (b1_msb, b1_lsb)
        #   self.params['b2'] = (b2_msb, b2_lsb)
        # Apply additive device-to-device variation if enabled

        if device_var_add is True:
          w1_msb, w1_lsb = self.params['W1']
          w2_msb, w2_lsb = self.params['W2']
          b1_msb, b1_lsb = self.params['b1']
          b2_msb, b2_lsb = self.params['b2']
          w1_msb = w1_msb + error_d2d_msb_W1
          w1_lsb = w1_lsb + error_d2d_lsb_W1
          w2_msb = w2_msb + error_d2d_msb_W2
          w2_lsb = w2_lsb + error_d2d_lsb_W2

          b1_msb = b1_msb + error_d2d_msb_b1
          b1_lsb = b1_lsb + error_d2d_lsb_b1
          b2_msb = b2_msb + error_d2d_msb_b2
          b2_lsb = b2_lsb + error_d2d_lsb_b2

          w1_msb = np.clip(w1_msb, -w_lim_msb, w_lim_msb)
          w2_msb = np.clip(w2_msb, -w_lim_msb, w_lim_msb)
          b1_msb = np.clip(b1_msb, -w_lim_msb, w_lim_msb)
          b2_msb = np.clip(b2_msb, -w_lim_msb, w_lim_msb)

          w1_lsb = np.clip(w1_lsb, -w_lim_lsb, w_lim_lsb)
          w2_lsb = np.clip(w2_lsb, -w_lim_lsb, w_lim_lsb)
          b1_lsb = np.clip(b1_lsb, -w_lim_lsb, w_lim_lsb)
          b2_lsb = np.clip(b2_lsb, -w_lim_lsb, w_lim_lsb)
          self.params['W1'] = (w1_msb, w1_lsb)
          self.params['W2'] = (w2_msb, w2_lsb)
          self.params['b1'] = (b1_msb, b1_lsb)
          self.params['b2'] = (b2_msb, b2_lsb)


        # Apply multiplicatie cycle-to-cycle variation if enabled

        # if cycle_var is True:
        #   error_c2c_msb_W1 = np.random.normal(
        #       0, sigma_c2c, size=(self.network_dims[0], self.network_dims[1]))
        #   error_c2c_msb_W2 = np.random.normal(
        #       0, sigma_c2c, size=(self.network_dims[1], num_classes))

        #   error_c2c_lsb_W1 = np.random.normal(
        #       0, sigma_c2c, size=(self.network_dims[0], self.network_dims[1]))
        #   error_c2c_lsb_W2 = np.random.normal(
        #       0, sigma_c2c, size=(self.network_dims[1], num_classes))

        #   error_c2c_msb_b1 = np.random.normal(
        #       0, sigma_c2c, size=(self.network_dims[1]))
        #   error_c2c_msb_b2 = np.random.normal(
        #       0, sigma_c2c, size=(num_classes))

        #   error_c2c_lsb_b1 = np.random.normal(
        #       0, sigma_c2c, size=(self.network_dims[1]))
        #   error_c2c_lsb_b2 = np.random.normal(
        #       0, sigma_c2c, size=(num_classes))

        #   w1_msb, w1_lsb = self.params['W1']
        #   w2_msb, w2_lsb = self.params['W2']
        #   b1_msb, b1_lsb = self.params['b1']
        #   b2_msb, b2_lsb = self.params['b2']
        #   w1_msb = w1_msb * (1 + error_c2c_msb_W1)
        #   w1_lsb = w1_lsb * (1 + error_c2c_lsb_W1)
        #   w2_msb = w2_msb * (1 + error_c2c_msb_W2)
        #   w2_lsb = w2_lsb * (1 + error_c2c_lsb_W2)

        #   b1_msb = b1_msb * (1 + error_c2c_msb_b1)
        #   b1_lsb = b1_lsb * (1 + error_c2c_lsb_b1)
        #   b2_msb = b2_msb * (1 + error_c2c_msb_b2)
        #   b2_lsb = b2_lsb * (1 + error_c2c_lsb_b2)
        #   w1_msb = np.clip(w1_msb, -w_lim_msb, w_lim_msb)
        #   w2_msb = np.clip(w2_msb, -w_lim_msb, w_lim_msb)
        #   b1_msb = np.clip(b1_msb, -w_lim_msb, w_lim_msb)
        #   b2_msb = np.clip(b2_msb, -w_lim_msb, w_lim_msb)

        #   w1_lsb = np.clip(w1_lsb, -w_lim_lsb, w_lim_lsb)
        #   w2_lsb = np.clip(w2_lsb, -w_lim_lsb, w_lim_lsb)
        #   b1_lsb = np.clip(b1_lsb, -w_lim_lsb, w_lim_lsb)
        #   b2_lsb = np.clip(b2_lsb, -w_lim_lsb, w_lim_lsb)
        #   self.params['W1'] = (w1_msb, w1_lsb)
        #   self.params['W2'] = (w2_msb, w2_lsb)
        #   self.params['b1'] = (b1_msb, b1_lsb)
        #   self.params['b2'] = (b2_msb, b2_lsb)

        # Apply additive cycle-to-cycle variation if enabled

        if cycle_var_add is True:
          error_c2c_msb_W1 = np.random.normal(
              0, sigma_c2c * w_lim_msb, size=(self.network_dims[0], self.network_dims[1]))
          error_c2c_msb_W2 = np.random.normal(
              0, sigma_c2c * w_lim_msb, size=(self.network_dims[1], num_classes))

          error_c2c_lsb_W1 = np.random.normal(
              0, sigma_c2c * w_lim_lsb, size=(self.network_dims[0], self.network_dims[1]))
          error_c2c_lsb_W2 = np.random.normal(
              0, sigma_c2c * w_lim_lsb, size=(self.network_dims[1], num_classes))

          error_c2c_msb_b1 = np.random.normal(
              0, sigma_c2c * w_lim_msb, size=(self.network_dims[1]))
          error_c2c_msb_b2 = np.random.normal(
              0, sigma_c2c * w_lim_msb, size=(num_classes))

          error_c2c_lsb_b1 = np.random.normal(
              0, sigma_c2c * w_lim_lsb, size=(self.network_dims[1]))
          error_c2c_lsb_b2 = np.random.normal(
              0, sigma_c2c * w_lim_lsb, size=(num_classes))

          w1_msb, w1_lsb = self.params['W1']
          w2_msb, w2_lsb = self.params['W2']
          b1_msb, b1_lsb = self.params['b1']
          b2_msb, b2_lsb = self.params['b2']

          w1_msb = w1_msb + error_c2c_msb_W1
          w1_lsb = w1_lsb + error_c2c_lsb_W1
          w2_msb = w2_msb + error_c2c_msb_W2
          w2_lsb = w2_lsb + error_c2c_lsb_W2

          b1_msb = b1_msb + error_c2c_msb_b1
          b1_lsb = b1_lsb + error_c2c_lsb_b1
          b2_msb = b2_msb + error_c2c_msb_b2
          b2_lsb = b2_lsb + error_c2c_lsb_b2
          w1_msb = np.clip(w1_msb, -w_lim_msb, w_lim_msb)
          w2_msb = np.clip(w2_msb, -w_lim_msb, w_lim_msb)
          b1_msb = np.clip(b1_msb, -w_lim_msb, w_lim_msb)
          b2_msb = np.clip(b2_msb, -w_lim_msb, w_lim_msb)

          w1_lsb = np.clip(w1_lsb, -w_lim_lsb, w_lim_lsb)
          w2_lsb = np.clip(w2_lsb, -w_lim_lsb, w_lim_lsb)
          b1_lsb = np.clip(b1_lsb, -w_lim_lsb, w_lim_lsb)
          b2_lsb = np.clip(b2_lsb, -w_lim_lsb, w_lim_lsb)
          self.params['W1'] = (w1_msb, w1_lsb)
          self.params['W2'] = (w2_msb, w2_lsb)
          self.params['b1'] = (b1_msb, b1_lsb)
          self.params['b2'] = (b2_msb, b2_lsb)



  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    WeightStep, w_lim_msb, w_lim_lsb, _, _, _, Weights, increasing_weights, _, mode, cycle_var, sigma_c2c, device_var, sigma_d2d, error_d2d_msb_W1, error_d2d_msb_W2, error_d2d_lsb_W1, error_d2d_lsb_W2, error_d2d_msb_b1, error_d2d_msb_b2, error_d2d_lsb_b1, error_d2d_lsb_b2, cycle_var_add, device_var_add = self.quant_params

    cache = []
    intermediate_pass = X
    for i in range(len(self.network_dims) - 1):
      intermediate_pass, intermediate_cache = affine_tanh_forward(
          intermediate_pass, self.params['W{}'.format(i + 1)], self.params['b{}'.format(i + 1)])
      cache.append(intermediate_cache)
    scores, final_cache = affine_forward(intermediate_pass, self.params['W{}'.format(
        len(self.network_dims))], self.params['b{}'.format(len(self.network_dims))])
    cache.append(final_cache)

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # Backward pass for the two-layer net. Loss is stored                      #
    # in the loss variable and gradients in the grads dictionary.              #
    ############################################################################
    loss, loss_grad = softmax_loss(scores, y)
    for i in range(len(self.network_dims)):
      if type(self.params['W{}'.format(i + 1)]) is not tuple:
        loss = loss + 0.5 * self.reg * \
            np.sum(self.params['W{}'.format(i + 1)]**2)
      else:
        loss = loss + 0.5 * self.reg * \
            np.sum(sum(self.params['W{}'.format(i + 1)])**2)
    dx = []
    dw = []
    db = []
    dx_inter, dw_inter, db_inter = affine_backward(
        loss_grad, cache[-1], self.quant_params)
    dx.append(dx_inter)
    dw.append(dw_inter)
    db.append(db_inter)

    for idx, c in enumerate(reversed(cache[:-1])):
      l = len(self.network_dims)
      dx_inter, dw_inter, db_inter = affine_tanh_backward(
          dx[-1], c, self.quant_params)
      if type(self.params['W{}'.format(l - idx - 1)]) is not tuple:
        dw_inter += self.reg * self.params['W{}'.format(l - idx - 1)]
      else:
        dw_inter += self.reg * sum(self.params['W{}'.format(l - idx - 1)])
      dx.append(dx_inter)
      dw.append(dw_inter)
      db.append(db_inter)

    for idx, t in enumerate(zip(reversed(dw), reversed(db))):
      dw_iter, db_iter = t
      grads['W{}'.format(idx + 1)] = dw_iter
      grads['b{}'.format(idx + 1)] = db_iter

    if device_var is True:
      w1_msb, w1_lsb = self.params['W1']
      w2_msb, w2_lsb = self.params['W2']
      b1_msb, b1_lsb = self.params['b1']
      b2_msb, b2_lsb = self.params['b2']
      w1_msb = w1_msb * (1 + error_d2d_msb_W1)
      w1_lsb = w1_lsb * (1 + error_d2d_lsb_W1)
      w2_msb = w2_msb * (1 + error_d2d_msb_W2)
      w2_lsb = w2_lsb * (1 + error_d2d_lsb_W2)

      b1_msb = b1_msb * (1 + error_d2d_msb_b1)
      b1_lsb = b1_lsb * (1 + error_d2d_lsb_b1)
      b2_msb = b2_msb * (1 + error_d2d_msb_b2)
      b2_lsb = b2_lsb * (1 + error_d2d_lsb_b2)
      w1_msb = np.clip(w1_msb, -w_lim_msb, w_lim_msb)
      w2_msb = np.clip(w2_msb, -w_lim_msb, w_lim_msb)
      b1_msb = np.clip(b1_msb, -w_lim_msb, w_lim_msb)
      b2_msb = np.clip(b2_msb, -w_lim_msb, w_lim_msb)

      w1_lsb = np.clip(w1_lsb, -w_lim_lsb, w_lim_lsb)
      w2_lsb = np.clip(w2_lsb, -w_lim_lsb, w_lim_lsb)
      b1_lsb = np.clip(b1_lsb, -w_lim_lsb, w_lim_lsb)
      b2_lsb = np.clip(b2_lsb, -w_lim_lsb, w_lim_lsb)
      self.params['W1'] = (w1_msb, w1_lsb)
      self.params['W2'] = (w2_msb, w2_lsb)
      self.params['b1'] = (b1_msb, b1_lsb)
      self.params['b2'] = (b2_msb, b2_lsb)

    if device_var_add is True:
      w1_msb, w1_lsb = self.params['W1']
      w2_msb, w2_lsb = self.params['W2']
      b1_msb, b1_lsb = self.params['b1']
      b2_msb, b2_lsb = self.params['b2']
      w1_msb = w1_msb + error_d2d_msb_W1
      w1_lsb = w1_lsb + error_d2d_lsb_W1
      w2_msb = w2_msb + error_d2d_msb_W2
      w2_lsb = w2_lsb + error_d2d_lsb_W2

      b1_msb = b1_msb + error_d2d_msb_b1
      b1_lsb = b1_lsb + error_d2d_lsb_b1
      b2_msb = b2_msb + error_d2d_msb_b2
      b2_lsb = b2_lsb + error_d2d_lsb_b2
      w1_msb = np.clip(w1_msb, -w_lim_msb, w_lim_msb)
      w2_msb = np.clip(w2_msb, -w_lim_msb, w_lim_msb)
      b1_msb = np.clip(b1_msb, -w_lim_msb, w_lim_msb)
      b2_msb = np.clip(b2_msb, -w_lim_msb, w_lim_msb)

      w1_lsb = np.clip(w1_lsb, -w_lim_lsb, w_lim_lsb)
      w2_lsb = np.clip(w2_lsb, -w_lim_lsb, w_lim_lsb)
      b1_lsb = np.clip(b1_lsb, -w_lim_lsb, w_lim_lsb)
      b2_lsb = np.clip(b2_lsb, -w_lim_lsb, w_lim_lsb)
      self.params['W1'] = (w1_msb, w1_lsb)
      self.params['W2'] = (w2_msb, w2_lsb)
      self.params['b1'] = (b1_msb, b1_lsb)
      self.params['b2'] = (b2_msb, b2_lsb)

    if cycle_var is True:
      error_c2c_msb_W1 = np.random.normal(
          0, sigma_c2c, size=(self.network_dims[0], self.network_dims[1]))
      error_c2c_msb_W2 = np.random.normal(
          0, sigma_c2c, size=(self.network_dims[1], self.num_classes))

      error_c2c_lsb_W1 = np.random.normal(
          0, sigma_c2c, size=(self.network_dims[0], self.network_dims[1]))
      error_c2c_lsb_W2 = np.random.normal(
          0, sigma_c2c, size=(self.network_dims[1], self.num_classes))

      error_c2c_msb_b1 = np.random.normal(
          0, sigma_c2c, size=(self.network_dims[1]))
      error_c2c_msb_b2 = np.random.normal(
          0, sigma_c2c, size=(self.num_classes))

      error_c2c_lsb_b1 = np.random.normal(
          0, sigma_c2c, size=(self.network_dims[1]))
      error_c2c_lsb_b2 = np.random.normal(
          0, sigma_c2c, size=(self.num_classes))

      w1_msb, w1_lsb = self.params['W1']
      w2_msb, w2_lsb = self.params['W2']
      b1_msb, b1_lsb = self.params['b1']
      b2_msb, b2_lsb = self.params['b2']
      w1_msb = w1_msb * (1 + error_c2c_msb_W1)
      w1_lsb = w1_lsb * (1 + error_c2c_lsb_W1)
      w2_msb = w2_msb * (1 + error_c2c_msb_W2)
      w2_lsb = w2_lsb * (1 + error_c2c_lsb_W2)

      b1_msb = b1_msb * (1 + error_c2c_msb_b1)
      b1_lsb = b1_lsb * (1 + error_c2c_lsb_b1)
      b2_msb = b2_msb * (1 + error_c2c_msb_b2)
      b2_lsb = b2_lsb * (1 + error_c2c_lsb_b2)
      w1_msb = np.clip(w1_msb, -w_lim_msb, w_lim_msb)
      w2_msb = np.clip(w2_msb, -w_lim_msb, w_lim_msb)
      b1_msb = np.clip(b1_msb, -w_lim_msb, w_lim_msb)
      b2_msb = np.clip(b2_msb, -w_lim_msb, w_lim_msb)

      w1_lsb = np.clip(w1_lsb, -w_lim_lsb, w_lim_lsb)
      w2_lsb = np.clip(w2_lsb, -w_lim_lsb, w_lim_lsb)
      b1_lsb = np.clip(b1_lsb, -w_lim_lsb, w_lim_lsb)
      b2_lsb = np.clip(b2_lsb, -w_lim_lsb, w_lim_lsb)
      self.params['W1'] = (w1_msb, w1_lsb)
      self.params['W2'] = (w2_msb, w2_lsb)
      self.params['b1'] = (b1_msb, b1_lsb)
      self.params['b2'] = (b2_msb, b2_lsb)
    if cycle_var_add is True:
      error_c2c_msb_W1 = np.random.normal(
          0, sigma_c2c * w_lim_msb, size=(self.network_dims[0], self.network_dims[1]))
      error_c2c_msb_W2 = np.random.normal(
          0, sigma_c2c * w_lim_msb, size=(self.network_dims[1], self.num_classes))

      error_c2c_lsb_W1 = np.random.normal(
          0, sigma_c2c * w_lim_lsb, size=(self.network_dims[0], self.network_dims[1]))
      error_c2c_lsb_W2 = np.random.normal(
          0, sigma_c2c * w_lim_lsb, size=(self.network_dims[1], self.num_classes))

      error_c2c_msb_b1 = np.random.normal(
          0, sigma_c2c * w_lim_msb, size=(self.network_dims[1]))
      error_c2c_msb_b2 = np.random.normal(
          0, sigma_c2c * w_lim_msb, size=(self.num_classes))

      error_c2c_lsb_b1 = np.random.normal(
          0, sigma_c2c * w_lim_lsb, size=(self.network_dims[1]))
      error_c2c_lsb_b2 = np.random.normal(
          0, sigma_c2c * w_lim_lsb, size=(self.num_classes))

      w1_msb, w1_lsb = self.params['W1']
      w2_msb, w2_lsb = self.params['W2']
      b1_msb, b1_lsb = self.params['b1']
      b2_msb, b2_lsb = self.params['b2']
      w1_msb = w1_msb + error_c2c_msb_W1
      w1_lsb = w1_lsb + error_c2c_lsb_W1
      w2_msb = w2_msb + error_c2c_msb_W2
      w2_lsb = w2_lsb + error_c2c_lsb_W2

      b1_msb = b1_msb + error_c2c_msb_b1
      b1_lsb = b1_lsb + error_c2c_lsb_b1
      b2_msb = b2_msb + error_c2c_msb_b2
      b2_lsb = b2_lsb + error_c2c_lsb_b2
      w1_msb = np.clip(w1_msb, -w_lim_msb, w_lim_msb)
      w2_msb = np.clip(w2_msb, -w_lim_msb, w_lim_msb)
      b1_msb = np.clip(b1_msb, -w_lim_msb, w_lim_msb)
      b2_msb = np.clip(b2_msb, -w_lim_msb, w_lim_msb)

      w1_lsb = np.clip(w1_lsb, -w_lim_lsb, w_lim_lsb)
      w2_lsb = np.clip(w2_lsb, -w_lim_lsb, w_lim_lsb)
      b1_lsb = np.clip(b1_lsb, -w_lim_lsb, w_lim_lsb)
      b2_lsb = np.clip(b2_lsb, -w_lim_lsb, w_lim_lsb)
      self.params['W1'] = (w1_msb, w1_lsb)
      self.params['W2'] = (w2_msb, w2_lsb)
      self.params['b1'] = (b1_msb, b1_lsb)
      self.params['b2'] = (b2_msb, b2_lsb)

    return loss, grads
