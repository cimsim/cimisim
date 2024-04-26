import numpy as np


from layers import *
from layer_utils import *




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
        network = [784, 100],
        num_classes=10,
        weight_scale=1e-3,
        quant_params = None,
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
          self.params['W{}'.format(count+1)] = np.random.normal(0, weight_scale, size = (dim1, dim2))
          self.params['b{}'.format(count+1)] = np.zeros(dim2)
          #print('Created W{} with shape {}!'.format(count+1, self.params['W{}'.format(count+1)].shape))


       
        self.params['W{}'.format(len(self.network_dims))] = np.random.normal(0, weight_scale, size = (self.network_dims[-1], num_classes))
        #print('Created W{} with shape {}!'.format(len(self.network_dims), self.params['W{}'.format(len(self.network_dims))].shape))
        self.params['b{}'.format(len(self.network_dims))] = np.zeros(num_classes)


        if quant_params is not None:
          WeightStep, _, _, _, _, _, Weights, increasing_weights, _, mode, _, _, _ = quant_params
          if mode == 'nonlinear_single':
            initial_weights = Weights[abs(Weights) < 1]
            for count, dim in enumerate(zip(self.network_dims, self.network_dims[1:])):
              dim1, dim2 = dim
              self.params['W{}'.format(count+1)] = initial_weights[np.random.randint(len(initial_weights), size=(dim1, dim2))]
              self.params['b{}'.format(count+1)] = Weights[abs(Weights).argmin()]*np.ones(dim2)
         
            self.params['W{}'.format(len(self.network_dims))] = initial_weights[np.random.randint(len(initial_weights), size=(self.network_dims[-1], num_classes))]
            self.params['b{}'.format(len(self.network_dims))] = Weights[abs(Weights).argmin()]*np.ones(num_classes)
          elif mode ==  'linear_single':
            for count, dim in enumerate(zip(self.network_dims, self.network_dims[1:])):
              dim1, dim2 = dim
              self.params['W{}'.format(count+1)] = np.rint((0.1*(2*np.random.random((dim1, dim2))-1))/WeightStep)*WeightStep
              self.params['b{}'.format(count+1)] =  np.zeros(dim2)
              #print('Created W{} with shape {}!'.format(count+1, self.params['W{}'.format(count+1)].shape))


            self.params['W{}'.format(len(self.network_dims))] = np.rint((0.1*(2*np.random.random((self.network_dims[-1], num_classes))-1))/WeightStep)*WeightStep
            self.params['b{}'.format(len(self.network_dims))] = np.zeros(num_classes)
            #print('Created W{} with shape {}!'.format(len(self.network_dims), self.params['W{}'.format(len(self.network_dims))].shape))
          elif mode == 'nonlinear_double':
            increasing_weights_msb, increasing_weights_lsb = increasing_weights
            weights_msb = increasing_weights_msb.keys()
            weights_lsb = increasing_weights_lsb.keys()




            initial_weights_msb = []
            initial_weights_lsb = []
            weight_absmin = (np.abs(Weights).max(), np.abs(Weights).max())
            for msb in weights_msb:
              for lsb in weights_lsb:
                if (abs(msb + lsb) < 1):
                    initial_weights_msb.append(msb)
                    initial_weights_lsb.append(lsb)
                if abs(msb + lsb) < abs(sum(weight_absmin)):
                  weight_absmin = (msb, lsb)
            initial_weights_msb = np.array(initial_weights_msb)
            initial_weights_lsb = np.array(initial_weights_lsb)
         
            bias_msb, bias_lsb = weight_absmin


            for count, dim in enumerate(zip(self.network_dims, self.network_dims[1:])):
              dim1, dim2 = dim
              w_indices = np.random.randint(len(initial_weights_msb), size=(dim1, dim2))
              w_msb = initial_weights_msb[w_indices]
              w_lsb = initial_weights_lsb[w_indices]
              self.params['W{}'.format(count+1)] = (w_msb, w_lsb)
              b_msb =  bias_msb * np.ones(dim2)
              b_lsb = bias_lsb * np.ones(dim2)
              self.params['b{}'.format(count+1)] = (b_msb, b_lsb)


           
            Wlast_indices = np.random.randint(len(initial_weights_msb), size=(self.network_dims[-1], num_classes))
            Wlast_msb = initial_weights_msb[Wlast_indices]
            Wlast_lsb = initial_weights_lsb[Wlast_indices]
            self.params['W{}'.format(len(self.network_dims))] = (Wlast_msb, Wlast_lsb)
           
            b_last_msb = bias_msb * np.ones(num_classes)
            b_last_lsb = bias_lsb * np.ones(num_classes)
            self.params['b{}'.format(len(self.network_dims))] = (b_last_msb, b_last_lsb)
           
          elif mode == 'linear_double':
            WeightStep_msb, WeightStep_lsb = WeightStep


            for count, dim in enumerate(zip(self.network_dims, self.network_dims[1:])):
              dim1, dim2 = dim
              W_msb = np.zeros((dim1, dim2))
              W_lsb = np.rint((0.1*(2*np.random.random((dim1, dim2))-1))/WeightStep_lsb)*WeightStep_lsb
              self.params['W{}'.format(count+1)] = (W_msb, W_lsb)
              b_msb = np.zeros(dim2)
              b_lsb = np.zeros(dim2)
              self.params['b{}'.format(count+1)] = (b_msb, b_lsb)


            Wlast_msb = np.zeros((self.network_dims[-1], num_classes))
            Wlast_lsb = np.rint((0.1*(2*np.random.random((self.network_dims[-1], num_classes))-1))/WeightStep_lsb)*WeightStep_lsb
            self.params['W{}'.format(len(self.network_dims))] = (Wlast_msb, Wlast_lsb)
           
            b_last_msb = np.zeros(num_classes)
            b_last_lsb = np.zeros(num_classes)
            self.params['b{}'.format(len(self.network_dims))] = (b_last_msb, b_last_lsb)






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
       
        cache = []
        intermediate_pass = X
        for i in range(len(self.network_dims) -1):
          intermediate_pass, intermediate_cache = affine_tanh_forward(intermediate_pass, self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)])
          cache.append(intermediate_cache)
        scores, final_cache = affine_forward(intermediate_pass, self.params['W{}'.format(len(self.network_dims))], self.params['b{}'.format(len(self.network_dims))])
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
          if type(self.params['W{}'.format(i+1)]) is not tuple:
            loss = loss + 0.5*self.reg*np.sum(self.params['W{}'.format(i+1)]**2)
          else:
            loss = loss + 0.5*self.reg*np.sum(sum(self.params['W{}'.format(i+1)])**2)
        dx = []
        dw = []
        db = []
        dx_inter, dw_inter, db_inter = affine_backward(loss_grad, cache[-1], self.quant_params)
        dx.append(dx_inter)
        dw.append(dw_inter)
        db.append(db_inter)




        for idx, c in enumerate(reversed(cache[:-1])):
          l = len(self.network_dims)
          dx_inter, dw_inter, db_inter = affine_tanh_backward(dx[-1], c, self.quant_params)
          if type(self.params['W{}'.format(l-idx-1)]) is not tuple:
            dw_inter += self.reg*self.params['W{}'.format(l-idx-1)]
          else:
            dw_inter += self.reg*sum(self.params['W{}'.format(l-idx-1)])
          dx.append(dx_inter)
          dw.append(dw_inter)
          db.append(db_inter)
       
        for idx, t in enumerate(zip(reversed(dw), reversed(db))):
          dw_iter, db_iter = t
          grads['W{}'.format(idx+1)] = dw_iter
          grads['b{}'.format(idx+1)] = db_iter


        return loss, grads



