import numpy as np






def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.


    The input x has shape (N, D) and contains a minibatch of N
    examples, where each example x[i] has shape (D, ). We will
    transform each example to an output vector of dimension M.


    Inputs:
    - x: A numpy array containing input data, of shape (N, D)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)


    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    if type(w) is not tuple:
        out = x.dot(w) + b
    else:
        out = x.dot(sum(w)) + sum(b)
    cache = (x, w, b)
 
    return out, cache




def affine_backward(dout, cache, quant_params):
    """
    Computes the backward pass for an affine layer (if quant is not None, it quantizes as well)


    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, D)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)


    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, D)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, W, B = cache
    N = x.shape[0]
    D = x.shape[1]
   
    if type(W) is not tuple:
        w = W
        b = B
    else:
        w = sum(W)
        b = sum(B)


    if quant_params is not None:
        ws, _, _, InputStep, threshold1, threshold2, _, _, _, _, _, _, _= quant_params
        if type(ws) is tuple:
            _, WeightStep = ws
        else:
            WeightStep = ws
        #Quantizing and thresholding dout (don't worry about the long formula)
        dout_part1 = WeightStep*(np.sign(0.5*(((dout - threshold1)*(WeightStep*np.ones(dout.shape)))+(np.absolute((dout - threshold1)*(WeightStep*np.ones(dout.shape)))))))
        dout_part2 = WeightStep*(np.sign(0.5*(((dout + threshold1)*(WeightStep*np.ones(dout.shape)))-(np.absolute((dout + threshold1)*(WeightStep*np.ones(dout.shape)))))))
        dout = dout_part1 + dout_part2
        dx = dout.dot(w.T)


        #Quantizing and thresholding the input
        x_part1 = InputStep*(np.sign(0.5*(((x - threshold2)*(InputStep*np.ones(x.shape)))+(np.absolute((x - threshold2)*(InputStep*np.ones(x.shape)))))))
        x_part2 = InputStep*(np.sign(0.5*(((x + threshold2)*(InputStep*np.ones(x.shape)))-(np.absolute((x + threshold2)*(InputStep*np.ones(x.shape)))))))
        x = x_part1 + x_part2




        dw = x.T.dot(dout)
        db = dout.T.dot(np.ones(N))
       
    else:
        dx = dout.dot(w.T)
        dw = x.T.dot(dout)
        db = dout.T.dot(np.ones(N))
       
    return dx, dw, db    




def __tanh(x, lamb):
    """
    function to calculate tanh() of an array of any size
    """
    out = (2 / (1 + (np.exp(-lamb * x)))) - 1
    return out




def __tanh_prime(x, lamb):
    """
    function to calculate derivative of tanh() function
    """
    out = (1 - __tanh(x, lamb) * __tanh(x, lamb)) * (lamb / 2)
    return out






def tanh_forward(x, lamb = 1.5):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).


    Input:
    - x: Inputs, of any shape


    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = __tanh(x, lamb)
    cache = x, lamb
    return out, cache




def tanh_backward(dout, cache):
    """
    Computes the backward pass for a layer of tanh activation.


    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout & lamb


    Returns:
    - dx: Gradient with respect to x
    """
    x, lamb = cache
    dx = dout*__tanh_prime(x, lamb)
    return dx






def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.


    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C


    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
   
    #LOSS
    N = x.shape[0]
    correct_scores = x[np.arange(N), y]
    loss = np.sum(np.log(np.sum(np.exp(x),axis = 1) + 10**-10) - correct_scores)
    loss/=N
    pass


    #GRADIENT
    sum_exp_scores = np.exp(x).sum(axis=1, keepdims=True)
    softmax_matrix = np.exp(x)/(sum_exp_scores + 10**-10)
    softmax_matrix[np.arange(N), y] -= 1
    dx = softmax_matrix
    dx/=N


    return loss, dx






