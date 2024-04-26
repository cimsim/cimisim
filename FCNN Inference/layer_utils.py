from layers import *




def affine_tanh_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a tanh


    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer


    Returns a tuple of:
    - out: Output from the tanh
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, tanh_cache = tanh_forward(a)
    cache = (fc_cache, tanh_cache)
    return out, cache




def affine_tanh_backward(dout, cache, quant_params):
    """
    Backward pass for the affine-tanh convenience layer
    """
    fc_cache, tanh_cache = cache
    da = tanh_backward(dout, tanh_cache)
    dx, dw, db = affine_backward(da, fc_cache, quant_params)
    return dx, dw, db
