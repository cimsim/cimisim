import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.layers.ops import core as core_ops

class Conv2D_quant(tf.keras.layers.Layer):
    '''
    The class which implements the thresholding of the input in the gradient calculation, for convolutional layers.

    Parameters:
    - th: Threshold value for quantization.
    - filters: Number of filters.
    - kernel_size: Size of the convolutional kernel.
    - padding: Padding type ('valid' or 'same').
    - activation: Activation function.
    - input_shape: Shape of the input (optional).
    - dv: Whether to add device variations.
    - std_dev: Standard deviation for device variations.
    '''
    def __init__(self, th=0.1, filters=16, kernel_size=(3, 3), padding="valid", activation='relu', input_shape=None, dv=False, std_dev=0.06):
        super(Conv2D_quant, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.input_shape1 = input_shape
        self.th = th
        self.dv = dv
        self.std_dev = std_dev

    def build(self, input_shape):
        if self.dv:
            if self.input_shape1:
                self.linear_1 = Conv2D_dv(filters=self.filters, kernel_size=self.kernel_size, std_dev=self.std_dev, padding=self.padding, input_shape=self.input_shape1)
            else:
                self.linear_1 = Conv2D_dv(filters=self.filters, kernel_size=self.kernel_size, std_dev=self.std_dev, padding=self.padding, input_shape=[int(input_shape[-1])])
        else:
            i_s = self.input_shape1 if self.input_shape1 is not None else [int(input_shape[-1])]
            self.linear_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding, input_shape=i_s)
        self.activation = tf.keras.layers.Activation(self.activation)

    @tf.custom_gradient
    def call(self, x):        
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            th = self.th
            x_th = x + tf.stop_gradient(0.5 * (tf.math.sign(x - th) + tf.math.sign(x + th)) - x)
            result = self.linear_1(x_th) + tf.stop_gradient(self.linear_1(x) - self.linear_1(x_th))
            y = self.activation(result)

            def backward(dy, variables=None):
                grad = tape1.gradient(y, [x_th] + self.trainable_weights, output_gradients=dy)
                return (grad[0], grad[1:])
        return y, backward

class Dense_quant(tf.keras.layers.Layer):
    '''
    Fully connected (dense) layer with quantization.

    Parameters:
    - th: Threshold value for quantization.
    - num_output: Number of output units.
    - activation: Activation function.
    - input_shape: Shape of the input (optional).
    - dv: Whether to add device variations.
    - std_dev: Standard deviation for device variations.
    '''
    def __init__(self, th, num_output, activation='relu', input_shape=None, dv=False, std_dev=0.06):
        super(Dense_quant, self).__init__()
        self.num_output = num_output
        self.activation = activation
        self.input_shape1 = input_shape
        self.th = th
        self.dv = dv
        self.std_dev = std_dev

    def build(self, input_shape):
        if not self.dv:
            if self.input_shape1:
                self.linear_1 = tf.keras.layers.Dense(self.num_output, input_shape=self.input_shape1)
            else:
                self.linear_1 = tf.keras.layers.Dense(self.num_output, input_shape=[int(input_shape[-1])])
        else:
            if self.input_shape1:
                self.linear_1 = Dense_dv(self.num_output, input_shape=self.input_shape1, std_dev=self.std_dev)
            else:
                self.linear_1 = Dense_dv(self.num_output, input_shape=[int(input_shape[-1])], std_dev=self.std_dev)
        self.activation = tf.keras.layers.Activation(self.activation)

    @tf.custom_gradient
    def call(self, x):        
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            th = self.th
            x_th = x + tf.stop_gradient(0.5 * (tf.math.sign(x - th) + tf.math.sign(x + th)) - x)
            result = self.linear_1(x_th) + tf.stop_gradient(self.linear_1(x) - self.linear_1(x_th))
            y = self.activation(result)

            def backward(dy, variables=None):
                grad = tape1.gradient(y, [x_th] + self.trainable_weights, output_gradients=dy)
                return (grad[0], grad[1:])
        return y, backward

class Conv2D_dv(tf.keras.layers.Conv2D):
    '''
    Adds device variations to the convolutional network by keeping a fixed set of random constants which 
    are multiplied to the kernel at every step of the forward computation. These multiplications are not
    accounted for when calculating the gradients. Inherits the Conv2D class from the Keras framework and 
    modifies some key functions.
    
    Parameters:
    - filters: Number of filters.
    - kernel_size: Size of the convolutional kernel.
    - std_dev: Standard deviation for device variations.
    - strides: Stride for the convolution operation.
    - padding: Padding type ('valid' or 'same').
    - data_format: Data format ('channels_last' or 'channels_first').
    - dilation_rate: Dilation rate for the convolution.
    - groups: Number of groups for grouped convolution.
    - activation: Activation function.
    - use_bias: Whether to use bias in the convolution.
    - kernel_initializer: Initializer for the kernel weights.
    - bias_initializer: Initializer for the bias.
    - kernel_regularizer: Regularizer for the kernel weights.
    - bias_regularizer: Regularizer for the bias.
    - activity_regularizer: Regularizer for the layer's output.
    - kernel_constraint: Constraint for the kernel weights.
    - bias_constraint: Constraint for the bias.
    '''
    def __init__(self, filters, kernel_size, std_dev=0.06, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        super(Conv2D_dv, self).__init__(filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)
        self.std_dev = std_dev

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel_dv = tf.constant(tf.random.normal(self.kernel.shape, 1, self.std_dev))
        self.bias_dv = tf.constant(tf.random.normal(self.bias.shape, 1, self.std_dev))

    def call(self, inputs):
        if self._is_causal:
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        kernel_ = self.kernel + tf.stop_gradient(tf.multiply(self.kernel, self.kernel_dv) - self.kernel)
        bias_ = self.bias + tf.stop_gradient(tf.multiply(self.bias, self.bias_dv) - self.bias)
        outputs = self._convolution_op(inputs, kernel_)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                bias = array_ops.reshape(bias_, (1, self.filters, 1))
                outputs += bias
            else:
                if output_rank is not None and output_rank > 2 + self.rank:
                    def _apply_fn(o):
                        return nn.bias_add(o, bias_, data_format=self._tf_data_format)

                    outputs = nn_ops.squeeze_batch_dims(outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(outputs, bias_, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

class Dense_dv(tf.keras.layers.Dense):
    '''
    Adds device variations to dense (fully connected) layers in a neural network. 
    It multiplies a fixed set of random constants to the kernel weights at every forward computation,
    without accounting for these multiplications in gradient calculations.
    
    Parameters:
    - units: Number of output units.
    - std_dev: Standard deviation for device variations.
    - activation: Activation function.
    - use_bias: Whether to use bias in the dense layer.
    - kernel_initializer: Initializer for the kernel weights.
    - bias_initializer: Initializer for the bias.
    - kernel_regularizer: Regularizer for the kernel weights.
    - bias_regularizer: Regularizer for the bias.
    - activity_regularizer: Regularizer for the layer's output.
    - kernel_constraint: Constraint for the kernel weights.
    - bias_constraint: Constraint for the bias.
    '''
    def __init__(self, units, std_dev=0.06, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        super(Dense_dv, self).__init__(units, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)
        self.std_dev = std_dev

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel_dv = tf.constant(tf.random.normal(self.kernel.shape, 1, self.std_dev))
        self.bias_dv = tf.constant(tf.random.normal(self.bias.shape, 1, self.std_dev))

    def call(self, inputs):
        kernel = self.kernel + tf.stop_gradient(tf.multiply(self.kernel, self.kernel_dv) - self.kernel)
        bias = self.bias + tf.stop_gradient(tf.multiply(self.bias, self.bias_dv) - self.bias)
        return core_ops.dense(inputs, kernel, bias, self.activation, dtype=self._compute_dtype_object)
