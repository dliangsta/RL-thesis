import os
import tensorflow as tf
from functools import reduce
from tensorflow.contrib.layers.python.layers import initializers


class CNN(object):
    """
    Deep Q Network architecture.
    """

    def __init__(self, sess,
                 history_length,
                 observation_dims,
                 output_size,
                 trainable=True,
                 hidden_activation_fn=tf.nn.relu,
                 output_activation_fn=None,
                 weights_initializer=initializers.xavier_initializer(),
                 biases_initializer=tf.constant_initializer(0.1),
                 value_hidden_sizes=[512],
                 advantage_hidden_sizes=[512],
                 name='CNN'):
        self.sess = sess
        self.copy_op = None
        self.name = name
        self.var = {}

        # Inputs.
        self.inputs = tf.placeholder('float32', [None] + observation_dims + [history_length], name='inputs')

        # Normalize inputs to range between 0 and 1.
        self.l0 = tf.div(self.inputs, 255.)

        # Build network.
        with tf.variable_scope(name):
            # Convolutional layers.
            self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(self.l0,
                32, [8, 8], [4, 4], weights_initializer, biases_initializer,
                hidden_activation_fn, name='l1_conv')
            self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
                64, [4, 4], [2, 2], weights_initializer, biases_initializer,
                hidden_activation_fn, name='l2_conv')
            self.l3, self.var['l3_w'], self.var['l3_b'] = conv2d(self.l2,
                64, [3, 3], [1, 1], weights_initializer, biases_initializer,
                hidden_activation_fn, name='l3_conv')

            # Fully connected layer.
            self.l4, self.var['l4_w'], self.var['l4_b'] = linear(self.l3, 
                512, weights_initializer, biases_initializer,
                hidden_activation_fn, name='l4_conv')

            # Output layer.
            self.outputs, self.var['w_out'], self.var['b_out'] = \
                linear(self.l4, output_size, weights_initializer,
                    biases_initializer, output_activation_fn, trainable, name='out')

            # Maxes of output layer.
            self.max_outputs = tf.reduce_max(self.outputs, reduction_indices=1)
            # Index of outputs.
            self.outputs_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            # Indexed outputs.
            self.outputs_with_idx = tf.gather_nd(self.outputs, self.outputs_idx)
            # Actions to take.
            self.actions = tf.argmax(self.outputs, axis=1)

    def run_copy(self):
        """
        Copy prediction network to target network.
        """
        if self.copy_op is None:
            raise Exception("run `create_copy_op` first before copy")
        else:
            self.sess.run(self.copy_op)

    def create_copy_op(self, network):
        """
        Create operation to copy prediction network to target network.
        """
        with tf.variable_scope(self.name):
            copy_ops = []

            for name in self.var.keys():
                copy_op = self.var[name].assign(network.var[name])
                copy_ops.append(copy_op)

            self.copy_op = tf.group(*copy_ops, name='copy_op')

    def eval_actions(self, observation):
        """
        Evaluate the actions to take.
        """
        return self.actions.eval({self.inputs: observation}, session=self.sess)

    def eval_max_outputs(self, observation):
        """
        Evaluate the maxes of the outputs of the network.
        """
        return self.max_outputs.eval({self.inputs: observation}, session=self.sess)

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer,
           activation_fn=tf.nn.relu,
           padding='VALID',
           name='conv2d',
           trainable=True):
    """
    Construct a convolutional layer.
    """
    with tf.variable_scope(name):
        # Stride and kernel shape.
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        # Weights, convolutional layer, bias, output.
        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=weights_initializer, trainable=trainable)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format='NHWC')
        b = tf.get_variable('b', [output_dim], tf.float32, initializer=biases_initializer, trainable=trainable)
        out = tf.nn.bias_add(conv, b)

    if activation_fn != None:
        # Apply activation function.
        out = activation_fn(out)

    return out, w, b


def linear(input_,
           output_size,
           weights_initializer=initializers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer,
           activation_fn=None,
           trainable=True,
           name='linear'):
    """
    Constructs a fully connected layer.
    """
    # Get shape of input.
    shape = input_.get_shape().as_list()

    if len(shape) > 2:
        # Flatten.
        input_ = tf.reshape(
            input_, [-1, reduce(lambda x, y: x * y, shape[1:])])
        shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        # Weights, bias, output.
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, initializer=weights_initializer, trainable=trainable)
        b = tf.get_variable('b', [output_size], initializer=biases_initializer, trainable=trainable)
        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn != None:
            # Apply activation function.
            out = activation_fn(out)

        return out, w, b


def batch_sample(probs, name='batch_sample'):
    """
    Take a random sample from a uniform distribution given some probabilities.
    """
    with tf.variable_scope(name):
        uniform = tf.random_uniform(tf.shape(probs), minval=0, maxval=1)
        samples = tf.argmax(probs - uniform, dimension=1)

    return samples
