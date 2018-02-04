import tensorflow as tf

class Network(object):

    def __init__(self, name, input_shape, output_dim):
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, shape=[None, *input_shape], name='input')
            self.target_q = tf.placeholder(tf.float32, shape=[None, output_dim], name='target_q')

            net = self.input

            with tf.variable_scope('layer1'):
                w = tf.Variable(tf.random_normal(
                                (*input_shape, output_dim),0,0.01), name='w')
                net = tf.matmul(net, w)

            self.q_out = net
            self.loss = tf.reduce_sum(tf.square(self.target_q - self.q_out))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.025)

        var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = self.optimizer.compute_gradients(self.loss, var_list)
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            self.gradients_placeholders.append(
                (tf.placeholder(var.dtype, shape=var.get_shape()), var))

        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)