import tensorflow as tf
import numpy as np


class CustomNetModel(object):

    def __init__(self, num_classes=1000, dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob

    def inference(self, x, training=False):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(x, 5, 5, 64, 1, 1, name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, name='pool1')
        norm1 = lrn(pool1, 1, 0.001 / 9.0, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 64, 1, 1, name='conv2')
        norm2 = lrn(conv2, 4, 0.001 / 9.0, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, name ='pool2')

        batch_size = int(x.get_shape()[-1])
        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        flattened = tf.reshape(pool2, [-1, dim])
        fc3 = fc(flattened, dim, 384, name='fc3')
        if training:
            fc3 = dropout(fc3, self.dropout_keep_prob)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc4 = fc(fc3, 384, 192, name='fc4')
        if training:
            fc4 = dropout(fc4, self.dropout_keep_prob)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.score = fc(fc4, 192, self.num_classes, relu=False, name='fc5')
        return self.score

    def loss(self, batch_x, batch_y=None):
        y_predict = self.inference(batch_x, training=True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
        return self.loss

    def optimize(self, learning_rate, train_layers=[]):
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=var_list)

    def load_original_weights(self, session, skip_layers=[]):
        print('Loading weights are not supported')


"""
Helper methods
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name=scope.name)
        return relu

def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out])
        biases = tf.get_variable('biases', [num_out])
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides = [1, stride_y, stride_x, 1],
                          padding = padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
