"""
Derived from: https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
"""
import tensorflow as tf
import numpy as np


class VggNetModel(object):

    def __init__(self, num_classes=1000, dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob

    def inference(self, x, training=False):
        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope.name)

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope.name)

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope.name)

        # pool2
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope.name)

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope.name)

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope.name)

        # pool3
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope.name)

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope.name)

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope.name)

        # pool4
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope.name)

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope.name)

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope.name)

        # pool5
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        # fc6
        with tf.variable_scope('fc6') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            fc6w = tf.get_variable('weights', initializer=tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1))
            fc6b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32))
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            fc6 = tf.nn.relu(fc6l)

            if training:
                fc6 = tf.nn.dropout(fc6, self.dropout_keep_prob)

        # fc7
        with tf.variable_scope('fc7') as scope:
            fc7w = tf.get_variable('weights', initializer=tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1))
            fc7b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32))
            fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
            fc7 = tf.nn.relu(fc7l)

            if training:
                fc7 = tf.nn.dropout(fc7, self.dropout_keep_prob)

        # fc8
        with tf.variable_scope('fc8') as scope:
            fc8w = tf.get_variable('weights', initializer=tf.truncated_normal([4096, self.num_classes], dtype=tf.float32, stddev=1e-1))
            fc8b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[self.num_classes], dtype=tf.float32))
            self.score = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)

        return self.score

    def loss(self, batch_x, batch_y=None):
        y_predict = self.inference(batch_x, training=True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
        return self.loss

    def optimize(self, learning_rate, train_layers=[]):
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=var_list)

    def load_original_weights(self, session, skip_layers=[]):
        weights = np.load('vgg16_weights.npz')
        keys = sorted(weights.keys())

        for i, name in enumerate(keys):
            parts = name.split('_')
            layer = '_'.join(parts[:-1])

            # if layer in skip_layers:
            #     continue

            if layer == 'fc8' and self.num_classes != 1000:
                continue

            with tf.variable_scope(layer, reuse=True):
                if parts[-1] == 'W':
                    var = tf.get_variable('weights')
                    session.run(var.assign(weights[name]))
                elif parts[-1] == 'b':
                    var = tf.get_variable('biases')
                    session.run(var.assign(weights[name]))
