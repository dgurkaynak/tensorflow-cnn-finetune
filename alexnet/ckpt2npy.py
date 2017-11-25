import tensorflow as tf
import numpy as np
import sys
from model import AlexNetModel


# Edit just these
FILE_PATH = '/Users/dgurkaynak/Projects/marvel-finetuning/training/alexnet_20171125_124517/checkpoint/model_epoch7.ckpt'
NUM_CLASSES = 26
OUTPUT_FILE = 'alexnet_20171125_124517_epoch7.npy'


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [128, 227, 227, 3])
    model = AlexNetModel(num_classes=NUM_CLASSES)
    model.inference(x)

    saver = tf.train.Saver()
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    data = {
        'conv1': [],
        'conv2': [],
        'conv3': [],
        'conv4': [],
        'conv5': [],
        'fc6': [],
        'fc7': [],
        'fc8': []
    }

    with tf.Session() as sess:
        saver.restore(sess, FILE_PATH)

        for op_name in layers:
          with tf.variable_scope(op_name, reuse = True):
            biases_variable = tf.get_variable('biases')
            weights_variable = tf.get_variable('weights')
            data[op_name].append(sess.run(biases_variable))
            data[op_name].append(sess.run(weights_variable))

        np.save(OUTPUT_FILE, data)

