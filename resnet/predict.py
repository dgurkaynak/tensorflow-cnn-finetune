import os, sys
import numpy as np
import tensorflow as tf
import cv2
import datetime
from model import ResNetModel
sys.path.insert(0, '../utils')


tf.app.flags.DEFINE_string('ckpt', '', 'Checkpoint path; it must end with ".ckpt"')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_classes', 26, 'Number of classes')
tf.app.flags.DEFINE_string('input_image', '', 'The path of input image')

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Placeholders
    x = tf.placeholder(tf.float32, [1, 224, 224, 3])
    is_training = tf.placeholder('bool', [])

    # Model
    model = ResNetModel(is_training, depth=FLAGS.resnet_depth, num_classes=FLAGS.num_classes)
    model.inference(x)

    saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, FLAGS.ckpt)

        batch_x = np.ndarray([1, 224, 224, 3])

        # Read image and resize it
        img = cv2.imread(FLAGS.input_image)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)

        # Subtract mean color
        img -= np.array([132.2766, 139.6506, 146.9702])

        batch_x[0] = img

        scores = sess.run(model.prob, feed_dict={x: batch_x, is_training: False})
        print(scores)

if __name__ == '__main__':
    tf.app.run()
