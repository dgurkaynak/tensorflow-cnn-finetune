import os, sys
import numpy as np
import tensorflow as tf
import cv2
import datetime
from model import AlexNetModel
sys.path.insert(0, '../utils')


tf.app.flags.DEFINE_string('ckpt', '', 'Checkpoint path; it must end with ".ckpt"')
tf.app.flags.DEFINE_integer('num_classes', 26, 'Number of classes')
tf.app.flags.DEFINE_string('input_image', '', 'The path of input image')

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Placeholders
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    model = AlexNetModel(num_classes=FLAGS.num_classes, dropout_keep_prob=dropout_keep_prob)
    model.inference(x)

    saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, FLAGS.ckpt)

        batch_x = np.ndarray([1, 227, 227, 3])

        # Read image and resize it
        img = cv2.imread(FLAGS.input_image)
        img = cv2.resize(img, (227, 227))
        img = img.astype(np.float32)

        # Subtract mean color
        img -= np.array([132.2766, 139.6506, 146.9702])

        batch_x[0] = img

        scores = sess.run(model.score, feed_dict={x: batch_x, dropout_keep_prob: 1.})
        print(scores)

if __name__ == '__main__':
    tf.app.run()
