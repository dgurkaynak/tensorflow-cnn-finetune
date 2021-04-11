import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import VggNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor


tf.app.flags.DEFINE_string('ckpt', '', 'Checkpoint path; it must end with ".ckpt"')
tf.app.flags.DEFINE_integer('num_classes', 26, 'Number of classes')
tf.app.flags.DEFINE_string('test_file', '../data/val.txt', 'Test dataset file')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Placeholders
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    model = VggNetModel(num_classes=FLAGS.num_classes, dropout_keep_prob=dropout_keep_prob)
    model.inference(x)

    # Accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    test_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.test_file, num_classes=FLAGS.num_classes, output_size=[224, 224])
    test_batches_per_epoch = np.floor(len(test_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, FLAGS.ckpt)

        test_acc = 0.
        test_count = 0

        for _ in range(test_batches_per_epoch):
            batch_tx, batch_ty = test_preprocessor.next_batch(FLAGS.batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, dropout_keep_prob: 1.})
            test_acc += acc
            test_count += 1

        test_acc /= test_count
        print("{} Test Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

if __name__ == '__main__':
    tf.app.run()
