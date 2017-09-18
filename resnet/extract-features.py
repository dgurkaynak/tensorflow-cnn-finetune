import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor


NUM_CLASSES = 109
IMAGES = '../data/pair-images.txt'
BATCH_SIZE = 100
RESNET_DEPTH = 50



def main(_):
    # Placeholders
    x = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
    is_training = tf.placeholder('bool', [])

    # Model
    model = ResNetModel(is_training, depth=RESNET_DEPTH, num_classes=NUM_CLASSES)
    model.inference(x)

    preprocessor = BatchPreprocessor(dataset_file_path=IMAGES, num_classes=NUM_CLASSES, output_size=[224, 224])
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    n_iteration = np.floor(len(preprocessor.labels) / BATCH_SIZE).astype(np.int16)
    print(n_iteration)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights
        # model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, "/home/dgurkaynak/Projects/marvel-finetuning/training/resnet_20170902_094004/checkpoint/model_epoch18.ckpt")

        for n in range(n_iteration):
            print("{} Iteration number: {}".format(datetime.datetime.now(), n+1))
            batch_tx, batch_ty = preprocessor.next_batch(BATCH_SIZE)
            features = sess.run(model.s5, feed_dict={x: batch_tx, is_training: False})
            
            print(features.shape)


if __name__ == '__main__':
    tf.app.run()
