import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import VggNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor


BATCH_SIZE = 100
NUM_CLASSES = 109
IMAGES_POS = '../data/pairs-pos.txt'
IMAGES_NEG = '../data/pairs-neg.txt'


def main(_):
    # Placeholders
    x = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    model = VggNetModel(num_classes=NUM_CLASSES, dropout_keep_prob=dropout_keep_prob)
    model.inference(x)

    saver = tf.train.Saver()
    preprocessor_pos = BatchPreprocessor(dataset_file_path=IMAGES_POS, num_classes=NUM_CLASSES, output_size=[224, 224])
    preprocessor_neg = BatchPreprocessor(dataset_file_path=IMAGES_NEG, num_classes=NUM_CLASSES, output_size=[224, 224])

    # Get the number of training/validation steps per epoch
    n_iteration_pos = np.ceil(len(preprocessor_pos.labels) / float(BATCH_SIZE)).astype(np.int16)
    n_iteration_neg = np.ceil(len(preprocessor_neg.labels) / float(BATCH_SIZE)).astype(np.int16)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights
        # model.load_original_weights(sess)

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, "/home/dgurkaynak/Projects/marvel-finetuning/training/vggnet_20170920_024745/checkpoint/model_epoch19.ckpt")

        # Positive pairs
        pos_deep_representations = None
        for epoch in range(n_iteration_pos):
            print("{} Iteration POS: {}".format(datetime.datetime.now(), epoch+1))

            batch_tx, batch_ty = preprocessor_pos.next_batch(BATCH_SIZE)
            deep_representation = sess.run(model.fc7l, feed_dict={x: batch_tx, dropout_keep_prob: 1.})
            
            if pos_deep_representations is None:
                pos_deep_representations = deep_representation
            else:              
                pos_deep_representations = np.concatenate((pos_deep_representations, deep_representation))

        print("Done positive pairs {}".format(pos_deep_representations.shape))

        # Negative pairs
        neg_deep_representations = None
        for epoch in range(n_iteration_neg):
            print("{} Iteration NEG: {}".format(datetime.datetime.now(), epoch+1))

            batch_tx, batch_ty = preprocessor_neg.next_batch(BATCH_SIZE)
            deep_representation = sess.run(model.fc7l, feed_dict={x: batch_tx, dropout_keep_prob: 1.})
                        
            if neg_deep_representations is None:
                neg_deep_representations = deep_representation
            else:              
                neg_deep_representations = np.concatenate((neg_deep_representations, deep_representation))

        print("Done negative pairs {}".format(neg_deep_representations.shape))

        all_deep_representations = np.concatenate((pos_deep_representations, neg_deep_representations))
        print("All pairs {}".format(all_deep_representations.shape))

        np.save("all_deep_representations.npy", all_deep_representations);
        print("Saved to all_deep_representations.npy")

if __name__ == '__main__':
    tf.app.run()
