import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import VggNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score


BATCH_SIZE = 100
NUM_CLASSES = 109
VESSEL_TYPE = "general-cargo"
IMAGES = '../data/recognition/{}.txt'.format(VESSEL_TYPE)


def main(_):
    # Placeholders
    x = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    model = VggNetModel(num_classes=NUM_CLASSES, dropout_keep_prob=dropout_keep_prob)
    model.inference(x)

    saver = tf.train.Saver()
    preprocessor = BatchPreprocessor(dataset_file_path=IMAGES, num_classes=NUM_CLASSES, output_size=[224, 224], disable_one_hot_label=True)

    # Get the number of training/validation steps per epoch
    n_iteration = np.ceil(len(preprocessor.labels) / float(BATCH_SIZE)).astype(np.int16)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights
        # model.load_original_weights(sess)

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, "/home/dgurkaynak/Projects/marvel-finetuning/training/vggnet_20170920_024745/checkpoint/model_epoch19.ckpt")

        # Positive pairs
        deep_representations = None
        label_y = None
        for epoch in range(n_iteration):
            print("{} Iteration: {}".format(datetime.datetime.now(), epoch+1))

            batch_tx, batch_ty = preprocessor.next_batch(BATCH_SIZE)
            deep_representation = sess.run(model.fc7l, feed_dict={x: batch_tx, dropout_keep_prob: 1.})

            if deep_representations is None:
                deep_representations = deep_representation
            else:              
                deep_representations = np.concatenate((deep_representations, deep_representation))

            if label_y is None:
                label_y = batch_ty
            else:              
                label_y = np.concatenate((label_y, batch_ty))


        print("Done. CNN codes: {} Labels: {}".format(deep_representations.shape, label_y.shape))

        print("Applying PCA...")
        pca = PCA(n_components=100, whiten=True)
        deep_representations_pca = pca.fit_transform(deep_representations)

        print("Done PCA {}".format(deep_representations_pca.shape))

        print("{} Fitting SVM...".format(datetime.datetime.now()))
        clf = svm.SVC()

        # Standart method
        # clf.fit(deep_representations_pca, label_y)

        # print("{} Fitted, calculating score...".format(datetime.datetime.now()))
        # score = clf.score(deep_representations_pca, label_y)
        # print(score)

        # Cross validation
        scores = cross_val_score(clf, deep_representations_pca, label_y, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    tf.app.run()
