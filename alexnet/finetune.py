import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import AlexNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor


tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc8,fc7', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at networs input size')
tf.app.flags.DEFINE_string('train_root_dir', '../training', 'Root directory to put the training data')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')

NUM_CLASSES = 26
TRAINING_FILE = '../data/train.txt'
VAL_FILE = '../data/val.txt'
FLAGS = tf.app.flags.FLAGS


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir): os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    dropout_keep_prob = tf.placeholder(tf.float32)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = AlexNetModel(num_classes=NUM_CLASSES, dropout_keep_prob=dropout_keep_prob)
    loss = model.loss(x, y)
    train_op = model.optimize(FLAGS.learning_rate, train_layers)

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver()

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2: 
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    train_preprocessor = BatchPreprocessor(dataset_file_path=TRAINING_FILE, num_classes=NUM_CLASSES,
                                           output_size=[227, 227], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor = BatchPreprocessor(dataset_file_path=VAL_FILE, num_classes=NUM_CLASSES, output_size=[227, 227])

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))

        for epoch in range(FLAGS.num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1

            # Start training
            while step < train_batches_per_epoch:
                batch_xs, batch_ys = train_preprocessor.next_batch(FLAGS.batch_size)
                sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: FLAGS.dropout_keep_prob})

                # Logging
                if step % FLAGS.log_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: 1.})
                    train_writer.add_summary(s, epoch * train_batches_per_epoch + step)

                step += 1

            # Epoch completed, start validation
            print("{} Start validation".format(datetime.datetime.now()))
            test_acc = 0.
            test_count = 0

            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty = val_preprocessor.next_batch(FLAGS.batch_size)
                acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, dropout_keep_prob: 1.})
                test_acc += acc
                test_count += 1

            test_acc /= test_count
            s = tf.Summary(value=[
                tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)
            ])
            val_writer.add_summary(s, epoch+1)
            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

            # Reset the dataset pointers
            val_preprocessor.reset_pointer()
            train_preprocessor.reset_pointer()

            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            #save checkpoint of the model
            checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_path)

            print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))

if __name__ == '__main__':
    tf.app.run()
