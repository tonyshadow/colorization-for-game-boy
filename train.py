from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime

import fnmatch
import tensorflow as tf
import numpy as np
from optparse import OptionParser

import cfg
import net
import feed_dict as fd

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('max_steps', 100000,
                        """The weight decay.""")


def get_feed_dict(batch_size, original_images_placeholder, gray_images_placeholder, image_list):

    original_images, gray_images = fd.get_batch(batch_size, image_list)

    feed_dict = {
        original_images_placeholder: original_images,
        gray_images_placeholder: gray_images
    }

    return feed_dict


# def train_model(checkpoint_dir, image_list, batch_size):
#         # summaries = tf.summary.scalar('learning_rate', 1)
#
#         # summary_op = tf.summary.merge(summaries)
#
#         summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)


def train(checkpoint_dir, image_list):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # # # Get images and labels for CIFAR-10.
        # # # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # # # GPU and resulting in a slow down.
        # with tf.device('/cpu:0'):
        #     images, labels = cifar10.distorted_inputs()

        original_images_placeholder = tf.placeholder(tf.float32, shape=(cfg.BATCH_SIZE, 144, 160, 3))
        gray_images_placeholder = tf.placeholder(tf.float32, shape=(cfg.BATCH_SIZE, 144, 160, 3))

        # image summary for tensorboard
        tf.summary.image('original_images', original_images_placeholder, max_outputs=100)
        tf.summary.image('gray_images', gray_images_placeholder, max_outputs=100)

        # Build a Graph that computes the logits predictions from the inference model.
        logits = net.inference(gray_images_placeholder)

        # Calculate loss.
        loss = net.loss(original_images_placeholder, logits)

        # # Build a Graph that trains the model with one batch of examples and
        # # updates the model parameters.
        # train_op = net.train(loss, global_step)

        # tf.summary.scalar('loss', loss)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss, global_step=global_step)

        # summary for tensorboard graph
        # summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        try:
            os.mkdir(checkpoint_dir)
        except:
            pass

        sess.run(init)
        print("\nRunning session\n")

        # restore previous model if there is one
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir + "training")
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring previous model...")
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored")
            except:
                print("Could not restore model")
                pass

        # Summary op
        # graph_def = sess.graph.as_graph_def(add_shapes=True)
        # summary_writer = tf.summary.FileWriter(checkpoint_dir+"training", graph=sess.graph)

        step = int(sess.run(global_step))
        gstep = step
        while gstep < FLAGS.max_steps:
            start_time = time.time()
            feed_dict = get_feed_dict(cfg.BATCH_SIZE, original_images_placeholder, gray_images_placeholder, image_list)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 1 == 0:
                num_examples_per_step = cfg.BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # if step % 10 == 0:
            #     summary_str = sess.run(summary_op)
            #     summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (gstep + 1) == FLAGS.max_steps:
                print("Saving model")
                saver.save(sess, checkpoint_dir + "training/checkpoint", global_step=global_step)

            step += 1
            gstep = int(sess.run(global_step))

        return


def main(argv=None):
    parser = OptionParser(usage='usage')
    parser.add_option('-c', '--checkpoint_dir',          type='str')
    parser.add_option('-d', '--data_dir', type='str')

    opts, args = parser.parse_args()
    opts = vars(opts)

    checkpoint_dir = opts['checkpoint_dir']
    data_dir = opts['data_dir']

    if checkpoint_dir is None:
        print ("checkpoint_dir is required")
        exit()

    print()
    print('checkpoint_dir: ' + str(checkpoint_dir))
    print('data_dir:       ' + str(data_dir))
    print()

    pattern = "*resized.png"
    image_list = list()
    for d, s, fList in os.walk(data_dir):
        for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
                image_list.append(os.path.join(d, filename))

    print(str(len(image_list)) + ' images...')
    train(checkpoint_dir, image_list)


if __name__ == "__main__":

    if sys.argv[1] == "--help" or sys.argv[1] == "-h" or len(sys.argv) < 2:
        print()
        print("-c --checkpoint_dir <str> [path to save the model]")
        print("-d --data_dir       <str> [path to root image folder]")
        print()
        exit()

    tf.app.run()
