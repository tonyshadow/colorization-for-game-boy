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
import image


def train(model_dir, image_list):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Variables that affect learning rate.
        num_batches_per_epoch = cfg.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / cfg.BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * cfg.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(cfg.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cfg.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create optimizer
        opt = tf.train.AdadeltaOptimizer(lr)

        # Get original images and gray images
        original_images, gray_images = image.read_images(image_list, cfg.BATCH_SIZE)

        # Build a Graph that computes the logits predictions from the inference model.
        logits = net.inference(gray_images)

        # Calculate loss.
        loss = net.loss(original_images, logits)

        # Calculate the gradients
        grads = opt.compute_gradients(loss)

        # Apply gradients
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(cfg.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)

        # Create summary writer
        # summary_writer = tf.summary.FileWriter(checkpoint_dir + "model", sess.graph)

        model_dir = os.path.join(model_dir, 'model')

        # Find previous model and restore it
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring model...")
            try:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored")
            except ValueError:
                print("Can not restore model")

        step = int(sess.run(global_step))
        gstep = step
        while gstep < cfg.MAX_STEPS:
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            step += 1
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
            if step % 1000 == 0 or (gstep + 1) == cfg.MAX_STEPS:
                print("Saving model")
                saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=global_step)

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
    for d, s, fList in os.walk(data_dir + '/images'):
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
