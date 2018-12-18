from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
from datetime import datetime

import fnmatch
import tensorflow as tf
import numpy as np

import cfg
import net
import image


def train(model_dir, image_paths):
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
        original_images, gray_images = image.read_images(image_paths, cfg.BATCH_SIZE)

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
            if step % 100 == 0 or (gstep + 1) == cfg.MAX_STEPS:
                print("Saving model")
                saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=global_step)

            gstep = int(sess.run(global_step))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a colorization model.')
    parser.add_argument('data_dir',
                        help='Directory of the training images, should have two sub-directories: gray/ and images/')
    parser.add_argument('model_dir', default='./', help='Specify the directory to store the model')

    args = parser.parse_args()

    pattern = "*.png"
    image_list = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(args.data_dir, 'images')):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                image_list.append(os.path.join(dirpath, filename))

    print("Found %s images" % len(image_list))

    train(args.model_dir, image_list)
