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
    global_step = tf.train.get_or_create_global_step()

    # Create gray images and original images placeholders
    gray_images = tf.placeholder(tf.float32, shape=(cfg.BATCH_SIZE, 144, 160, 3))
    original_images = tf.placeholder(tf.float32, shape=(cfg.BATCH_SIZE, 144, 160, 3))

    # Build a Graph that computes the logits predictions from the inference model.
    logits = net.inference(gray_images)

    # Calculate loss
    loss = net.loss(original_images, logits)

    # Optimization
    train_op = tf.train.AdadeltaOptimizer(learning_rate=1e-5).minimize(loss, global_step=global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph
    with tf.Session() as sess:
        sess.run(init)

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

        step = tf.train.global_step(sess, global_step)
        while step < cfg.MAX_STEPS:
            start_time = time.time()
            original, gray = image.read_images(image_paths, cfg.BATCH_SIZE)
            _, loss_value = sess.run([train_op, loss], feed_dict={gray_images: gray, original_images: original})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            step += 1
            if step % 1 == 0:
                examples_per_sec = cfg.BATCH_SIZE / duration
                sec_per_batch = duration

                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step % 100 == 0 or step == cfg.MAX_STEPS:
                print("Saving model")
                saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=global_step)


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
