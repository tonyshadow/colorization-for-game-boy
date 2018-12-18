from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import cv2
import tensorflow as tf

import net


def test(model_dir, image_dir):
    with tf.Graph().as_default():
        image = cv2.imread(image_dir)
        image = cv2.resize(image, (160, 144)).astype('float')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = net.inference(image)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            # Find previous model and restore it
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restoring model...")
                try:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("Model restored")
                except ValueError:
                    print("Can not restore model")

        output = sess.run([logits])[0]
        dir_out = image[:image.index('.png')]+'_output.png'
        print ('Store output at '+dir_out)
        cv2.imwrite(dir_out, output[0, :, :, :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a colorization model.')
    parser.add_argument('model_dir', help='The model directory')
    parser.add_argument('image', help='Path to the testing image file')

    args = parser.parse_args()

    test(args.model_dir, args.image)
