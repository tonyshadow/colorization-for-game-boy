from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def inference(inputs):
    kwargs = {'padding': 'same',
              'activation': tf.nn.relu,
              'kernel_initializer': tf.truncated_normal_initializer(stddev=0.1),
              'bias_initializer': tf.constant_initializer(0.1)}

    # # encoder
    # conv1 = tf.layers.conv2d(inputs, 64, 3, strides=2, **kwargs)
    # conv2 = tf.layers.conv2d(conv1, 128, 3, **kwargs)
    # conv3 = tf.layers.conv2d(conv2, 128, 3, strides=2, **kwargs)
    # conv4 = tf.layers.conv2d(conv3, 256, 3, **kwargs)
    # conv5 = tf.layers.conv2d(conv4, 256, 3, strides=2, **kwargs)
    # conv6 = tf.layers.conv2d(conv5, 512, 3, **kwargs)
    # conv7 = tf.layers.conv2d(conv6, 512, 3, **kwargs)
    # conv8 = tf.layers.conv2d(conv7, 256, 3, **kwargs)
    #
    # # decoder
    # conv9 = tf.layers.conv2d(conv8, 128, 3, **kwargs)
    # conv10 = tf.image.resize_nearest_neighbor(conv9, [2 * conv9.shape[1], 2 * conv9.shape[2]])
    # conv11 = tf.layers.conv2d(conv10, 64, 3, **kwargs)
    # conv12 = tf.layers.conv2d(conv11, 64, 3, **kwargs)
    # conv13 = tf.image.resize_nearest_neighbor(conv12, [2 * conv12.shape[1], 2 * conv12.shape[2]])
    # conv14 = tf.layers.conv2d(conv13, 32, 3, **kwargs)
    # conv15 = tf.layers.conv2d(conv14, 3, 3, padding='same', activation=tf.nn.tanh,
    #                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                           bias_initializer=tf.constant_initializer(0.1))
    # conv16 = tf.image.resize_nearest_neighbor(conv15, [2 * conv15.shape[1], 2 * conv15.shape[2]])

    # encoder
    conv1 = tf.layers.conv2d(inputs, 32, 3, **kwargs)
    conv2 = tf.layers.conv2d(conv1, 32, 3, **kwargs)
    conv3 = tf.layers.conv2d(conv2, 64, 3, **kwargs)
    conv4 = tf.layers.conv2d(conv3, 64, 3, **kwargs)
    conv5 = tf.layers.conv2d(conv4, 128, 3, **kwargs)
    conv6 = tf.layers.conv2d(conv5, 128, 3, **kwargs)
    conv7 = tf.layers.conv2d(conv6, 256, 3, **kwargs)
    conv8 = tf.layers.conv2d(conv7, 256, 3, **kwargs)

    # decoder
    conv9 = tf.layers.conv2d(conv8, 128, 3, **kwargs)
    conv10 = tf.layers.conv2d(conv9, 128, 3, **kwargs)
    conv11 = tf.layers.conv2d(conv10, 64, 1, **kwargs)
    conv12 = tf.layers.conv2d(conv11, 64, 1, **kwargs)
    conv13 = tf.layers.conv2d(conv12, 32, 1, **kwargs)
    conv14 = tf.layers.conv2d(conv13, 32, 1, **kwargs)
    conv15 = tf.layers.conv2d(conv14, 16, 1, **kwargs)
    conv16 = tf.layers.conv2d(conv15, 16, 1, **kwargs)
    conv17 = tf.layers.conv2d(conv16, 8, 1, **kwargs)
    conv18 = tf.layers.conv2d(conv17, 3, 1, **kwargs)

    return conv18


def loss(logits, labels):
    l2_loss = tf.nn.l2_loss(logits - labels)
    return l2_loss
