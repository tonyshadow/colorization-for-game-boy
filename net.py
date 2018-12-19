from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def inference(inputs):
    # # encoder
    # conv1 = _conv_layer(inputs, [3, 3, 3, 64], 2, 1)
    # conv2 = _conv_layer(conv1, [3, 3, 64, 128], 1, 2)
    # conv3 = _conv_layer(conv2, [3, 3, 128, 128], 2, 3)
    # conv4 = _conv_layer(conv3, [3, 3, 128, 256], 1, 4)
    # conv5 = _conv_layer(conv4, [3, 3, 256, 256], 2, 5)
    # conv6 = _conv_layer(conv5, [3, 3, 256, 512], 1, 6)
    # conv7 = _conv_layer(conv6, [3, 3, 512, 512], 1, 7)
    # conv8 = _conv_layer(conv7, [3, 3, 512, 256], 1, 8)

    # # decoder
    # conv9 = _conv_layer(conv8, [3, 3, 256, 128], 1, 9)
    # conv10 = _deconv_layer(conv9, [3, 3, 128, 64], 2, 10)
    # conv11 = _conv_layer(conv10, [3, 3, 64, 64], 1, 11)
    # conv12 = _conv_layer(conv11, [3, 3, 64, 64], 1, 12)
    # conv13 = _deconv_layer(conv12, [3, 3, 64, 32], 2, 13)
    # conv14 = _conv_layer(conv13, [3, 3, 32, 32], 1, 14)
    # conv15 = _deconv_layer(conv14, [3, 3, 32, 3], 2, 15)
    # conv16 = _deconv_layer(conv15, [3, 3, 3, 3], 1, 16)

    kwargs = {'padding': 'same',
              'activation': tf.nn.leaky_relu,
              'kernel_initializer': tf.truncated_normal_initializer(stddev=0.1),
              'bias_initializer': tf.constant_initializer(0.1)}
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
