from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import cfg


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv_layer(inputs, kernel_shape, stride, index):
    with tf.variable_scope('conv_%s' % index) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             stddev=5e-2,
                                             wd=cfg.WEIGHT_DECAY)
        conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding='SAME')
        biases = _variable_on_cpu('biases', kernel_shape[3:], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_relu = tf.nn.relu(pre_activation, name=scope.name)

        return conv_relu


def _deconv_layer(inputs, kernel_shape, stride, index):
    batch_size, height, width, in_channel = [int(i) for i in inputs.shape]
    out_channel = kernel_shape[3]
    kernel_size = [kernel_shape[0], kernel_shape[1], kernel_shape[3], kernel_shape[2]]
    output_shape = [batch_size, height * stride, width * stride, out_channel]
    with tf.variable_scope('conv_%s' % index) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_size,
                                             stddev=5e-2,
                                             wd=cfg.WEIGHT_DECAY)
        deconv = tf.nn.conv2d_transpose(inputs, kernel, output_shape, [1, stride, stride, 1], padding='SAME')

        biases = _variable_on_cpu('biases', out_channel, tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(deconv, biases)
        deconv_relu = tf.nn.relu(pre_activation, name=scope.name)

        return deconv_relu


def inference(inputs):
    # encoder
    conv1 = _conv_layer(inputs, [3, 3, 3, 64], 2, 1)
    conv2 = _conv_layer(conv1, [3, 3, 64, 128], 1, 2)
    conv3 = _conv_layer(conv2, [3, 3, 128, 128], 2, 3)
    conv4 = _conv_layer(conv3, [3, 3, 128, 256], 1, 4)
    conv5 = _conv_layer(conv4, [3, 3, 256, 256], 2, 5)
    conv6 = _conv_layer(conv5, [3, 3, 256, 512], 1, 6)
    conv7 = _conv_layer(conv6, [3, 3, 512, 512], 1, 7)
    conv8 = _conv_layer(conv7, [3, 3, 512, 256], 1, 8)

    # decoder
    conv9 = _conv_layer(conv8, [3, 3, 256, 128], 1, 9)
    conv10 = _deconv_layer(conv9, [3, 3, 128, 64], 2, 10)
    conv11 = _conv_layer(conv10, [3, 3, 64, 64], 1, 11)
    conv12 = _conv_layer(conv11, [3, 3, 64, 64], 1, 12)
    conv13 = _deconv_layer(conv12, [3, 3, 64, 32], 2, 13)
    conv14 = _conv_layer(conv13, [3, 3, 32, 32], 1, 14)
    conv15 = _deconv_layer(conv14, [3, 3, 32, 3], 2, 15)
    conv16 = _deconv_layer(conv15, [3, 3, 3, 3], 1, 16)

    return conv16


def loss(logits, labels):
    l2_loss = tf.nn.l2_loss(logits - labels)
    return l2_loss


def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = cfg.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / cfg.BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * cfg.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cfg.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cfg.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdadeltaOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(cfg.MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op
