"""
Divisive normalization implementation.

See paper Normalizing the Normalizers: Comparing and Extending Network
Normalization Schemes. Mengye Ren*, Renjie Liao*, Raquel Urtasun, Fabian H.
Sinz, Richard S. Zemel. 2016. https://arxiv.org/abs/1611.04520
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf


def div_norm_2d(
        x,
        sum_window,
        sup_window,
        strides,
        padding,
        gamma=None,
        beta=None,
        layer=None,
        eps=1.0,
        scope=None,
        name="dn_out",
        return_mean=False):
    """Applies divisive normalization on CNN feature maps.
    Collect mean and variances on x on a local window across channels.
    And apply normalization as below:
      x_ = gamma * (x - mean) / sqrt(var + eps) + beta
    https://github.com/renmengye/div-norm/blob/master/div_norm.py

    Args:
      x: Input tensor, [B, H, W, C].
      sum_window: Summation window size, [H_sum, W_sum].
      sup_window: Suppression window size, [H_sup, W_sup].
      gamma: Scaling parameter.
      beta: Bias parameter.
      eps: Denominator bias.
      return_mean: Whether to also return the computed mean.

    Returns:
      normed: Divisive-normalized variable.
      mean: Mean used for normalization (optional).
    """
    if not isinstance(sum_window, list):
        sum_window = list(np.repeat(sum_window, 2))
    if not isinstance(sup_window, list):
        sup_window = list(np.repeat(sup_window, 2))

    if scope is None:
        scope = '%s_%s' % (layer['names'][0], layer['normalization'][0])
    with tf.variable_scope(scope):
        w_sum = tf.ones(sum_window + [1, 1]) / np.prod(np.array(sum_window))
        w_sup = tf.ones(sup_window + [1, 1]) / np.prod(np.array(sup_window))
        x_mean = tf.reduce_mean(x, [3], keep_dims=True)
        x_mean = tf.nn.conv2d(
            x_mean,
            w_sum,
            strides=strides,
            padding=padding)
        normed = x - x_mean
        x2 = tf.square(normed)
        x2_mean = tf.reduce_mean(x2, [3], keep_dims=True)
        x2_mean = tf.nn.conv2d(
            x2_mean,
            w_sup,
            strides=strides,
            padding=padding)
        denom = tf.sqrt(x2_mean + eps)
        normed = normed / denom
        if gamma is None:
            gamma = tf.get_variable(
                name='%s_%s' % ('gamma', scope),
                initializer=1.)
        normed *= gamma
        if beta is not None:
            beta = tf.get_variable(
                name='%s_%s' % ('beta', scope),
                initializer=0.)
        normed += beta
    normed = tf.identity(normed, name='%s_%s' % (scope, name))
    if return_mean:
        return normed, x_mean
    else:
        return normed


def div_norm_1d(
        x,
        sum_window,
        sup_window,
        strides,
        padding,
        gamma=None,
        beta=None,
        layer=None,
        eps=1.0,
        scope='dn',
        name="dn_out",
        return_mean=False):
    """Applies divisive normalization on fully connected layers.
    Collect mean and variances on x on a local window. And apply
    normalization as below:
      x_ = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
      x: Input tensor, [B, D].
      sum_window: Summation window size, W_sum.
      sup_window: Suppression window size, W_sup.
      gamma: Scaling parameter.
      beta: Bias parameter.
      eps: Denominator bias.
      return_mean: Whether to also return the computed mean.

    Returns:
      normed: Divisive-normalized variable.
      mean: Mean used for normalization (optional).
    """

    if scope is None:
        scope = '%s_%s' % (layer['names'][0], layer['normalization'][0])
    with tf.variable_scope(scope):
        x = tf.expand_dims(x, 2)
        w_sum = tf.ones([sum_window, 1, 1], dtype='float') / float(sum_window)
        w_sup = tf.ones([sup_window, 1, 1], dtype='float') / float(sup_window)
        mean = tf.nn.conv1d(x, w_sum, stride=strides, padding=padding)
        x_mean = x - mean
        x2 = tf.square(x_mean)
        var = tf.nn.conv1d(x2, w_sup, stride=strides, padding=padding)
        normed = (x - mean) / tf.sqrt(eps + var)
        normed = tf.squeeze(normed, [2])
        mean = tf.squeeze(mean, [2])
        if gamma is None:
            gamma = tf.get_variable(
                name='%s_%s' % ('gamma', scope),
                initializer=1.)
        normed *= gamma
        if beta is not None:
            beta = tf.get_variable(
                name='%s_%s' % ('beta', scope),
                initializer=0.)
        normed += beta
    normed = tf.identity(normed, name='%s_%s' % (scope, name))
    if return_mean:
        return normed, mean
    else:
        return normed
