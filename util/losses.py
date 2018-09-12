import tensorflow as tf
import numpy as np


def sigmoid_focal_loss(_sentinel=None, labels=None, logits=None):
    assert (_sentinel is None and labels is not None and logits is not None)
    # gamma == 2 in the paper http://arxiv.org/abs/1708.02002
    prob_error = tf.square(labels - tf.nn.sigmoid(logits))
    loss = prob_error * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return loss


def sparse_softmax_ignore_neg_label(_sentinel=None, labels=None, logits=None):
    assert (_sentinel is None and labels is not None and logits is not None)
    raw_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.maximum(labels, 0))
    mask = tf.cast(tf.greater_equal(labels, 0), tf.float32)
    loss = tf.reduce_sum(raw_losses*mask) / tf.maximum(tf.reduce_sum(mask), 1)
    return loss


def tf_argmax_with_neg(data, axis=None):
    argmax = tf.argmax(data, axis=axis)
    neg_mask = tf.reduce_max(data, axis=axis) >= 0
    argmax_with_neg = tf.where(neg_mask, argmax, -tf.ones_like(argmax))
    return argmax_with_neg


def np_argmax_with_neg(data, axis=None):
    argmax = np.argmax(data, axis=axis)
    neg_mask = np.max(data, axis=axis) >= 0
    argmax_with_neg = np.where(neg_mask, argmax, -np.ones_like(argmax))
    return argmax_with_neg


class SharpenLossScaler:
    def __init__(self, cfg):
        scaling_type = cfg.TRAIN.SHARPEN_LOSS_SCALING_TYPE
        if scaling_type == 'warmup_scaling':
            self.warmup_begin_iter = cfg.TRAIN.SHARPEN_SCHEDULE_BEGIN
            self.warmup_end_iter = cfg.TRAIN.SHARPEN_SCHEDULE_END
        elif scaling_type == 'func_scaling':
            self.scaling_func = eval(cfg.TRAIN.SHARPEN_LOSS_SCALING_FUNC)
            assert callable(self.scaling_func)
        else:
            raise ValueError('Unknown scaling_type {}'.format(scaling_type))
        self.scaling_type = scaling_type

    def __call__(self, n_iter):
        if self.scaling_type == 'warmup_scaling':
            return warmup_scaling(
                n_iter, self.warmup_begin_iter, self.warmup_end_iter)
        else:
            return self.scaling_func(n_iter)


def warmup_scaling(n_iter, begin_iter, end_iter):
    if n_iter >= end_iter:
        return 1.
    elif n_iter < begin_iter:
        return 0.

    return (n_iter - begin_iter) * 1. / (end_iter - begin_iter)
