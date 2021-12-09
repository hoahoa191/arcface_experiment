import tensorflow as tf
import math

def SoftmaxLoss():
    """softmax loss"""
    def softmax_loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)
        return tf.reduce_mean(ce)
    return softmax_loss


def CosLoss(margin=0.4, logist_scale=64, num_classes=2):
    def cosloss(y_true, y_pred):
        cos_t = y_pred
        mask = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        logists = tf.where(mask == 1., cos_t - margin, cos_t)
        logists = tf.multiply(logists, logist_scale)
        
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)
        return tf.reduce_mean(ce)
    return cosloss

def ArcLoss(margin=0.5, logist_scale=64, num_classes=2):
    cos_m = tf.identity(math.cos(margin), name='cos_margin')
    sin_m = tf.identity(math.sin(margin), name='sin_margin')
    th = tf.identity(math.cos(math.pi - margin), name='threshold')
    mm = tf.multiply(sin_m, margin, name='safe_margin')

    def arcloss(y_true, y_pred):
        cos_t = y_pred
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(cos_t * cos_m, sin_t * sin_m, name='cos_mt')
        cos_mt = tf.where(cos_t > th, cos_mt, cos_t - mm)

        mask = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, logist_scale)
        
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)
        return tf.reduce_mean(ce)
    return arcloss
