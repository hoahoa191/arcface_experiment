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


def CosLoss(margin=0.35, logits_scale=64, num_classes=2):
    def cosloss(y_true, y_pred):
        cos_t = y_pred
        mask = tf.one_hot(tf.reshape(y_true, [-1]), depth=num_classes)
        logits = tf.where(mask == 1., cos_t - margin, cos_t)
        logits = tf.multiply(logits, logits_scale)

        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=logits)
        return tf.reduce_mean(ce)
    return cosloss

def ArcLoss(margin=0.5, logits_scale=64, num_classes=2):
    cos_m = tf.identity(math.cos(margin), name='cos_margin')
    sin_m = tf.identity(math.sin(margin), name='sin_margin')
    th = tf.identity(math.cos(math.pi - margin), name='threshold')
    mm = tf.multiply(sin_m, margin, name='safe_margin')
    eps = tf.keras.backend.epsilon()
    def arcloss(y_true, y_pred):
        cos_t = tf.clip_by_value(y_pred, clip_value_min=-1. + eps, clip_value_max=1. - eps)
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(cos_t * cos_m, sin_t * sin_m, name='cos_mt')
        cos_mt = tf.where(cos_t > th, cos_mt, cos_t - mm)

        mask = tf.one_hot(tf.reshape(y_true, [-1]), depth=num_classes)
        logits = tf.where(mask == 1., cos_mt, cos_t)
        logits = tf.multiply(logits, logits_scale)

        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=logits)
        return tf.reduce_mean(ce)
    return arcloss
