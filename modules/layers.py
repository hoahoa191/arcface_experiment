import tensorflow as tf
import math


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class ArcLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

        self.cos_m = tf.identity(math.cos(self.margin), name='cos_margin')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_margin')
        self.mm = tf.multiply(self.sin_m, self.margin, name='safe_margin')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='threshold')

    def build(self, input_shape):
        self.w = self.add_weight(shape=[input_shape[-1], self.num_classes],
                                      dtype=tf.float32,
                                      initializer=tf.keras.initializers.GlorotUniform(),
                                      regularizer=tf.keras.regularizers.L2(5e-4),
                                      trainable=True,
                                      name='arc_kernel')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')
        cos_t = tf.matmul(normed_embds, normed_w)
        sin_t = tf.sqrt(1. - cos_t ** 2)
        cos_mt = cos_t * self.cos_m - sin_t * self.sin_m #theta + margin
        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm) #when theta + margin > 180

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        #mask = tf.one_hot(tf.reshape(labels, [-1]), depth=self.num_classes)
        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists *= self.logist_scale
        return logists

class CosLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(CosLayer, self).__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.w = self.add_weight(shape=[input_shape[-1], self.num_classes],
                                      dtype=tf.float32,
                                      initializer=tf.keras.initializers.GlorotUniform(),
                                      regularizer=tf.keras.regularizers.L2(5e-4),
                                      trainable=True,
                                      name='kernel')

    def call(self, embds):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        return cos_t