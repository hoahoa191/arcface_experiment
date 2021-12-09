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