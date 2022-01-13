import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications import  MobileNetV2, ResNet50V2, InceptionResNetV2, ResNet152V2
from .layers import BatchNormalization, CosLayer, ArcLayer 


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def Backbone(backbone_type='ResNet50'):
    """Backbone Model"""
    backbone_type = backbone_type.lower()
    def backbone(x_in):
        if backbone_type == 'resnet50':
            return ResNet50V2(input_shape=x_in.shape[1:], include_top=False)(x_in)
        elif backbone_type == 'mobinet':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False)(x_in)
        elif backbone_type == 'inceptionresnet':
            return InceptionResNetV2(input_shape=x_in.shape[1:], include_top=False)(x_in)
        else:
            raise TypeError('backbone_type error!')
    return backbone


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer

def ArcHead(num_classes, name='ArcHead'):
    """Normalize input and weight before multiplication Head"""
    def arc_head(x_in):
        x = inputs1 = Input(x_in.shape[1:])
        x = CosLayer(num_classes=num_classes)(x)
        return Model((inputs1), x, name=name)((x_in))
    return arc_head

def CosHead(num_classes, name='ArcHead'):
    """Normalize input and weight before multiplication Head"""
    def arc_head(x_in):
        x = inputs1 = Input(x_in.shape[1:])
        x = CosLayer(num_classes=num_classes)(x)
        return Model((inputs1), x, name=name)((x_in))
    return arc_head


def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=_regularizer(w_decay))(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head


def getModel(input_shape=None, num_classes=None, name='', embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, training=False, **kwargs):
    """Arc Face Model"""
    x = inputs = Input(input_shape, name='input_image')
    x = Backbone(backbone_type=backbone_type)(x)
    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    if training:
        assert num_classes is not None
        if head_type == 'ArcHead':
            logist = ArcHead(num_classes=num_classes)(embds)
        else:
            logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
        return Model((inputs), logist, name=name)
    else:
        return Model(inputs, embds, name=name)