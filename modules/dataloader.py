#@title tfrecord
import torch
from torchvision import transforms as T
import tensorflow as tf
"sourse : https://github.com/peteryuX/arcface-tf2/blob/6447e9d04451aba1ef262f43c77ae50cdfaf19f8/modules/dataset.py"

def _parse_tfrecord(binary_img=True):
    def parse_tfrecord(tfrecord):
        if binary_img:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        else:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/img_path': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            image_encoded = tf.io.read_file(x['image/img_path'])
            x_train = tf.image.decode_jpeg(image_encoded, channels=3)

        y_train = tf.cast(x['image/source_id'], tf.int32)

        x_train = _transform_images()(x_train)
        y_train = _transform_targets(y_train)
        return x_train, y_train
    return parse_tfrecord


def _transform_images():
    def transform_images(x_train):
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = x_train.numpy()
        x_train = torch.from_numpy(x_train)
        x_train = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_train)
        return x_train
    return transform_images


def _transform_targets(y_train):
    return torch.from_numpy(y_train.numpy())


def load_tfrecord_dataset(tfrecord_path, batch_size,
                          binary_img=True, shuffle=True, buffer_size=10240):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    #raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(binary_img=binary_img),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset 
