import tensorflow as tf
"sourse : https://github.com/peteryuX/arcface-tf2/blob/6447e9d04451aba1ef262f43c77ae50cdfaf19f8/modules/dataset.py"

def _parse_tfrecord(dtype="train", transform_img=True):
    def parse_tfrecord(tfrecord):
        if dtype == "train":
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/filename': tf.io.FixedLenFeature([], tf.string),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
            y_train = tf.cast(x['image/source_id'], tf.int64)

            x_train = _transform_images(transform_img)(x_train)
            y_train = _transform_targets(y_train)
            return x_train, y_train
        else:
            #lfw data
            features = {'image_1': tf.io.FixedLenFeature([], tf.string),
                        'image_2': tf.io.FixedLenFeature([], tf.string),
                        'issame' : tf.io.FixedLenFeature([], tf.int64)}
            x = tf.io.parse_single_example(tfrecord, features)
            img1 = tf.image.decode_jpeg(x['image_1'], channels=3)
            img2 = tf.image.decode_jpeg(x['image_2'], channels=3)
            issame = tf.cast(x['issame'], tf.int64)

            img1 = _transform_images(transform_img)(img1)
            img2 = _transform_images(transform_img)(img2)
            return img1, img2, issame
    return parse_tfrecord


def _transform_images(transform_img):
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (128, 128))
        x_train = tf.image.random_crop(x_train, (112, 112, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        if transform_img:
            x_train = tf.image.random_saturation(x_train, 0.5, 1.4)
            x_train = tf.image.random_brightness(x_train, 0.3)
        x_train = tf.cast(x_train,tf.float32) / 255
        return x_train
    return transform_images


def _transform_targets(y_train):
    return y_train


def load_tfrecord_dataset(tfrecord_path, batch_size,
                          dtype="train", shuffle=True, buffer_size=10240, transform_img=True):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    # raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(dtype=dtype, transform_img=transform_img),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset