import os
import tqdm
import glob
import random
import tensorflow as tf

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Convert Folder of Images to dataset and save in binary tfrecord format ')

    parser.add_argument('--dataset_path', type=str, default='/home/nhuntn/K64/face_emore/images', help='images folder path')
    parser.add_argument('--output_path', type=str, default='./5kdataset.tfrecord', help='dir for save a record')

    return parser.parse_args()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id, filename):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(dataset_path, output_path):
    if not os.path.isdir(dataset_path):
        print('Please define valid dataset path.')
    else:
        print('Loading {}'.format(dataset_path))

    samples = []
    print('Reading data list...')

    ids = os.listdir(dataset_path)
    num_sample = 0
    for i, id_name in enumerate(ids) :
        img_paths = glob.glob(os.path.join(dataset_path, id_name, '*.jpg'))
        for img_path in img_paths:
            filename = os.path.join(id_name, os.path.basename(img_path))
            samples.append((img_path, id_name, filename))
            num_sample += 1

#        if i + 1 ==5000 :
#            break 
    print("Number of sample : {} imgs - {} ids".format(num_sample, ids))

    random.shuffle(samples)

    print('Writing tfrecord file...')
    with tf.io.TFRecordWriter(output_path) as writer:
        for img_path, id_name, filename in tqdm.tqdm(samples):
            tf_example = make_example(img_str=open(img_path, 'rb').read(),
                                      source_id=int(id_name),
                                      filename=str.encode(filename)) #read binary

            writer.write(tf_example.SerializeToString())

if __name__ == "__main__":
    args = get_args()
    main(args.dataset_path, args.output_path)