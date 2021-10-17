from modules.evaluate import *
from modules.preprocess import _transform_images
from modules.models import getModel
from modules.plot import plot_roc

import os
import argparse
import tqdm
import tensorflow as tf
import numpy as np
import yaml
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--c', type=str, help='config path')
    return parser.parse_args()

def get_base_model(isize, backbone_type, name_model, embd_size):
    model = getModel(input_shape=(isize, isize, 3),
                    backbone_type=backbone_type,
                    embd_shape=embd_size,
                    training=False)
    ckpt_path = tf.train.latest_checkpoint('./save/' + name_model)
    if ckpt_path is not None:
        print("\t[*] load ckpt from {}".format(ckpt_path))
        tf.train.Checkpoint(model).restore(ckpt_path).expect_partial()
    else:
        print("\t[*] Cannot find ckpt from {}.".format(ckpt_path))
        return None
    return model
def evaluate(test_path, pair_file, model, name, datatype, isplot=True):
    result = {'issame': [], 'prob':[]}
    with open(os.path.join(pair_file)) as src:
        for line in tqdm.tqdm(src):
            if line == "" or len(line.split(" ")) < 3 :
                continue

            line = line.split(" ")
            img1 = tf.io.read_file(os.path.join(test_path,line[0]))
            img2 = tf.io.read_file(os.path.join(test_path,line[1]))

            img1 = _transform_images()(tf.image.decode_jpeg(img1, channels=3))
            img2 = _transform_images()(tf.image.decode_jpeg(img2, channels=3))

            batch = tf.Variable([img1, img2])
            embds = model(batch)
            embds = l2_norm(embds)

            diff = np.subtract(embds[0], embds[1])
            dist = np.sum(np.square(diff))

            result['issame'].append(int(line[2])==1)
            result['prob'].append(dist)
    
    thresholds = np.arange(0, 4, 0.01)
    dists = np.array(result['prob'])
    actual_issame = np.array(result['issame']) 
    tpr, fpr, acc, best = calculate_roc(thresholds, dists, actual_issame)
    if isplot:
        print("best thres:  {} \t best acc: {}".format(best, np.max(acc)))
        plot_roc(fpr, tpr, 
        "/home/nhuntn/K64/FaceRecognition/save/figures/{}-{}.png".format(name, datatype), name,
        max_acc=np.max(acc), dtype=datatype)
    return np.max(acc)

def validate(config, dtype="", isplot=True):
    val_dic = config["val_data"]
    model = get_base_model(config["image_size"], config["backbone"], 
            config["model_name"], config["embd_size"])
    if model is None:
        return 0
    acc = 0
    if dtype == "":
        for k in val_dic.keys():
            evaluate(test_path=val_dic[k], pair_file=os.path.join(val_dic[k], "../", k + '_pair.txt'),
             model=model, name=config["model_name"], datatype=k, isplot=isplot) 
    else:
        acc = evaluate(test_path=val_dic[dtype], pair_file=os.path.join(val_dic[dtype], "../", dtype + '_pair.txt'),
             model=model, name=config["model_name"], datatype=dtype, isplot=isplot)
    return acc

if __name__ == "__main__":
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.full_load(file)
    validate(config, dtype="lfw")  