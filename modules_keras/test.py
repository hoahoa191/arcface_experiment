from modules_keras.evaluate import *
from modules_keras.models import getModel


import os
import argparse
import tensorflow as tf
import numpy as np
import yaml
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--c', type=str, help='config path')
    parser.add_argument('--wf', type=str, help='weight file path')
    return parser.parse_args()

def get_base_model(isize, backbone_type, embd_size, wfile):
    model = getModel(input_shape=(isize, isize, 3),
                    backbone_type=backbone_type,
                    embd_shape=embd_size,
                    training=False)
    model.load_weights(wfile, by_name=True, skip_mismath=True)
    return model


def validate(config, wfile, dtype="", isplot=True):
    val_dic = config["val_data"]
    model = get_base_model(config["image_size"], 
        config["backbone"], 
        config["model_name"], 
        config["embd_size"],
        wfile=wfile)

    if model is None:
        return 0
    if dtype == "":
        for k in val_dic.keys():
            evaluate(test_path=val_dic[k], 
            pair_file=os.path.join(val_dic[k], "../", k + '_pair.txt'),
            model=model, name=config["model_name"], datatype=k, isplot=isplot) 
    else:
        return evaluate(test_path=val_dic[dtype], 
            pair_file=os.path.join(val_dic[dtype], "../", dtype + '_pair.txt'),
            model=model, name=config["model_name"], datatype=dtype, isplot=isplot)
    return -1

if __name__ == "__main__":
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.full_load(file)
    validate(config, dtype="lfw", wfile=args.wf)  