import yaml
import os
import numpy as np
from evaluate import *
from models import getModel
from preprocess import  load_tfrecord_dataset
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument('--sf', type=str, default='./save', help='save folder path')
    parser.add_argument('--e', type=int, default=1, help='epoch start')
    return parser.parse_args()

def load_model(model, weight_file):
    model.load_weights(weight_file, by_name=True, skip_mismatch=True)

def main(cfg, save_folder, start_e=1):
    #getdata
    test_dataset = load_tfrecord_dataset(cfg['test_data'], cfg['batch_size'],
                          dtype="test", shuffle=True, buffer_size=1028, transform_img=False)
    ckpt_path = os.path.join(save_folder, "{}/ckpt/".format(cfg['model_name']))
    wpaths = [f.path for f in os.scandir(ckpt_path)]
    max_acc = 0.
    best_e = None
    for path in wpaths[start_e:]:
        print("src: ", path, "\ntesting...")
        
        #getmodel and load weight
        model = getModel(input_shape=(cfg['image_size'], cfg['image_size'], 3),
                     backbone_type=cfg['backbone'],
                     num_classes=cfg['class_num'],
                     head_type=cfg['headtype'],
                     embd_shape=cfg['embd_size'],
                     w_decay=cfg['weight_decay'],
                     training=False, name=cfg['model_name'])
        load_model(model,path)
        #valid
        acc = evaluate_model(model, test_dataset)
        if acc > max_acc:
            best_e = path
            max_acc = acc
    print("best model {} - best acc: {}".format(best_e, max_acc))

if __name__ == '__main__':
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    main(config, save_folder=args.sf, start_e=args.e)