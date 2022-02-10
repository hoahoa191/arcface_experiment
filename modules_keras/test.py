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
    parser.add_argument('--fn', type=str, default='w.{e:02d}.hdf5', help='format name of weight file')
    parser.add_argument('--e', type=int, default=1, help='epoch start')
    return parser.parse_args()

def load_model(model, weight_file):
    model.load_weights(weight_file, by_name=True, skip_mismatch=True)

def main(cfg, save_folder, format_name, start_e=1):
    #getdata
    valid_dataset = load_tfrecord_dataset(cfg['val_data'], cfg['batch_size'],
                          dtype="valid", shuffle=True, buffer_size=1028, transform_img=False)
    ckpt_path = os.path.join(save_folder, "{}/ckpt/".format(cfg['model_name']))
    max_acc = 0.
    best_e = None
    for epoch in range(start_e, cfg['epoch_num']):
        wpath = os.path.join(ckpt_path, format_name.format(name=cfg['model_name'], e=epoch))
        print("src: ", wpath, "\ntesting...")
        if not os.path.exists(wpath): continue
        #getmodel and load weight
        model = getModel(input_shape=(cfg['image_size'], cfg['image_size'], 3),
                     backbone_type=cfg['backbone'],
                     num_classes=cfg['class_num'],
                     head_type=cfg['headtype'],
                     embd_shape=cfg['embd_size'],
                     w_decay=cfg['weight_decay'],
                     training=False, name=cfg['model_name'])
        load_model(model,wpath)
        #valid
        acc = evaluate_model(model, valid_dataset)
        if acc > max_acc:
            best_e = epoch
            max_acc = acc
    print("best model {} - best acc: {}".format(best_e, max_acc))

if __name__ == '__main__':
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    main(config, save_folder=args.sf, format_name=args.fn, start_e=args.e)