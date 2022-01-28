import yaml
import os
import numpy as np
from evaluate import *
from models import get_model
from tfrecord_load import  load_tfrecord_dataset
import torch

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument('--sf', type=str, default='./save', help='save folder path')
    parser.add_argument('--fn', type=str, default='{name}_{e}.pth', help='format name of weight file')
    parser.add_argument('--e', type=int, default=1, help='epoch start')
    return parser.parse_args()


def main(cfg, save_folder, format_name, start_e=1):
    best_e = None
    max_acc = 0.0
    #getdata
    valid_dataset = load_tfrecord_dataset(cfg['val_data'], cfg['batch_size'],
                          dtype="valid", shuffle=True, buffer_size=1028, transform_img=False)
    #getmodel and load weight

    ckpt_path = os.path.join(save_folder, "{}/ckpt/".format(cfg['model_name']))
    for epoch in range(start_e, cfg['epoch_num']):
        wpath = os.path.join(ckpt_path, format_name.format(name=cfg['model_name'], e=epoch))
        if not os.path.exists(wpath): continue
        print("src: {}\n Evaluate...".format(wpath))
        model = get_model(cfg)
        load_model(model,wpath)
        #valid
        model.eval()
        with torch.no_grad():
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