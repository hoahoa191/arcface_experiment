import yaml
import os
import numpy as np
from modules.evaluate import *
from modules.models import get_model
from modules.dataloader import *
import torch

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument('--s', type=str, default='./save', help='save folder path')
    parser.add_argument('--t', type=str,  help='lfw folder contains pair file')
    parser.add_argument('--e', type=int, default=1, help='epoch start')
    return parser.parse_args()


def main(cfg, save_folder, test_file, start_e=1, n_workers=2):
    best_e = None
    best_t = None
    max_acc = 0.0
    #getdata
    test_dataset = LFWdataset(data_list_file=os.path.join(test_file, "lfw_pair.txt").replace("\\", "/"))
    testloader = get_DataLoader(test_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=True,
                                num_workers=n_workers)
    #getmodel and load weight

    ckpt_path = os.path.join(save_folder, "{}/ckpt/".format(cfg['model_name']))
    wpaths = [f.path for f in os.scandir(ckpt_path) if f.is_file()]
    for path in wpaths[start_e:] :
        print("src: {}\n Evaluate...".format(path))
        model = get_model(cfg)
        load_model(model, path)
        #valid
        model.eval()
        with torch.no_grad():
            _, acc, t = get_featurs(model, testloader)
            acc = np.max(acc)
            if acc > max_acc:
                best_t = t
                best_e = path
                max_acc = acc
    print("best model {} - best acc: {} - threshold: {}".format(best_e, max_acc, best_t))

if __name__ == '__main__':
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    main(config, save_folder=args.s, test_file=args.t, start_e=args.e)