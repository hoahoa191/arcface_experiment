import os
import argparse
import torch
import yaml
import tqdm

from torch.nn import DataParallel
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

from models import get_model
from dataloader import get_DataLoader, Dataset
from partial_fc import CosMarginProduct, ArcMarginProduct, NormalFCLayer
from evaluate import evaluate_model
from tfrecord_load import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    return parser.parse_args()

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def main(cfg):
    #setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    #setup path
    save_path = os.path.join(os.getcwd(), "save", cfg['model_name'])
    ckpt_path = os.path.join(save_path, "ckpt")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(ckpt_path)

    #train data
    # train_dataset = Dataset(data_list_file='/content/drive/MyDrive/Colab Notebooks/facenet/data_list.txt',
    #                   is_training=False,
    #                   input_shape=(3, cfg['image_size'], cfg['image_size']))
    # trainloader = get_DataLoader(train_dataset,
    #                               batch_size=cfg['batch_size'],
    #                               shuffle=True,
    #                               num_workers=2)
    trainloader = load_tfrecord_dataset(cfg['train_data'], cfg['batch_size'],
                          dtype="train", shuffle=True, buffer_size=1028, transform_img=True)
    #valid data
    valid_dataset = load_tfrecord_dataset(cfg['val_data'], cfg['batch_size'],
                          dtype="valid", shuffle=True, buffer_size=1028, transform_img=False)

    #get backbone + head
    backbone = get_model(cfg)

    if cfg['loss'].lower() == 'cosloss':
        print("use Cos-Loss")
        partial_fc = CosMarginProduct(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
    elif cfg['loss'].lower() == 'arcloss':
        print("use Arc-Loss")
        partial_fc = ArcMarginProduct(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'], easy_margin=False)
    else:
        print("No Additative Margin")
        partial_fc = NormalFCLayer(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'])
    #data parapell
    backbone = DataParallel(backbone.to(device))
    partial_fc = DataParallel(partial_fc.to(device))

    #optimizer
    if 'optimizer' in cfg.keys() and cfg['optimizer'].lower() == 'adam':
        optimizer = Adam([{'params': backbone.parameters()}, {'params': partial_fc.parameters()}],
                                    lr=cfg['base_lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = SGD([{'params': backbone.parameters()}, {'params': partial_fc.parameters()}],
                                     lr=cfg['base_lr'], weight_decay=cfg['weight_decay'])
    #LossFunction+scheduerLR
    LossFuntion = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=cfg['step_size'], gamma=0.1)

    #loop
    max_acc = 0.
    num_batchs =  cfg['sample_num'] // cfg['batch_size'] + 1
    for e in range(cfg['epoch_num']):
        print("Epoch: {}\n-Train...".format(e))
        backbone.train()
        total_loss = 0.0
        for data in tqdm.tqdm(iter(trainloader)):
            inputs, label = data
            # inputs = inputs.to(device)
            # label = label.to(device).long()

            inputs = torch.from_numpy(inputs.numpy()).permute(0, 3, 1, 2)
            inputs = inputs.to(device)
            label = torch.from_numpy(label.numpy()).to(device).long()

            embds = backbone(inputs)
            logits = partial_fc(embds, label)
            loss = LossFuntion(logits, label)
            total_loss += loss.item()
            #print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
        scheduler.step()
        backbone.eval()
        print("-Validate...") 
        with torch.no_grad():
            acc = evaluate_model(backbone, valid_dataset, device=device)
            print("\t--Train Loss: {} ".format(total_loss / num_batchs))
            if acc >= max_acc:
                max_acc = acc
                save_model(backbone,ckpt_path, cfg['model_name'], e+1)



if __name__ == '__main__':
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    main(config)