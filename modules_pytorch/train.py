import os
import argparse
import torch
import yaml
import tqdm

from torch.nn import DataParallel
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR

from models import get_model
from dataloader import get_DataLoader, Dataset
from partial_fc import CosMarginProduct, ArcMarginProduct, NormalFCLayer
from evaluate import evaluate_model
from focal_loss import *
from tfrecord_load import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    parser.add_argument("--t", type=str, help="file contain img paths")
    return parser.parse_args()

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def main(cfg, img_file, n_workers=2):
    #setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    #setup path
    save_path = os.path.join(os.getcwd(), "save", cfg['model_name'])
    ckpt_path = os.path.join(save_path, "ckpt")
    if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)

    #train data
    train_dataset = Dataset(data_list_file=img_file,
                       is_training=False,
                       input_shape=(3, cfg['image_size'], cfg['image_size']))
    trainloader = get_DataLoader(train_dataset,
                                   batch_size=cfg['batch_size'],
                                   shuffle=True,
                                  num_workers=n_workers)

    #valid data
    test_dataset = load_tfrecord_dataset(cfg['test_data'], cfg['batch_size'],
                          dtype="test", shuffle=True, buffer_size=1028, transform_img=False)

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
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
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
    if cfg['criterion'] == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    steps_per_epoch = cfg['sample_num'] // cfg['batch_size'] + 1
    scale = int(512 / cfg['batch_size'])
    lr_steps = [ scale * s for s in cfg['lr_steps']] #epochs
    #scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00001)

    #loop
    max_acc = 0.
    
    for e in range(1,cfg['epoch_num']+1):
        print("Epoch: {} \n-LR: {} \n-Train...".format(e, scheduler.get_lr()))
        backbone.train()
        total_loss = 0.0
        num_batchs = 0
        for data in tqdm.tqdm(iter(trainloader)):
            scheduler.step()
            inputs, label = data
            inputs = inputs.to(device)
            label = label.to(device).long()

            embds = backbone(inputs)
            logits = partial_fc(embds, label)
            loss = LossFuntion(logits, label)
            total_loss += loss.item()
            num_batchs += 1
            #print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
        #test
        backbone.eval()
        print("-Validate...") 
        with torch.no_grad():
            acc = evaluate_model(backbone, test_dataset, device=device)
            print("\t--Train Loss: {:.5f} ".format(total_loss / num_batchs))
            if acc >= max_acc:
                max_acc = acc
                save_model(backbone,ckpt_path, cfg['model_name'], e)



if __name__ == '__main__':
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    main(config, img_file=args.t ,n_workers=args.n)