import os
import argparse
import torch
import yaml
import tqdm

from torch.nn import DataParallel
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from modules.models import get_model
from modules.dataloader import get_DataLoader, Dataset, LFWdataset
from modules.partial_fc import CosMarginProduct, ArcMarginProduct, NormalFCLayer, L_ArcMarginProduct
from modules.evaluate import evaluate_model
from modules.focal_loss import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    parser.add_argument("--t", type=str, help="file contain training img paths")
    parser.add_argument("--v", type=str, help="folder contain verification imgs")
    return parser.parse_args()

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def main(cfg, img_file, test_file, n_workers=2):
    #setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    #setup path
    save_path = os.path.join(os.getcwd(), "save", cfg['model_name'])
    ckpt_path = os.path.join(save_path, "ckpt")
    log_path  = os.path.join(save_path, "log")
    if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    #train data
    train_dataset = Dataset(data_list_file=img_file,
                       is_training=True,
                       input_shape=(3, cfg['image_size'], cfg['image_size']))
    trainloader = get_DataLoader(train_dataset,
                                   batch_size=cfg['batch_size'],
                                   shuffle=True,
                                  num_workers=n_workers)
    #valid data
    test_dataset = LFWdataset(data_list_file=os.path.join(test_file, "lfw_pair.txt").replace("\\", "/"),
                                path=test_file)
    testloader = get_DataLoader(test_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=True,
                                num_workers=n_workers)
    #get backbone + head
    backbone = get_model(cfg)
    if cfg['loss'].lower() == 'cosloss':
        print("use Cos-Loss")
        partial_fc = CosMarginProduct(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
    elif cfg['loss'].lower() == 'arcloss':
        print("use ArcLoss")
        partial_fc = ArcMarginProduct(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
    elif cfg['loss'].lower() == 'l-arcloss':
        print("use L-ArcLoss")
        partial_fc = L_ArcMarginProduct(in_features=cfg['embd_size'],
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
    if cfg['criterion'] == 'focal':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    steps_per_epoch = cfg['sample_num'] // cfg['batch_size'] + 1
    scale = int(512 / cfg['batch_size'])
    lr_steps = [ int(scale * s) for s in cfg['lr_steps']] #epochs
    scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)
    #scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)
    print(lr_steps)
    #loop
    max_acc = 0.
    writer = SummaryWriter(log_path)
    for e in range(1,cfg['epoch_num']+1):
        print("Epoch: {}/{} \n-LR: {:.6f} \n-Train...".format(e,cfg['epoch_num'], scheduler.get_last_lr()[0]))
        backbone.train()
        total_loss = 0.0
        num_batchs = 0
        num_correct = 0.
        for data in tqdm.tqdm(iter(trainloader)):
            inputs, label = data
            inputs = inputs.to(device)
            label = label.to(device).long()

            embds = backbone(inputs)
            logits = partial_fc(embds, label)
            loss = criterion(logits, label)
            
            #update metrics
            total_loss += loss.item()
            num_batchs += 1
            indices = torch.max(logits, 1)[1]
            num_correct += torch.sum(torch.eq(indices, label).view(-1)).item()
            
            #update weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
            
            if num_batchs % 100 == 0:    # every 1000 mini-batches...
                # ...log the running loss
                writer.add_scalar('training loss',
                            total_loss / num_batchs,
                            (e-1) * len(trainloader) + num_batchs)
                writer.add_scalar('learning rate',
                            scheduler.get_last_lr()[0],
                            (e-1) * len(trainloader) + num_batchs)
        scheduler.step()         
        #test
        backbone.eval()
        print("-Validate...") 
        with torch.no_grad():
            acc = evaluate_model(backbone, testloader, device=device)
            if acc >= max_acc:
                max_acc = acc
                save_model(backbone,ckpt_path, cfg['model_name'], e)
            writer.add_scalar('verification accuracy',acc, e * num_batchs)
            writer.add_scalar('training accuracy', num_correct / cfg['sample_num'], e * num_batchs)
            print("\t--Train Loss: {:.5f} \n\t--Train accuracy: {:.5f}".format(total_loss / num_batchs, num_correct / cfg['sample_num']))
    writer.close()


if __name__ == '__main__':
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    print(config)
    main(config, img_file=args.t, test_file=args.v ,n_workers=args.n)