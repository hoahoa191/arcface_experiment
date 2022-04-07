import os
import argparse
import torch
import yaml
import tqdm
import time
from torch.nn import DataParallel
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR

from modules.models import Backbone
from modules.dataloader import load_tfrecord_dataset
from modules.partial_fc import CosMarginProduct, ArcMarginProduct, MagMarginProduct
#from modules.evaluate import evaluate_model
#from modules.focal_loss import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    parser.add_argument("--t", type=str, help="file contain training img paths")
    parser.add_argument("--v", type=str, help="file contain training verification imgs")
    return parser.parse_args()

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def main(n_workers=2):
    #setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    #setup path
    save_path = os.path.join(os.getcwd(), "save", "magface")
    ckpt_path = os.path.join(save_path, "ckpt")
    if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)
    trainloader = load_tfrecord_dataset("./casia.tfrecord", batch_size=128, binary_img=True, shuffle=True, buffer_size=10240)
    
    #get backbone + head
    backbone = Backbone(50, drop_ratio=0.4, embedding_size=512, mode='ir_se')
    
#     partial_fc = CosMarginProduct(in_features=512,
#                                 out_features=10572,
#                                 s=30, m=0.35)
    
    partial_fc = MagMarginProduct(in_features=512,
                                out_features=10572,
                                s=30,l_a=10, u_a=110, l_m=0.45, u_m=0.8, lambda_g=20)
    
    #data parapell
    backbone = DataParallel(backbone.to(device))
    partial_fc = DataParallel(partial_fc.to(device))
    optimizer = SGD([{'params': backbone.parameters()}, {'params': partial_fc.parameters()}],
                                     lr=0.1, weight_decay=5e-4)
    #LossFunction+scheduerLR
    criterion = torch.nn.CrossEntropyLoss()
    lr_steps = [4, 9, 15, 18] #epochs
    scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)
    #scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)

    #loop

    for e in range(1, 25):
        s = time.time()
        print("Epoch: {}/{} \n-LR: {:.6f} \n-Train...".format(e,cfg['epoch_num'], scheduler.get_last_lr()[0]))
        backbone.train()
        total_loss = 0.0
        num_batchs = 0
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
            
            #update weights
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
            
        scheduler.step()  
        save_model(backbone,ckpt_path, "magface", e)
        #test
#         backbone.eval()
#         print("-Validate...") 
#         with torch.no_grad():
#             acc = evaluate_model(backbone, testloader, device=device)
#             if acc >= max_acc:
#                 max_acc = acc
                
#             writer.add_scalar('verification accuracy',acc, e * num_batchs)
#             writer.add_scalar('training accuracy', num_correct / cfg['sample_num'], e * num_batchs)
            
#             print('\t--lfw face verification accuracy: {:.5f}'.format(acc))
        print("\t--Train Loss: {:.5f}".format(total_loss / num_batchs))
        print('\t--total time is {:.3f}'.format(time.time()-s))
  


if __name__ == '__main__':
    args = get_args()
    main(n_workers=args.n)
