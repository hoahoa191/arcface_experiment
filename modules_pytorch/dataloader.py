import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


class Dataset(data.Dataset):
    "Load data from tfrecord"
    def __init__(self, data_list_file, is_training=True, input_shape=(1, 112, 112)):
        self.is_training = is_training
        self.input_shape = input_shape
        
        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(img.strip()) for img in imgs]
        self.imgs = np.random.permutation(imgs)
        self.size        = len(imgs)

        normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        if self.is_training:
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                T.ColorJitter(brightness=.4, hue=.3),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        img_path = sample
        #x
        data = Image.open(img_path)
        data = data.convert('RGB') # convert image to monochrome('L') or RGB
        data = self.transforms(data)
        #y
        label = np.int32(sample.split("/")[-2])
        return data.float(), label

    def __len__(self):
        return self.size

def get_DataLoader(dataset, batch_size, shuffle=True,num_workers=4):
    return data.DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers)

if __name__ == '__main__':
    dataset = Dataset(data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      is_training=False,
                      input_shape=(1, 112, 112))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data).numpy()
        img = np.transpose(img, (1, 2, 0))
        img += np.array([1, 1, 1]) #revert norm 
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break