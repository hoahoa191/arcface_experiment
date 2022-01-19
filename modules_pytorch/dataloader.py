import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


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
    def __init__(self, data_list_file, num_images, is_training=True, input_shape=(1, 128, 128)):
        self.is_training = is_training
        self.input_shape = input_shape
        self.size        = num_images
        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(img) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.is_training:
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
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
        data = data.convert('L') # convert image to monochrome
        data = self.transforms(data)
        #y
        label = np.int32(sample.split("/")[-2])

        return data.float(), label

    def __len__(self):
        return self.size



if __name__ == '__main__':
    dataset = Dataset(root='/content/drive/MyDrive/Colab Notebooks/facenet/data',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      is_training=False,
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data).numpy()
        img = np.transpose(img, (1, 2, 0))
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break