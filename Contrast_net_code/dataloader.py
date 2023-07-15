import torch
import numpy as np
import os
import torch.utils.data as data
import cv2
import random


class lowlight_loader(data.Dataset):

    def __init__(self,image_path,ground_truth_path):
        self.size = 256
        self.image_path=image_path
        self.ground_truth=ground_truth_path
        self.name=os.listdir(image_path)
        self.name_gt = os.listdir(ground_truth_path)

        print("Total training examples:", len(self.name))

    def __trans__(self, img):
        # print(2)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = (np.asarray(img) / 255.0)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img
    def __getitem__(self, index):

        self.name.sort(key=lambda x: int(x[:4]))
        self.name_gt.sort(key=lambda x: int(x[:4]))

        name_gt = self.name_gt[int(index/35)]
        name = self.name[index]
        img = cv2.imread(os.path.join(self.image_path, name))

        gt = cv2.imread(os.path.join(self.ground_truth, name_gt))



        return self.__trans__(img),self.__trans__(gt)


    def __len__(self):
        return len(self.name)
