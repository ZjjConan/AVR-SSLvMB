import os
import random
import glob
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset



class RAVEN(Dataset):
    def __init__(self, dataset_dir, data_split=None, image_size=80, transform=None, subset=None, permute=False):
        self.dataset_dir = dataset_dir
        self.data_split = data_split
        self.image_size = image_size
        self.transform = transform
        self.permute = permute

        subsets = os.listdir(self.dataset_dir)

        self.file_names = []
        for i in subsets:
            file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.dataset_dir, i, "*_" + self.data_split + ".npz"))]
            file_names.sort()
            self.file_names += [os.path.join(i, f) for f in file_names]


    def __len__(self):
        return len(self.file_names)

    def get_data(self, idx):
        data_file = self.file_names[idx]
        data_path = os.path.join(self.dataset_dir, data_file)
        data = np.load(data_path)

        image = data["image"].reshape(16, 160, 160)
        if self.image_size != 160:
            resize_image = np.zeros((16, self.image_size, self.image_size))
            for idx in range(0, 16):
                resize_image[idx] = cv2.resize(
                    image[idx], (self.image_size, self.image_size)
                )
        else:
            resize_image = image

        return resize_image, data, data_file

    def __getitem__(self, idx):
        image, data, data_path = self.get_data(idx)

        # Get additional data
        target = data["target"]
        meta_target = data["meta_target"]
        structure = data["structure"]
        structure_encoded = data["meta_matrix"]
        del data

        if self.transform:
            image = torch.from_numpy(image).type(torch.float32)
            C, H, W = image.size()
            image = self.transform(image.reshape(C, 1, H, W))
            if type(image) is list:
                image = [im.reshape(C, H, W) for im in image]
            else:
                image = image.reshape(C, H, W)

        if self.permute:
            new_target = random.choice(range(8))
            if new_target != target:
                image[[8 + new_target, 8 + target]] = image[[8 + target, 8 + new_target]]
                target = new_target

        target = torch.tensor(target, dtype=torch.long)
        meta_target = torch.tensor(meta_target, dtype=torch.float32)
        structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)

        return image, target, meta_target, structure_encoded, data_path