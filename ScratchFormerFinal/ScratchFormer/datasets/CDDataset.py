import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CDDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256, is_train=True):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.is_train = is_train

        # Read list of image names (without .png extension)
        list_path = os.path.join(root_dir, 'list', f'{split}.txt')
        with open(list_path, 'r') as f:
            self.img_list = [os.path.splitext(line.strip())[0] for line in f if line.strip()]

        # LEVIR normalization stats
        self.mean = np.array([0.40, 0.43, 0.45])
        self.std = np.array([0.19, 0.18, 0.21])

    def __len__(self):
        return len(self.img_list)

    def preprocess_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # to RGB
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img

    def preprocess_label(self, label_path):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        label = torch.from_numpy(label).long().unsqueeze(0)  # [1, H, W]
        return label

    def __getitem__(self, idx):
        name = self.img_list[idx]

        img_A_path = os.path.join(self.root_dir, 'A', f'{name}.png')
        img_B_path = os.path.join(self.root_dir, 'B', f'{name}.png')
        label_path = os.path.join(self.root_dir, 'label', f'{name}.png')

        A = self.preprocess_img(img_A_path)
        B = self.preprocess_img(img_B_path)
        L = self.preprocess_label(label_path)

        return {'A': A, 'B': B, 'L': L}
