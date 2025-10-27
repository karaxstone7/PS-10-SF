import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class CDDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256, is_train=True, label_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.is_train = is_train
        self.label_transform = label_transform

        # Read image names from list file (without .png extension)
        list_path = os.path.join(root_dir, 'list', f'{split}.txt')
        with open(list_path, 'r') as f:
            self.img_list = [os.path.splitext(line.strip())[0] for line in f if line.strip()]

        # Standard transforms (resize + to tensor)
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        self.label_transform = label_transform or transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = self.img_list[idx]

        img_A_path = os.path.join(self.root_dir, 'A', f'{name}.png')
        img_B_path = os.path.join(self.root_dir, 'B', f'{name}.png')
        label_path = os.path.join(self.root_dir, 'label', f'{name}.png')

        img_A = Image.open(img_A_path).convert('RGB')
        img_B = Image.open(img_B_path).convert('RGB')
        label = Image.open(label_path)

        A = self.img_transform(img_A)
        B = self.img_transform(img_B)
        L = self.label_transform(label)

        return {'A': A, 'B': B, 'L': L}