import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

import data_config
from datasets.CDDataset import CDDataset  # ✅ Correct import


def get_loader_with_paths(data_name, root_dir, list_dir, img_size, batch_size, is_train, split):
    if data_name in ['LEVIR_256', 'CDD']:
        dataset = CDDataset(
            root_dir=root_dir,
            list_path=list_dir,
            split=split,
            img_size=img_size,
            is_train=is_train
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {data_name}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return loader


def get_loaders(args):
    """
    Create both train and val dataloaders.
    """
    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = getattr(args, "split_val", "val")

    if args.dataset == 'CDDataset':
        training_set = CDDataset(
            root_dir=root_dir,
            split=split,
            img_size=args.img_size,
            is_train=True,
            label_transform=label_transform
        )
        val_set = CDDataset(
            root_dir=root_dir,
            split=split_val,
            img_size=args.img_size,
            is_train=False,
            label_transform=label_transform
        )
    else:
        raise NotImplementedError(
            f"Wrong dataset name {args.dataset} (choose one from [CDDataset])"
        )

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        for x in ['train', 'val']
    }
    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    """
    Convert tensor batch to numpy grid for visualization.
    """
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    """
    Denormalize image tensors from [-1,1] to [0,1].
    """
    return tensor_data * 0.5 + 0.5


def get_device(args):
    """
    Safely select GPU or CPU device, handling both string and list formats.
    """
    # ✅ handle both "0" and [0] cases
    if isinstance(args.gpu_ids, str):
        gpu_list = [int(i) for i in args.gpu_ids.split(',') if i.strip().isdigit()]
    elif isinstance(args.gpu_ids, list):
        gpu_list = [int(i) for i in args.gpu_ids if isinstance(i, int) or str(i).isdigit()]
    else:
        raise ValueError("gpu_ids must be a string like '0' or a list like [0]")

    args.gpu_ids = [i for i in gpu_list if i >= 0]

    if torch.cuda.is_available() and len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
    else:
        print("⚠️ No GPU found — using CPU.")

    return args
