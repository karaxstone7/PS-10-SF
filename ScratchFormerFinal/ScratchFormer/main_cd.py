from argparse import ArgumentParser
import torch
import os

import utils
from models.trainer import CDTrainer


def train(args):
    dataloaders = utils.get_loaders(args)  # uses root_dir + list_dir
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(
        args.data_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        is_train=False,
        split='test'
    )
    model = CDEvaluator(args=args, dataloader=dataloader)
    model.eval_models()


if __name__ == '__main__':

    parser = ArgumentParser()

    # device -------------------------
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, e.g., 0 or 0,1 or -1 for CPU')
    
    # project dirs -------------------
    parser.add_argument('--project_name', type=str, default='scratchformer')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints')
    parser.add_argument('--vis_root', type=str, default='./vis')

    # dataset settings ----------------
    parser.add_argument('--dataset', type=str, default='CDDataset')
    parser.add_argument('--data_name', type=str, default='LEVIR_256')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to dataset root, e.g. /content/drive/.../LEVIR_256')
    parser.add_argument('--list_dir', type=str, required=True,
                        help='Path containing train.txt / val.txt / test.txt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--split_val', type=str, default='val')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--shuffle_AB', action='store_true')

    # model --------------------------
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--multi_scale_train', action='store_true')
    parser.add_argument('--multi_scale_infer', action='store_true')
    parser.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0])
    parser.add_argument('--net_G', type=str, default='ScratchFormer')
    parser.add_argument('--loss', type=str, default='ce')

    # optimization -------------------
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--lr_policy', type=str, default='linear')
    parser.add_argument('--lr_decay_iters', type=int, default=200)

    args = parser.parse_args()

    # device setup
    utils.get_device(args)
    print("[Using GPUs]:", args.gpu_ids)

    # checkpoint + vis dirs
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # train + test
    train(args)
    test(args)
