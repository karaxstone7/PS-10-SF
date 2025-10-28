import os
import sys
import torch
from argparse import ArgumentParser

# === Fix imports no matter where this is run from ===
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
sys.path.append(os.path.join(FILE_DIR, "datasets"))  # ✅ for CDDataset.py

# === Core imports ===
from models.evaluator import CDEvaluator
import utils
from CDDataset import CDDataset  # ✅ from datasets/CDDataset.py

def main():
    parser = ArgumentParser()

    # ----- Core settings -----
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--project_name', type=str, default='scratchformer_eval')
    parser.add_argument('--checkpoints_root', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, default='best_ckpt.pt')
    parser.add_argument('--vis_root', type=str, default='vis')

    # ----- Dataset settings -----
    parser.add_argument('--data_name', type=str, default='LEVIR_256')
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')  # ✅ used to build list file path
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    # ----- Model settings -----
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--net_G', type=str, default='ScratchFormer')
    parser.add_argument('--multi_scale_infer', action='store_true')

    args = parser.parse_args()

    # Fix GPU handling
    args.gpu_ids = [int(i) for i in args.gpu_ids.split(',') if i.strip().isdigit()]
    utils.get_device(args)
    print(f"Using GPU(s): {args.gpu_ids}")

    # Setup paths
    args.checkpoint_dir = args.checkpoints_root
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    # Dataset & loader
    dataset = CDDataset(
        root_dir=args.root_dir,
        split=args.split,
        img_size=args.img_size,
        is_train=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Evaluate
    evaluator = CDEvaluator(args=args, dataloader=dataloader)
    evaluator.eval_models(checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    main()