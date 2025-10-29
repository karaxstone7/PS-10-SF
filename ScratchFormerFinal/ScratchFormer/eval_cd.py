import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
from PIL import Image
import cv2

# === Fix import paths ===
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
sys.path.append(os.path.join(FILE_DIR, "datasets"))

from models.evaluator import CDEvaluator
from CDDataset import CDDataset
import utils


# --------------------------------------------------------------
# ✅ Red‑Blue Visualizer (TP=white, FP=red, FN=blue, TN=black)
# --------------------------------------------------------------
def visualize_redblue(A, B, GT, PRED, out_path, sample_id):

    # Ensure numpy
    if isinstance(GT, torch.Tensor): GT = GT.cpu().numpy()
    if isinstance(PRED, torch.Tensor): PRED = PRED.cpu().numpy()

    GT = np.squeeze(GT).astype(np.uint8)
    PRED = np.squeeze(PRED).astype(np.uint8)

    # ✅ Force same spatial size
    if PRED.shape != GT.shape:
        PRED = cv2.resize(PRED, (GT.shape[1], GT.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert A/B to uint8 images
    A = (A * 255).astype(np.uint8)
    B = (B * 255).astype(np.uint8)

    GT_3 = np.stack([GT] * 3, axis=-1) * 255
    PRED_3 = np.stack([PRED] * 3, axis=-1) * 255

    # Create red-blue composite
    redblue = np.zeros((GT.shape[0], GT.shape[1], 3), dtype=np.uint8)

    mask_tp = (GT == 1) & (PRED == 1)
    mask_fp = (GT == 0) & (PRED == 1)
    mask_fn = (GT == 1) & (PRED == 0)

    if np.any(mask_tp): redblue[mask_tp] = [255, 255, 255]   # TP
    if np.any(mask_fp): redblue[mask_fp] = [255, 0, 0]       # FP
    if np.any(mask_fn): redblue[mask_fn] = [0, 0, 255]       # FN

    strip = np.concatenate([A, B, GT_3, PRED_3, redblue], axis=1)

    os.makedirs(out_path, exist_ok=True)
    save_path = os.path.join(out_path, f"sample_{sample_id:04d}.png")
    Image.fromarray(strip).save(save_path)
    print(f"[✅ Saved] {save_path}")


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    parser = ArgumentParser()

    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--project_name', type=str, default='scratchformer_eval')
    parser.add_argument('--checkpoints_root', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, default='best_ckpt.pt')
    parser.add_argument('--vis_root', type=str, default='vis')

    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--net_G', type=str, default='ScratchFormer')
    parser.add_argument('--multi_scale_infer', action='store_true')

    parser.add_argument('--redblue_vis', action='store_true')

    args = parser.parse_args()
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x.isdigit()]
    utils.get_device(args)
    print(f"Using GPU(s): {args.gpu_ids}")

    args.checkpoint_dir = args.checkpoints_root
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    print(f"[📂] Loading test data from: {args.root_dir}")
    dataset = CDDataset(root_dir=args.root_dir, split=args.split, img_size=args.img_size, is_train=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if len(dataloader) == 0:
        print("❌ Dataloader empty. Check list/test.txt and filenames.")
        return

    print(f"[✅] Total test samples: {len(dataloader)}")

    evaluator = CDEvaluator(args=args, dataloader=dataloader)

    if args.redblue_vis:
        print("[🎨] Generating Red‑Blue visual CM maps...")
        evaluator._load_checkpoint(args.checkpoint_name)
        evaluator.net_G.eval()

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                A = batch['A'].to(evaluator.device)
                B = batch['B'].to(evaluator.device)
                L = batch['L'].to(evaluator.device)

                pred = evaluator.net_G(A, B)
                if isinstance(pred, list): pred = pred[-1]
                pred = torch.argmax(pred, dim=1)

                visualize_redblue(
                    A.squeeze().cpu().numpy().transpose(1, 2, 0),
                    B.squeeze().cpu().numpy().transpose(1, 2, 0),
                    L.squeeze(),
                    pred.squeeze(),
                    args.vis_dir,
                    idx
                )

        print(f"\n📁 Done. Saved to:\n{args.vis_dir}\n")

    else:
        print("[🧪] Running evaluation metrics...")
        evaluator.eval_models(checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    main()
