import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
from collections import OrderedDict
import cv2
import torch
import torch.nn.functional as F


class CDEvaluator():

    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.n_class = args.n_class
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device(
            f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu"
        )
        print(self.device)

        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        # output folders
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = os.path.join(args.vis_dir, 'colored')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.G_pred = None
        self.batch = None
        self.batch_id = 0

    # ------------------------------------------------------------
    # Load model weights
    # ------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        ckpt_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'❌ No such checkpoint: {ckpt_path}')

        self.logger.write(f'🧩 Loading checkpoint from: {ckpt_path}\n')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # handle both raw and full-format checkpoints
        if isinstance(checkpoint, dict) and 'model_G_state_dict' in checkpoint:
            state_dict = checkpoint['model_G_state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = v

        if isinstance(self.net_G, torch.nn.DataParallel):
            self.net_G = self.net_G.module

        self.net_G.load_state_dict(new_state_dict, strict=False)
        self.net_G.to(self.device)
        self.logger.write(f"✅ Checkpoint loaded successfully from {checkpoint_name}\n")

    # ------------------------------------------------------------
    # Metric update
    # ------------------------------------------------------------
    def _update_metric(self):
        target = self.batch['L'].to(self.device).detach()
        G_pred = torch.argmax(self.G_pred.detach(), dim=1)
        return self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())

    # ------------------------------------------------------------
    # Colored visualization (TP=white, FP=red, FN=blue, TN=black)
    # ------------------------------------------------------------
    def _save_colored_vis(self):
        A = utils.make_numpy_grid(de_norm(self.batch['A']))
        B = utils.make_numpy_grid(de_norm(self.batch['B']))
        gt = self.batch['L'][0].squeeze().cpu().numpy().astype(np.uint8)   # ✅ fixed shape
        pred = torch.argmax(self.G_pred[0], dim=0).cpu().numpy().astype(np.uint8)

        # resize prediction to match GT
        h, w = gt.shape
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        # create color composite
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        tp = (pred == 1) & (gt == 1)
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)
        tn = (pred == 0) & (gt == 0)

        vis[tp] = [255, 255, 255]  # white
        vis[fp] = [255, 0, 0]      # red
        vis[fn] = [0, 0, 255]      # blue
        vis[tn] = [0, 0, 0]        # black

        fname = os.path.join(self.vis_dir, f'vis_{self.batch_id:04d}.png')
        cv2.imwrite(fname, vis)

    # ------------------------------------------------------------
    # Epoch metric summary
    # ------------------------------------------------------------
    def _collect_epoch_states(self):
        scores_dict = self.running_metric.get_scores()
        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        msg = ' '.join([f"{k}: {v:.5f}" for k, v in scores_dict.items()])
        self.logger.write(f"{msg}\n\n")
        print(f"[🧠] Final Eval Metrics:\n{msg}\n")

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------
    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)[-1]
        self.G_pred = F.interpolate(self.G_pred, size=[256, 256], mode='bilinear')

    # ------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------
    def eval_models(self, checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)
        self.logger.write('Begin evaluation...\n')

        self.running_metric.clear()
        self.net_G.eval()

        for self.batch_id, batch in enumerate(self.dataloader):
            with torch.no_grad():
                self._forward_pass(batch)
            self._update_metric()
            self._save_colored_vis()

        self._collect_epoch_states()
