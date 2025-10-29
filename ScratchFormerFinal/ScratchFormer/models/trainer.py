import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

import utils
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, softmax_helper, FocalLoss, mIoULoss, mmIoULoss
from misc.logger_tool import Logger, Timer


class CDTrainer():

    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders = dataloaders
        self.n_class = args.n_class
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu")
        print(self.device)

        # Optimizer
        self.lr = args.lr
        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)

        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)
        self.running_metric = ConfuseMatrixMeter(n_class=2)

        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        self.timer = Timer()

        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs
        self.global_step = 0
        self.batch_size = args.batch_size
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.shuffle_AB = args.shuffle_AB
        self.multi_scale_train = args.multi_scale_train
        self.multi_scale_infer = args.multi_scale_infer
        self.weights = tuple(args.multi_pred_weights)

        # Loss setup
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'fl':
            alpha = get_alpha(dataloaders['train'])
            self._pxl_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)
        elif args.loss == "miou":
            alpha = np.asarray(get_alpha(dataloaders['train']))
            alpha = alpha / np.sum(alpha)
            weights = 1 - torch.from_numpy(alpha).cuda()
            self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=args.n_class).cuda()
        elif args.loss == "mmiou":
            self._pxl_loss = mmIoULoss(n_classes=args.n_class).cuda()
        else:
            raise NotImplementedError(args.loss)

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        print("\n")
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            self.logger.write('Loading checkpoint...\n')
            import torch.serialization, numpy as np
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(ckpt_path, map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(self.device)
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.logger.write('Resuming from epoch %d | Best acc: %.4f at epoch %d\n' %
                              (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
        else:
            self.logger.write('Training from scratch...\n')
            self.net_G.to(self.device)
        print("\n")

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)

        if self.multi_scale_infer:
            self.G_final_pred = torch.zeros_like(self.G_pred[-1]).to(self.device)
            for pred in self.G_pred:
                if pred.size(2) != self.G_pred[-1].size(2):
                    self.G_final_pred += F.interpolate(pred, size=self.G_pred[-1].size()[2:], mode="nearest")
                else:
                    self.G_final_pred += pred
            self.G_final_pred /= len(self.G_pred)
        else:
            self.G_final_pred = self.G_pred[-1]

    def _backward_G(self):
        gt = self.batch['L'].to(self.device).float()
        if self.multi_scale_train:
            temp_loss = 0.0
            for i, pred in enumerate(self.G_pred):
                resized_gt = F.interpolate(gt, size=pred.size()[2:], mode="nearest")
                temp_loss += self.weights[i] * self._pxl_loss(pred, resized_gt)
            self.G_loss = temp_loss
        else:
            self.G_loss = self._pxl_loss(self.G_pred[-1], gt)
        self.G_loss.backward()

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def train_models(self):
        self._load_checkpoint()
        for epoch in range(self.epoch_to_start, self.max_num_epochs):
            self.epoch_id = epoch
            self.net_G.train()
            self.logger.write(f"\n[Epoch {epoch}] LR: {self.optimizer_G.param_groups[0]['lr']:.7f}\n")
            for i, batch in tqdm(enumerate(self.dataloaders['train']), total=len(self.dataloaders['train'])):
                self._forward_pass(batch)
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()

            self._save_checkpoint('last_ckpt.pt')
            self._save_checkpoint(f'epoch_{epoch:03d}.pt')
            self._save_visuals(epoch)
            self._evaluate(epoch)

            print(f"[Epoch {epoch}] Training done. Loss: {self.G_loss.item():.6f}")
            self.logger.write(f"Epoch {epoch}: Loss = {self.G_loss.item():.6f}\n")

    def _save_checkpoint(self, checkpoint_name):
        ckpt_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        state = {
            'epoch_id': self.epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id
        }
        torch.save(state, ckpt_path)
        print(f"[Saved] Checkpoint: {ckpt_path}")

    def _save_visuals(self, epoch):
        self.net_G.eval()
        vis_path = os.path.join(self.vis_dir, f"epoch_{epoch:03d}")
        os.makedirs(vis_path, exist_ok=True)
        with torch.no_grad():
            batch = next(iter(self.dataloaders['val']))
            A = batch['A'].to(self.device)
            B = batch['B'].to(self.device)
            L = batch['L'].to(self.device)
            preds = self.net_G(A, B)
            if isinstance(preds, list):
                preds = preds[-1]
            preds = torch.argmax(preds, dim=1)

            A = A.cpu().numpy().transpose(0, 2, 3, 1)
            B = B.cpu().numpy().transpose(0, 2, 3, 1)
            L = L.squeeze(1).cpu().numpy()
            preds = preds.cpu().numpy()

            for i in range(min(8, len(A))):
                a_img = np.clip(A[i], 0, 1)
                b_img = np.clip(B[i], 0, 1)
                a_img = (a_img * 255).astype(np.uint8)
                b_img = (b_img * 255).astype(np.uint8)
                gt_img = (np.stack([L[i]]*3, axis=-1) * 255).astype(np.uint8)
                pd_img = (np.stack([preds[i]]*3, axis=-1) * 255).astype(np.uint8)
                concat = np.concatenate([a_img, b_img, gt_img, pd_img], axis=1)
                Image.fromarray(concat).save(os.path.join(vis_path, f"sample_{i:02d}.png"))
        print(f"[Saved] Visuals: {vis_path}")

    def _evaluate(self, epoch):
        self.net_G.eval()
        self.running_metric.clear()
        with torch.no_grad():
            for batch in self.dataloaders['val']:
                img_in1 = batch['A'].to(self.device)
                img_in2 = batch['B'].to(self.device)
                label = batch['L'].to(self.device)
                pred = self.net_G(img_in1, img_in2)
                if isinstance(pred, list):
                    pred = pred[-1]
                pred = torch.argmax(pred, dim=1)
                label = label.squeeze(1)
                self.running_metric.update_cm(pr=pred, gt=label)

        # ⏱️ Print and log metrics
        scores = self.running_metric.get_scores()
        print(f"\n[Validation Metrics] Epoch {epoch:03d}:")
        for k, v in sorted(scores.items()):
            print(f"    {k:<12s}: {v:.4f}")
        log_str = "\n".join([f"{k}: {v:.5f}" for k, v in scores.items()])
        self.logger.write(f"\n[Validation Metrics] Epoch {epoch}:\n{log_str}\n")
