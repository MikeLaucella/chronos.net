"""
bolt.py

The lightning Mae wrapper module.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR, LinearLR, SequentialLR
from torchmetrics.classification import (Accuracy, JaccardIndex)

from lightning import LightningModule
from chronos.model.segformer.lr_decay import param_groups_seg


def denorm(tensor, mean, std):
    # Non-inplace denorm and clip to [0, 1]
    return tensor * std + mean

def to_wandb_uint8(tensor):
    """Clamps to [0, 1], scales to 255, and converts to uint8."""
    return (tensor.clamp(0, 1) * 255).to(torch.uint8)

def color_label(label):
    """
    Maps class indices to RGB colors.
    label: Tensor of shape [B, H, W]
    returns: RGB Tensor of shape [B, 3, H, W]
    """
    palette = torch.tensor([
        [0, 0, 0],       # 0: Background (Black)
        [128, 128, 128], # 1: Parking Lot (Gray)
        [255, 0, 0],     # 2: Building (Red)
        [0, 255, 0],     # 3: Grass (Light Green)
        [50, 50, 50],    # 4: Road (Dark Gray/Black)
        [0, 100, 0],     # 5: Tree (Dark Green)
        [255, 255, 0],   # 6: Agriculture (Yellow)
        [0, 0, 255],     # 7: Water (Blue)
        [165, 42, 42],   # 8: Baren (Brown)
        [200, 200, 200], # 9: Other (Light Gray)
    ], device=label.device).to(torch.uint8)

    # Map index to RGB: [B, H, W, 3]
    rgb = palette[label.long()]
    # Permute to [B, 3, H, W]
    return rgb.permute(0, 3, 1, 2).float() / 255.0

def get_weights():
    return torch.tensor([ # log frequency weights to handle class imbalance
        0.0,
        2.81260978,
        1.80141839,
        1.,
        1.49230635,
        1.25192591,
        5.35576274,
        3.61756328,
        4.37721388,
        1.# clamp other to 1.0 5.90910363
    ], dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.weight = get_weights()

    def forward(self, inputs, targets):
        # Compute standard cross entropy loss (with no reduction initially)
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index #, weight=self.weight.to(inputs.device)
        )
        
        # Calculate pt (probability of the ground truth class)
        pt = torch.exp(-ce_loss)
        
        # Calculate the focal loss component
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SegmentationMetrics(nn.Module):
    """Segmentation metrics for evaluation."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.acc = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=0)
        self.iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0)
        #self.loss = F.cross_entropy
        self.loss = FocalLoss(gamma=2, ignore_index=0)

    def _score_empty(self, y: torch.Tensor, y_hat: torch.Tensor) -> dict:
        """Score empty predictions."""
        return {
            'y': y,
            'y_hat': y_hat,
            'loss': y_hat.sum() * 0.0,  # zero loss if both are empty
            'acc': torch.tensor(0.0, device=y.device),
            'iou': torch.tensor(0.0, device=y.device)
        }

    def score(self, y: torch.Tensor, y_hat: torch.Tensor) -> dict:
        """Calculate the metrics."""
        acc = self.acc(y_hat, y)
        iou = self.iou(y_hat, y)
        #self.weight = self.weight.to(y.device)
        loss = self.loss(y_hat, y).float()

        mask = y == 0
        if mask.all():
             return self._score_empty(y, y_hat)

        return {
            'y': y,
            'y_hat': y_hat,
            'loss': loss,
            'acc': acc,
            'iou': iou
        }


class SegmentationLogger:
    """Segmentation metrics logger."""

    def __init__(self, hook: LightningModule, log_step: int = 100):
        self.log_step = log_step
        self.hook = hook

    def train(self, metrics: dict, batch_idx: int):
        """Log training metrics."""
        self._log("train", metrics, batch_idx)

    def validation(self, metrics: dict, batch_idx: int):
        """Log validation metrics."""
        self._log("val", metrics, batch_idx)

    def test(self, metrics: dict, batch_idx: int):
        """Log test metrics."""
        self._log("test", metrics, batch_idx)

    def _log(self, mode: str, metrics: dict, batch_idx: int):
        """Log the metrics."""
        batch_size = metrics['y'].size(0)
        loss = metrics['loss']
        acc = metrics['acc']
        iou = metrics['iou']

        if batch_idx % self.log_step == 0: # Log every n steps for validation/test
            #self.hook.log(f"{mode}_acc_step", acc, on_epoch=False, on_step=True, prog_bar=True, batch_size=batch_size)
            #self.hook.log(f"{mode}_iou_step", iou, on_epoch=False, on_step=True, prog_bar=True, batch_size=batch_size)
            self.hook.log(f"{mode}_loss_step", loss, on_epoch=False, on_step=True, prog_bar=True, batch_size=batch_size)

        # combined
        self.hook.log(f"{mode}/acc", acc, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.hook.log(f"{mode}/iou", iou, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.hook.log(f"{mode}/loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.hook.log(f"{mode}/ce_loss", metrics['ce_loss'], on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)


class SegformerBolt3(LightningModule):
    """The lightning Segformer wrapper module."""

    def __init__(self,
                 teacher: nn.Module,
                 student: nn.Module,
                 classes: int,
                 lr: float=1e-5,
                 alpha: float=0.99):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.lr = lr
        self.alpha = alpha
        self.classes = classes
        self._metrics = SegmentationMetrics(classes)
        self._logger = SegmentationLogger(self)

    def forward(self, inp: dict) -> torch.Tensor:
        """Perform forward pass"""
        img1 = inp['naip_hist']
        img2 = inp['eros_hist']
        mask = inp['masks']

        with torch.amp.autocast(self.device.type):
            with torch.no_grad():
                feat_modern = self.teacher.full(img1).hidden_states

            pred_modern = self.student.full(img1)
            out = self._metrics.score(mask, pred_modern.logits)
            ce_loss = out['loss']

            feat_hist = self.student.full(img2).hidden_states

        loss_const = self._structural_consistency_loss(feat_modern, feat_hist, stages=[2, 3])

        # Total Loss
        # lambda_c is usually small (e.g., 0.1) to prevent the style 
        # from 'overwhelming' the actual class learning
        total_loss = ce_loss + (0.1 * loss_const)
        out['ce_loss'] = ce_loss
        out['loss'] = total_loss

        return out

    def _structural_consistency_loss(self, feats_mod, feats_hist, stages=[2, 3]):
        loss_c = 0
        for i in stages:
            f_m = feats_mod[i]
            f_h = feats_hist[i]

            # 1. Spatial Pooling (Handle the ~10px orthorectification jitter)
            f_m = F.avg_pool2d(f_m, kernel_size=3, stride=1, padding=1)
            f_h = F.avg_pool2d(f_h, kernel_size=3, stride=1, padding=1)

            # 2. Cosine Similarity (Focus on 'What' it is, not 'How bright' it is)
            cos_sim = F.cosine_similarity(f_m, f_h, dim=1)

            # 3. Mean across spatial dimensions
            # loss_c += F.mse_loss(f_m, f_h).float()
            loss_c += (1.0 - cos_sim.mean())

        return loss_c

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch < 5:
            return  # Skip EMA updates for first 5 epochs to let student learn basic patterns
        # EMA Update: teacher = alpha * teacher + (1 - alpha) * student
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            with torch.no_grad():
                for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                    t_param.data = self.alpha * t_param.data + (1.0 - self.alpha) * s_param.data

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        out = self(batch)
        self._logger.train(out, batch_idx)
        self.last_train_batch = batch
        return out['loss']

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        out = self(batch)
        self._logger.validation(out, batch_idx)
        self.last_val_batch = batch
        return out['loss']

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        out = self(batch)
        self._logger.test(out, batch_idx)
        return out['loss']

    def on_validation_epoch_end(self):
        if not self.logger or not hasattr(self, 'last_val_batch'):
            return

        # 1. Fetch last batch data
        batch = self.last_val_batch
        self._on_end(batch, type="val")

    def on_train_epoch_end(self):
        if not self.logger or not hasattr(self, 'last_train_batch'):
            return

        # 1. Fetch last batch data
        batch = self.last_train_batch
        self._on_end(batch, type="train")
    
    def _on_end(self, batch, type: str):
        # Take first 4 for a 4x4 grid
        naip = batch['naip_hist'][:4]
        eros = batch['eros_hist'][:4]
        labels = batch['masks'][:4]

        B = naip.shape[0]

        # 2. Get Model Predictions
        self.eval()
        with torch.no_grad():
            # SegFormer often outputs logits at 1/4 res; upsample to input size
            logits_naip = self.student(naip)
            logits_eros = self.student(eros)
            preds_naip = torch.argmax(logits_naip, dim=1) # [B, H, W]
            preds_eros = torch.argmax(logits_eros, dim=1) # [B, H, W]

        naip_mean = torch.tensor((0.6493820041822563, 0.6493820041822563, 0.6493820041822563), device=self.device).view(1, 3, 1, 1)
        naip_std = torch.tensor((0.17010567701149337, 0.17010567701149337, 0.17010567701149337), device=self.device).view(1, 3, 1, 1)

        # 3. Prepare Visual Tensors [0, 1]
        naip_vis = F.interpolate(denorm(naip, naip_mean, naip_std), size=(512, 512), mode="bilinear", align_corners=False)
        eros_vis = F.interpolate(denorm(eros, naip_mean, naip_std), size=(512, 512), mode="bilinear", align_corners=False)

        gt_vis = color_label(labels.squeeze(1))
        naip_pred_vis = color_label(preds_naip)
        eros_pred_vis = color_label(preds_eros)

        # 4. Concatenate for a 4x4 Grid [Rows: NAIP, GT, Pred, EROS]
        # combined shape: [16, 3, H, W]
        combined1 = torch.cat([naip_vis, gt_vis, naip_pred_vis], dim=0)
        grid1 = to_wandb_uint8(torchvision.utils.make_grid(combined1, nrow=B))

        combined2 = torch.cat([eros_vis, gt_vis, eros_pred_vis], dim=0)
        grid2 = to_wandb_uint8(torchvision.utils.make_grid(combined2, nrow=B))

        # 5. Log to WandB / TensorBoard
        self.logger.log_image(key=f"NAIP_{type}", images=[grid1])
        self.logger.log_image(key=f"EROS_{type}", images=[grid2])

    def configure_optimizers(self) -> tuple:
        """Configure the model optimizers"""
        pg = param_groups_seg(self.student, self.lr)
        optimizer = AdamW(pg, lr=self.lr, weight_decay=0.05)

        scheduler1 = LinearLR(optimizer, start_factor=1e-6, total_iters=1500)
        scheduler2 = PolynomialLR(optimizer, total_iters=(self.trainer.estimated_stepping_batches - 1500), power=1.0)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[1500]   # switch to scheduler2 at step 1500
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
