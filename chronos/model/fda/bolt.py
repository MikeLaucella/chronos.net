import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR, LinearLR, SequentialLR
from torchmetrics.classification import (Accuracy, JaccardIndex)
from lightning import LightningModule
from chronos.model.segformer.lr_decay import param_groups_seg


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

def get_counts():
    return torch.tensor([
        1e-6,       #background clamp to avoid zero division
        458867493,  #parking lot
        951703234,  #building
        2166024527, #grass
        1249752222, #road
        1596053439, #tree
        65039489,   #agriculture
        271761047,  #water
        160507667,  #barren
        25414388    #other
    ], dtype=torch.float32)

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


class SegmentationMetrics(nn.Module):
    """Segmentation metrics for evaluation."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.acc = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=0)
        self.iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0)
        self.loss = F.cross_entropy
        self.class_weights = get_weights()  # Class imbalance weights

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
        self.class_weights = self.class_weights.to(y.device)
        loss = self.loss(y_hat, y, ignore_index=0, weight=self.class_weights).float()

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
        ce = metrics['loss_ce']
        ent = metrics['loss_ent']

        if batch_idx % self.log_step == 0: # Log every n steps for validation/test
            #self.hook.log(f"{mode}_acc_step", acc, on_epoch=False, on_step=True, prog_bar=True, batch_size=batch_size)
            #self.hook.log(f"{mode}_iou_step", iou, on_epoch=False, on_step=True, prog_bar=True, batch_size=batch_size)
            self.hook.log(f"{mode}_loss_step", loss, on_epoch=False, on_step=True, prog_bar=True, batch_size=batch_size)

        # combined
        self.hook.log(f"{mode}/acc", acc, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.hook.log(f"{mode}/iou", iou, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.hook.log(f"{mode}/loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.hook.log(f"{mode}/ce", ce, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.hook.log(f"{mode}/ent", ent, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)


class FDABolt(LightningModule):
    def __init__(self,
                 model: nn.Module,
                 classes: list,
                 lr: float=6e-5,
                 beta=0.01,
                 ent_w=0.005,
                 eta=2.0,
                 eps=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.beta = beta
        self.ent_w = ent_w  # Weight of entropy loss
        self.eta = eta      # Charbonnier exponent (>0.5 penalizes uncertainty more)
        self.eps_sq = eps ** 2
        self.div_w = 0.5  # Weight of diversity loss
        self.classes = classes
        self._logger = SegmentationLogger(self)
        self._metrics = SegmentationMetrics(num_classes=len(classes))
        self.class_weights = get_weights()  # Class imbalance weights
        self.class_counts = get_counts()
        self.target_dist = self.class_counts / self.class_counts.sum()
        self.ent_switch = 50_000 / 16 # after 50k samples, start applying entropy loss (adjusted for batch size of 16)

    def forward(self, inp: dict) -> torch.Tensor:
        """Perform forward pass"""
        img1 = inp['naip_hist']
        mask = inp['masks']

        with torch.amp.autocast(self.device.type):
            img1_y_hat = self.model(img1)
  
        out = self._metrics.score(mask, img1_y_hat)
        return out

    def _step(self, batch, batch_idx):
        # 'batch' should be a dict from a CombinedLoader or similar
        x_s = batch["naip_hist"]
        y_s = batch["eros_hist"]
        x_t = batch["masks"]

        with torch.amp.autocast(self.device.type):
            # 2. Task Loss: CE on Adapted Source
            logits_s = self.model(x_s)
            out = self._metrics.score(x_t, logits_s)
            loss_ce = out['loss']

            if self.global_step > self.ent_switch:
                # 3. Regularization: Charbonnier Entropy on Target
                logits_t = self.model(y_s)
                probs_t = F.softmax(logits_t, dim=1)
                log_probs_t = F.log_softmax(logits_t, dim=1)

                # Shannon Entropy
                ent_map = -torch.sum(probs_t * log_probs_t, dim=1)
                ent_map = ent_map / 2.30258509299  # Normalize by log(num_classes) to get [0, 1]
                #ent_map = ent_map * pred_weights  # Weight by class imbalance

                # Charbonnier Penalty: (entropy^2 + eps^2)^eta
                char_ent = (ent_map**2 + self.eps_sq) ** self.eta
                loss_ent = torch.mean(char_ent).float()
            else:
                loss_ent = torch.tensor(0.0).to(self.device)

            total_loss = loss_ce + (self.ent_w * loss_ent)
            out['loss_ce'] = loss_ce
            out['loss_ent'] = loss_ent
            out['loss'] = total_loss

        return out

    def training_step(self, batch, batch_idx):
        # 'batch' should be a dict from a CombinedLoader or similar
        out = self._step(batch, batch_idx)
        self._logger.train(out, batch_idx)
        self.last_train_batch = batch
        return out['loss']

    def validation_step(self, batch, batch_idx):
        out = self._step(batch, batch_idx)
        self._logger.validation(out, batch_idx)
        self.last_val_batch = batch
        return out['loss']

    def test_step(self, batch, batch_idx):
        out = self._step(batch, batch_idx)
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

        # 2. Get Model Predictions
        self.eval()
        with torch.no_grad():
            # SegFormer often outputs logits at 1/4 res; upsample to input size
            logits_naip = self.model(naip)
            logits_eros = self.model(eros)
            preds_naip = torch.argmax(logits_naip, dim=1) # [B, H, W]
            preds_eros = torch.argmax(logits_eros, dim=1) # [B, H, W]

        naip_mean = torch.tensor((0.6493820041822563, 0.6493820041822563, 0.6493820041822563), device=self.device).view(1, 3, 1, 1)
        naip_std = torch.tensor((0.17010567701149337, 0.17010567701149337, 0.17010567701149337), device=self.device).view(1, 3, 1, 1)

        # 3. Prepare Visual Tensors [0, 1]
        naip_vis = denorm(naip, naip_mean, naip_std)
        eros_vis = denorm(eros, naip_mean, naip_std)

        gt_vis = color_label(labels.squeeze(1))
        naip_pred_vis = color_label(preds_naip)
        eros_pred_vis = color_label(preds_eros)

        # 4. Concatenate for a 4x4 Grid [Rows: NAIP, GT, Pred, EROS]
        # combined shape: [16, 3, H, W]
        combined1 = torch.cat([naip_vis, gt_vis, naip_pred_vis], dim=0)
        grid1 = to_wandb_uint8(torchvision.utils.make_grid(combined1, nrow=4))

        combined2 = torch.cat([eros_vis, gt_vis, eros_pred_vis], dim=0)
        grid2 = to_wandb_uint8(torchvision.utils.make_grid(combined2, nrow=4))

        # 5. Log to WandB / TensorBoard
        self.logger.log_image(key=f"NAIP_{type}", images=[grid1])
        self.logger.log_image(key=f"EROS_{type}", images=[grid2])

    def configure_optimizers(self) -> tuple:
        """Configure the model optimizers"""
        pg = param_groups_seg(self.model, self.lr)
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
