import lightning as l
import torch.nn as nn
import torch
import torchmetrics
import numpy as np
from thunderseg.blocks.unet import AttentionUNet
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import rasterio as rio
from pycocotools.coco import COCO

from thunderseg.utils.tool import coco_to_binary_mask

class ATTUNet(l.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = AttentionUNet(in_channels=5)
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        metrics = torchmetrics.MetricCollection({
            'f1': torchmetrics.F1Score(task='binary', threshold=0.5),
            'precision': torchmetrics.Precision(task='binary', threshold=0.5),
            'recall': torchmetrics.Recall(task='binary', threshold=0.5),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.small_obj_weight = 3

    def forward(self, x):
        return self.model(x)   

    def _calculate_object_size_weights(self, mask):
        obj_sizes = torch.sum(mask, dim=(2,3))
        return torch.where(obj_sizes < 100, self.small_obj_weight, 1.0)

    def training_step(self, batch, batch_idx):
        if not batch:
            return None
        x, y = batch
        y_hat = self(x)
        weights = self._calculate_object_size_weights(y)
        bce_loss = self.loss_fn(y_hat, y.float()) * weights
        dice_loss = self.dice_loss(torch.sigmoid(y_hat), y.float())
        loss = bce_loss.mean() + dice_loss
        
        self.train_metrics(torch.sigmoid(y_hat), y.int())
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {'optimizer':optimizer, 'scheduler':{
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
        }}

    def validation_step(self, batch, batch_idx):
        if not batch:
            return None
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())
        self.val_metrics(torch.sigmoid(y_hat), y.int())
        self.log('val_loss', loss, prog_bar=True)
        return loss
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics, prog_bar=True)
    def test_step(self, batch, batch_idx):
        pass

class TrainDataset(Dataset):
    def __init__(self, train_coco:str | COCO, transform=None):
        """
        Args:
            coco : The merged json file path exported from merged_coco represent the image dataset
            transform : transform method use for image agumentation
        """
        self.train_coco = COCO(train_coco) if isinstance(train_coco, str) else train_coco
        self.train_mask = coco_to_binary_mask(self.train_coco)
        self._transform = transform

       
    def __len__(self):
        return len(self.train_coco.imgs)

    def __getitem__(self, idx):
        img_id = list(self.train_coco.imgs.keys())[idx]
        img = rio.open(self.train_coco.imgs[img_id]['file_name']).read()
        mask = self.train_mask[img_id]
        if np.any(mask == 1):
            if self._transform:
                augmented = self._transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask'].unsqueeze(0)
            return img, mask
        else:
            if idx + 1 < len(self.train_coco.imgs):
                return self.__getitem__(idx + 1)
            else:
            # If you reach the end of the dataset, return None or a default value
                return None

class ValDataset(Dataset):
    def __init__(self, valid_coco:str | COCO, transform=None):
        """
        Args:
            coco : The merged json file path exported from merged_coco represent the image dataset
            transform : transform method use for image agumentation
        """
        self.valid_coco = COCO(valid_coco) if isinstance(valid_coco, str) else valid_coco
        self.valid_mask = coco_to_binary_mask(self.valid_coco)
        self._transform = transform

       
    def __len__(self):
        return len(self.valid_coco.imgs)

    def __getitem__(self, idx):
        img_id = list(self.valid_coco.imgs.keys())[idx]
        img = rio.open(self.valid_coco.imgs[img_id]['file_name']).read()
        mask = self.valid_mask[img_id]
        if np.any(mask == 1):
            if self._transform:
                augmented = self._transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask'].unsqueeze(0)
            return img, mask
        else:
            if idx + 1 < len(self.valid_coco.imgs):
                return self.__getitem__(idx + 1)
            else:
            # If you reach the end of the dataset, return None or a default value
                return None

class ATTUNetDataModule(l.LightningDataModule):
    def __init__(self, train_coco, valid_coco, crop_size=256, batch_size=8, num_workers=4):
        super().__init__()
        self.train_coco = train_coco
        self.valid_coco = valid_coco
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = A.Compose([
            A.CropNonEmptyMaskIfExists(crop_size, crop_size, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.val_transform = A.Compose([
                    A.CropNonEmptyMaskIfExists(crop_size, crop_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

    def setup(self, stage=None):
        self.train_ds = TrainDataset(self.train_coco, self.train_transform)
        self.val_ds = ValDataset(self.valid_coco, self.val_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1).float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice