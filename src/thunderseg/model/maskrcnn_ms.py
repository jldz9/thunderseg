#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thunderseg built in instance segementation Maskrcnn_ms module
"""
import copy
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import warnings
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L
import numpy as np
import rasterio as rio
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
torch.set_float32_matmul_precision('high')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from thunderseg.utils import  bbox_from_mask

class MaskRCNN_MS(L.LightningModule):
    def __init__(self, model, num_classes: int = 2, learning_rate: float = 1e-3):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
