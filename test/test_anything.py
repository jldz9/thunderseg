from pathlib import Path
import sys
sys.path.append('/home/vscode/remotehome/DL_packages/DLtreeseg/src')
from torchvision.transforms import v2 as T
from DLtreeseg.core import MaskRCNN_RGB, LoadDataModule, get_transform, train_model
import torch
import lightning as L
from DLtreeseg.core import TrainDataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2




fpth2 = Path('/home/vscode/remotehome/DL_drake/output/datasets/train/Drake20220928_MS_row5742_col5742.tif')
shp_path = Path('/home/vscode/remotehome/DL_drake/shp/shurbcrown_train.shp')
output_path = Path('/home/vscode/remotehome/DL_drake/output')
coco_path = "/home/vscode/remotehome/DL_drake/output/datasets/train/Drake20220928_MS_row5742_col5742_coco.json"
"""
def get_transform(train=True):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

data_module = LoadDataModule(num_workers=5,train_coco=coco_path,batch_size=2)
model = MaskRCNNLightning(num_classes=2)
trainer = L.Trainer(max_epochs=1, accelerator="gpu", devices=1)
trainer.fit(model, data_module)
"""
#train_model(coco_path=coco_path, model_name='my_model', num_classes=1, batch_size=5, learning_rate=0.0001, num_epochs=20)

model = MaskRCNN_RGB()
dataset2 = TrainDataset(coco_path)
data = dataset2[2]
dataset = LoadDataModule(train_coco=coco_path, batch_size=5)
trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=10)
trainer.fit(model, dataset)

