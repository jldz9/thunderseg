
from pathlib import Path
import sys
sys.path.append('/home/vscode/remotehome/DL_packages/thunderseg/src')
import torch
import numpy as np
import lightning as L
from thunderseg.core import Tile, create_project_structure, MaskRCNNModule, LoadDataModule

"""
Image preprocess module for thunderseg, include image IO, tilling
"""
import io
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import cv2
import geopandas as gpd
import lightning as L
import numpy as np
from pycocotools.coco import COCO
import rasterio as rio
from rasterio.io import DatasetReader, MemoryFile
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely import box
import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms 
from torchvision.transforms import v2 as T

from thunderseg.utils import pack_h5_list, to_pixelcoord, COCO_parser, window_to_dict, get_mean_std, assert_json_serializable
from thunderseg.core import save_h5, to_file, TrainDataset, PreditDataset
from thunderseg.model import mask_rcnn_R_101_FPN_3x

import torchvision.datasets.mnist

fpath = Path('/home/vscode/remotehome/DL_drake/Drake/Ref/Drake20220928_MS.tif')
fpth2 = Path('/home/vscode/remotehome/DL_drake/output/datasets/train/Drake20220928_MS_row5742_col5742.tif')
shp_path = Path('/home/vscode/remotehome/DL_drake/shp/shurbcrown_train.shp')
output_path = Path('/home/vscode/remotehome/DL_drake/output')
"""
structure = create_project_structure(output_path)
tiles = Tile(fpth2, output_path=structure.train, tile_size=50, buffer_size=5)
tiles.tile_image()
tiles.tile_shape(shp_path)
coco_path = tiles.to_COCO()
"""
def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
coco_path = "/home/vscode/remotehome/DL_drake/output/datasets/train/Drake20220928_MS_row5742_col5742_coco.json"
#data_module = LoadDataModule(num_workers=5,train_coco=coco_path,batch_size=2)
#model = MaskRCNNModule(num_classes=num_classes)
#trainer = L.Trainer(max_epochs=2, accelerator="gpu", devices=1)
#trainer.fit(model, data_module)
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

model = get_model_instance_segmentation(2)
dataset = TrainDataset(coco=coco_path, transform=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)