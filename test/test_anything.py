from pathlib import Path
import sys
sys.path.append('/home/vscode/remotehome/DL_packages/thunderseg/src')
from torchvision.transforms import v2 as T
from thunderseg.core import MaskRCNN_RGB, LoadDataModule, get_transform, train_model
import torch
import lightning as L
from thunderseg.core import TrainDataset, Postprocess
from thunderseg.utils import check_image_target
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from thunderseg.core.preprocess import Tile
import rasterio as rio
from pycocotools.coco import COCO
from lightning.pytorch.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('high')
from lightning.pytorch.tuner import Tuner

fpath = Path('/home/vscode/remotehome/DL_drake/Drake/Ref/Drake20220928_MS.tif')
shp_path = Path('/home/vscode/remotehome/DL_drake/shp/shp_20220928.shp')
output_path = Path('/home/vscode/remotehome/DL_drake/output')
coco_path = '/home/vscode/remotehome/DL_drake/output/datasets/annotations/train_coco.json'
"""
a = Tile(fpth=fpath, output_path=output_path, buffer_size=0, tile_size=100)
a.tile_image()
a.tile_shape(shp_path)
a.to_COCO('/home/vscode/remotehome/DL_drake/demo.json')

"""

"""

data_module = LoadDataModule(num_workers=5,train_coco=coco_path,batch_size=2)
model = MaskRCNNLightning(num_classes=2)
trainer = L.Trainer(max_epochs=1, accelerator="gpu", devices=1)
trainer.fit(model, data_module)
"""
#train_model(coco_path=coco_path, model_name='my_model', num_classes=1, batch_size=5, learning_rate=0.0001, num_epochs=20)

model = MaskRCNN_RGB()
logger = TensorBoardLogger("shurb", name='shurb_model')
dataset = LoadDataModule(train_coco=coco_path, batch_size=5, num_workers=15)
trainer = L.Trainer(logger=logger, accelerator='gpu', devices=1, max_epochs=100)
#tuner = Tuner(trainer)
##tuner.scale_batch_size(dataset, mode='power')
trainer.fit(model, dataset)
'''
fpath = Path('/home/vscode/remotehome/DL_drake/output/')
checkpoint = '/workspaces/thunderseg/lightning_logs/version_2/checkpoints/epoch=9-step=190.ckpt'
coco_path = '/home/vscode/remotehome/DL_drake/demo.json'
predict = '/home/vscode/remotehome/DL_drake/predict.pkl'
'''
'''
coco = COCO(coco_path)
all_image_ids = coco.getImgIds()
all_images = coco.loadImgs(all_image_ids)
tifs = [img['file_name'] for img in all_images]
model = MaskRCNN_RGB.load_from_checkpoint(checkpoint)
model.to('cpu')
model.eval()
images = []
coco = COCO(coco_path)
mean=coco.dataset['info']['pixel_mean']
std=coco.dataset['info']['pixel_std']
for i in tifs:
    with rio.open(i) as f:
        img = f.read()
        image = np.transpose(img, (1,2,0)) 
        image = get_transform(image, train=False, mean=mean, std=std)
        images.append(image)
with torch.no_grad():
    y_hat = model(images)
'''
'''
import pickle 
with open(predict, "rb") as file:
    y_hat = pickle.load(file)
result = Postprocess(coco_path, y_hat,'/home/vscode/remotehome/DL_drake/')
result.mask_rcnn_postprocess()


for idx, img in enumerate(tifs):
    savepath = img.with_name(f'{img.stem}_predict.png')
    check_image_target(images[idx], y_hat[idx], savepath=savepath)
'''