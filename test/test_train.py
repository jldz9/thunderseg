
from pathlib import Path
import os
os.environ.get('PROJ_LIB')
os.environ['PROJ_LIB'] = '/home/jldz9/miniconda3/envs/DL/share/proj'
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import sys
sys.path.append('/home/jldz9/DL/DL_packages/DLtreeseg/src')
import os
import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage

from DLtreeseg.core import MultiBand_Trainer
from DLtreeseg.core import Tile, create_project_structure, modify_conv1_weights

fpath = Path('/home/jldz9/DL/DL_drake/Drake/Ref/Drake20220928_MS.tif')
train_shp_path = Path('/home/jldz9/DL/DL_drake/shp/shurbcrown_train.shp')
val_shp_path = Path('/home/jldz9/DL/DL_drake/shp/shurbcrown_val1.shp')
output_path = Path('/home/jldz9/DL/output')
outputdir = create_project_structure(output_path)
train = Tile(fpth=fpath, output_path=outputdir.train, buffer_size=20, tile_size=100)
train.tile_image()
train.tile_shape(train_shp_path)
train.to_COCO(outputdir.train / 'shurbtrain.json', 'shurbtrain')
val = Tile(fpth=fpath, output_path=outputdir.val, buffer_size=20, tile_size=100)
val.tile_image()
val.tile_shape(val_shp_path)
val.to_COCO(outputdir.val / 'shurbval.json', 'shurbval')
"""
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("shurbtrain",)
cfg.DATASETS.TEST = ("shurbval",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
cfg.INPUT.FORMAT = "BGREN"
cfg.MODEL.PIXEL_MEAN = [0.02572454698383808, 0.036039356142282486, 0.04472881183028221, 0.06555074453353882, 0.09009517729282379]
cfg.MODEL.PIXEL_STD = [0.02774825319647789, 0.038967959582805634, 0.049449577927589417, 0.06904447078704834, 0.0936022475361824]
cfg.MODEL.BACKBONE.NAME = "build_custom_resnet_backbone"

cfg.INPUT.IN_CHANNELS = 5
model = build_model(cfg)
trainer = MultiBand_Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
"""

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("shurbtrain",)
cfg.DATASETS.TEST = ("shurbval",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.OUTPUT_DIR = outputdir.result.as_posix()
np.bool = np.bool_
#trainer = DefaultTrainer(cfg) 
#trainer.resume_or_load(resume=False)
#trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)
import random
import cv2
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.model_zoo import get_config
from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
"""
dataset_dicts = DatasetCatalog.get("shurbval")
for d in dataset_dicts:    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("shurbval"), 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('image',out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    """


dataset_dicts = DatasetCatalog.get("shurbtrain")

for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('shurbtrain'), scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite(f'{Path(d["file_name"]).parent}/{Path(d["file_name"]).stem}_val.{Path(d["file_name"]).suffix}',out.get_image()[:, :, ::-1])
    
