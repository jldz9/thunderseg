#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IO related module for DLtreeseg
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio as rio
import shapefile
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# save hdf5 to local drive
def save_h5(save_path:str, data:np.ndarray, attrs:dict = None, **kwarg):
    """
    Use of h5py to storage data to local disk. **kwarg should contains packed binary data from
    function pack_h5_list.
    """
    save_path = Path(save_path)
    if save_path.is_file():
        while True:
            rmfile = input(f'File {save_path.name} exist, do you want to open this dataset? Y(es)/O(verwrite)/C(ancel) ')
            if rmfile.lower() == 'y':
                break
            if rmfile.lower() == 'o':
                save_path.unlink()
                break
            if rmfile.lower() == 'c':
                sys.exit()
    with h5py.File(save_path, 'r+') as hf:
        if save_path.stem in hf:
            while True:
                rmgroup = input(f'Group {save_path.stem} exist, do you want to overwrite? Y(es)/N(o) ')
                if rmgroup.lower() == 'y':
                    del hf[save_path.stem]
                    break
                if rmgroup.lower() == 'n':
                    sys.exit()
        grp = hf.create_group(save_path.stem)
        grp.create_dataset('data', data=data)
        if attrs is not None:
            for key, value in attrs.items():
                grp.attrs[key] = value
        if kwarg:
            for key, value in kwarg.items():
                grp.create_dataset(key, data=value)
    print(f'{save_path} saved!')

# save file to local gis
def save_gis(path_to_file:str, data:np.ndarray, profile):
    with rio.open(path_to_file, 'w', **profile) as dst:
        dst.write(data)
        sys.stdout.write(f'\rExport {path_to_file.name}')
        sys.stdout.flush()

def setup_cfg(
    base_model: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    trains=("trees_train", ),
    tests=("trees_val", ),
    update_model=None,
    workers=2,
    ims_per_batch=2,
    gamma=0.1,
    backbone_freeze=3,
    warm_iter=120,
    momentum=0.9,
    batch_size_per_im=1024,
    base_lr=0.0003389,
    weight_decay=0.001,
    max_iter=1000,
    num_classes=1,
    eval_period=100,
    out_dir="./train_outputs",
    resize=True,
):
    """Set up config object # noqa: D417.
    function adapted from detectree2: https://github.com/PatBall1/detectree2 train.py
    Args:
        base_model: base pre-trained model from detectron2 model_zoo
        trains: names of registered data to use for training
        tests: names of registered data to use for evaluating models
        update_model: updated pre-trained model from detectree2 model_garden
        workers: number of workers for dataloader
        ims_per_batch: number of images per batch
        gamma: gamma for learning rate scheduler
        backbone_freeze: backbone layer to freeze
        warm_iter: number of iterations for warmup
        momentum: momentum for optimizer
        batch_size_per_im: batch size per image
        base_lr: base learning rate
        weight_decay: weight decay for optimizer
        max_iter: maximum number of iterations
        num_classes: number of classes
        eval_period: number of iterations between evaluations
        out_dir: directory to save outputs
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    cfg.DATASETS.TRAIN = trains
    cfg.DATASETS.TEST = tests
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.GAMMA = gamma
    cfg.MODEL.BACKBONE.FREEZE_AT = backbone_freeze
    cfg.SOLVER.WARMUP_ITERS = warm_iter
    cfg.SOLVER.MOMENTUM = momentum
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = batch_size_per_im
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.SOLVER.BASE_LR = base_lr
    cfg.OUTPUT_DIR = out_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if update_model is not None:
        cfg.MODEL.WEIGHTS = update_model
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.RESIZE = resize
    cfg.INPUT.MIN_SIZE_TRAIN = 1000
    return cfg

def shp2list(fpth:str):
    fpth = Path(fpth)
    sf = shapefile.Reader(fpth)
    flat_coords = []
    for shape in sf.shapes():
        coords = shape.points
        flat_coords.append([coord for pair in coords for coord in pair])
    return flat_coords