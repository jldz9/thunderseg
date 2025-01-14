#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point of thunderseg
"""
import argparse
import sys
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import torch
torch.set_float32_matmul_precision('high')
from colorama import Fore, init
init(autoreset=True)

from pycocotools.coco import COCO
from thunderseg.core import Tile, Train, Postprocess
from thunderseg.utils import merge_coco, Config, create_project_structure
from thunderseg.model import maskrcnn_rgb
from thunderseg._version import __version__

def create_parser():
    synopsis = 'This is a python interface for thunderseg program'
    name = 'thunderseg'
    parser = argparse.ArgumentParser(name, description=synopsis, add_help=False)
    parser.add_argument("-v", "--version", action='version', version=f'thunderseg {__version__}')
    parser.add_argument('-c', '--config', metavar='PATH', help='Path to config for automate process')
    parser.add_argument('--init', metavar='PATH', help='Initiliza the working dir')
    subparser = parser.add_subparsers(dest='step')
    preprocess = subparser.add_parser('preprocess', help='Preprocess raster into tiles and register dataset')
    preprocess.add_argument('-c', '--config',metavar='PATH', help='Path to config')
    train = subparser.add_parser('train', help='Train models')
    train.add_argument('-c', '--config', metavar='PATH', help='Path to config')
    predict = subparser.add_parser('predict', help='Predict using trained model')
    predict.add_argument('-c','--config', metavar='PATH', help='Path to config')
    predict.add_argument('--ckpt', metavar='PATH', help='The saved checkpoint path, if not provided will try to search the latest one')
    return parser, preprocess, train, predict

def find_config(config):
    cfg_path = Path(config)
    if not cfg_path.is_file():
        search_cfg = list(cfg_path.glob('*.toml'))
        if len(search_cfg) == 0:
            raise FileNotFoundError(f'Cannot find any config file under {config}')
        if len(search_cfg) > 1:
            raise ValueError(f'Duplicate config files found under {config}')
        return search_cfg[0].resolve()
    else:
        return cfg_path.resolve()

def preprocess_step(cfg):
    """Step to prepare everything for project loop
    """
    train_rasters = list(Path(cfg.IO.TRAIN_RASTER_DIR).glob('*.tiff')) + list(Path(cfg.IO.TRAIN_RASTER_DIR).glob('*.tif'))
    assert len(train_rasters) > 0, f"Not able to find any rasters under {cfg.IO.TRAIN_SHP_DIR}"
    train_shps = list(Path(cfg.IO.TRAIN_SHP_DIR).glob('*.shp')) #TODO might need to support more formats
    assert len(train_shps) > 0, f"Not able to find any shapefiles under {cfg.IO.TRAIN_SHP_DIR}"
    shps = [gpd.read_file(shp) for shp in train_shps]
    merged_shp = pd.concat(shps, ignore_index=True)
    merged_shp.drop_duplicates(inplace=True)
    merged_shp.to_file(cfg.IO.TRAIN_SHP+'/train_shp.shp')
    train_coco_list = []
    for r in train_rasters:
        print(f'{Fore.GREEN}Processing train raster {r.stem}')
        train_r = Tile(fpth=r, output_path=cfg.IO.TRAIN, buffer_size=cfg.PREPROCESS.BUFFER_SIZE, tile_size=cfg.PREPROCESS.TILE_SIZE)
        train_r.tile_image(mode=cfg.PREPROCESS.MODE, shp_path=cfg.IO.TRAIN_SHP+'/train_shp.shp')
        coco_path = train_r.to_COCO(cfg.IO.TEMP + f'/{r.stem}_coco.json', **cfg.PREPROCESS.COCO_INFO.to_dict())
        train_coco_list.append(coco_path)
    merge_coco(train_coco_list, cfg.IO.ANNOTATIONS+'/train_coco.json')
    return cfg

def train_step(cfg, customize_transform=None): 
    coco_train = COCO(cfg.IO.ANNOTATIONS+'/train_coco.json')
    if cfg.TRAIN.MODEL == 'maskrcnn_rgb':
        train = Train(maskrcnn_rgb.MaskRCNN_RGB,
                      maskrcnn_rgb.LoadDataModule,
                      coco_train,
                      predict_coco=None,
                      batch_size=cfg.TRAIN.BATCH_SIZE,
                      num_workers=cfg.TRAIN.NUM_WORKERS,
                      num_classes=cfg.TRAIN.NUM_CLASSES,
                      learning_rate = cfg.TRAIN.LEARNING_RATE,
                      max_epochs = cfg.TRAIN.MAX_EPOCHS,
                      save_path = cfg.IO.RESULT,
                                                  )# TODO add path to checkpointpath and also add callbacks
        train.fit()
    return cfg
def predict_step(cfg):
    assert Path(cfg.IO.ANNOTATIONS+'/train_coco.json').is_file, f"Cannot find {cfg.IO.ANNOTATIONS+'/train_coco.json'}, did you run preprocess step?"
    coco_train = COCO(cfg.IO.ANNOTATIONS+'/train_coco.json')
    if Path(cfg.IO.ANNOTATIONS+'/predict_coco.json').is_file():
        predict_coco = COCO(cfg.IO.ANNOTATIONS+'/predict_coco.json')
    else:
        predict_rasters = list(Path(cfg.IO.PREDICT_RASTER_DIR).glob('*.tiff')) + list(Path(cfg.IO.PREDICT_RASTER_DIR).glob('*.tif'))
        assert len(predict_rasters) > 0, f"Not able to find any rasters under {cfg.IO.PREDICT_SHP_DIR}"
        predict_coco_list = []
        for p in predict_rasters:
            print(f'{Fore.GREEN}Processing predict raster{p.stem}')
            predict_r = Tile(fpth=p, output_path = cfg.IO.PREDICT, buffer_size=50, tile_size=cfg.PREPROCESS.TRANSFORM.RANDOM_CROP_HEIGHT-100, tile_mode='pixel')
            predict_r.tile_image(mode=cfg.PREPROCESS.MODE)
            coco_path = predict_r.to_COCO(cfg.IO.TEMP + f'{p.stem}_coco.json')
            predict_coco_list.append(coco_path)
        merge_coco(predict_coco_list, cfg.IO.ANNOTATIONS+'/predict_coco.json')
        predict_coco = COCO(cfg.IO.ANNOTATIONS+'/predict_coco.json')
        
    if cfg.TRAIN.MODEL == 'maskrcnn_rgb':
        predict = Train(maskrcnn_rgb.MaskRCNN_RGB,
                      maskrcnn_rgb.LoadDataModule,
                      coco_train,
                      predict_coco=predict_coco,
                      batch_size=cfg.TRAIN.BATCH_SIZE,
                      num_workers=cfg.TRAIN.NUM_WORKERS,
                      num_classes=cfg.TRAIN.NUM_CLASSES,
                      learning_rate = cfg.TRAIN.LEARNING_RATE,
                      max_epochs = cfg.TRAIN.MAX_EPOCHS,
                      save_path = cfg.IO.RESULT,
                      ckpt_path=cfg.IO.CHECKPOINT
                                                  )
        result = predict.predict() 
        postprocess = Postprocess(predict_coco, result, cfg.IO.RESULT)
        postprocess.mask_rcnn_postprocess()
        return cfg
        
def main(iargs=None):
    _default_cfg_path = Path(__file__).parent / 'utils/config.toml'
    parser, preprocess_parser, train_parser, predict_parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if len(iargs) <1:
        parser.print_help()
        sys.exit(1)
    if inps.init:
        Path(inps.init).mkdir(exist_ok=True, parents=True)
        shutil.copy(_default_cfg_path, inps.init)
        print(f'{Fore.GREEN}Config copied under {Path(inps.init).resolve()}')
    if inps.step == 'preprocess':   
        if inps.config is None:
            preprocess_parser.print_help()
            sys.exit(1)
        cfg_path = find_config(inps.config)
        cfg = Config(config_path=cfg_path)
        directories = create_project_structure(cfg.IO.WORKDIR)
        cfg.IO.update(directories)
        cfg = preprocess_step(cfg)
        cfg.to_file(cfg_path)
    if inps.step == 'train': #TODO add support of customize_transform
        if inps.config is None:
            train_parser.print_help()
            sys.exit(1)
        cfg_path = find_config(inps.config)
        cfg = Config(config_path=cfg_path)
        cfg = train_step(cfg)
        cfg.to_file(cfg_path)
    if inps.step == 'predict':
        if inps.config is None:
            predict_parser.print_help()
            sys.exit(1)
        cfg_path = find_config(inps.config)
        cfg = Config(config_path=cfg_path)
        if inps.ckpt is not None:
            cfg.IO.CHECKPOINT = inps.ckpt
        else:
            ckpt_paths = list(Path(cfg.IO.RESULT).rglob('*.ckpt'))
            assert ckpt_paths, f'not able to find any checkpoint output under {cfg.IO.RESULT}'
            latest_ckpt = max(ckpt_paths, key=lambda f: f.stat().st_mtime)
            cfg.IO.CHECKPOINT = latest_ckpt.resolve().as_posix()
        cfg = predict_step(cfg)
        cfg.to_file(cfg_path)

if __name__ == '__main__':
    main(sys.argv[1:])
