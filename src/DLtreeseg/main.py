import argparse
import sys
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import torch
import lightning as L
torch.set_float32_matmul_precision('high')
from colorama import Fore, init
init(autoreset=True)

from lightning.pytorch.loggers import TensorBoardLogger
from pycocotools.coco import COCO
from DLtreeseg.core import Tile, LoadDataModule, get_transform, MaskRCNN_RGB
from DLtreeseg.utils import merge_coco, Config, create_project_structure
from DLtreeseg._version import __version__

def create_parser():
    synopsis = 'This is a python interface for DLtreeseg program'
    name = 'DLtreeseg'
    parser = argparse.ArgumentParser(name, description=synopsis, add_help=False)
    parser.add_argument("-v", "--version", action='version', version=f'DLtreeseg {__version__}')
    parser.add_argument('-c', '--config', metavar='PATH', help='Path to config for automate process')
    parser.add_argument('--init', metavar='PATH', help='Initiliza the working dir')
    subparser = parser.add_subparsers(dest='step')
    preprocess = subparser.add_parser('preprocess', help='Preprocess raster into tiles and register dataset')
    preprocess.add_argument('-c', '--config',metavar='PATH', help='Path to config')
    train = subparser.add_parser('train', help='Train models')
    train.add_argument('-c', '--config', metavar='PATH', help='Path to config')
    return parser, preprocess, train

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
        print(f'{Fore.GREEN}Processing {r.stem}')
        train_r = Tile(fpth=r, output_path=cfg.IO.TRAIN, buffer_size=cfg.PREPROCESS.BUFFER_SIZE, tile_size=cfg.PREPROCESS.TILE_SIZE)
        train_r.tile_image(mode=cfg.PREPROCESS.MODE, shp_path=cfg.IO.TRAIN_SHP+'/train_shp.shp')
        coco_path = train_r.to_COCO(cfg.IO.TEMP + f'/{r.stem}_coco.json', **cfg.PREPROCESS.COCO_INFO.to_dict())
        train_coco_list.append(coco_path)
    merge_coco(train_coco_list, cfg.IO.ANNOTATIONS+'/train_coco.json')

    predict_rasters = list(Path(cfg.IO.PREDICT_RASTER_DIR).glob('*.tiff')) + list(Path(cfg.IO.PREDICT_RASTER_DIR).glob('*.tif'))
    predict_coco_list = []
    if len(predict_rasters) <1: 
        print(f"{Fore.YELLOW}No raster found under {cfg.IO.PREDICT_RASTER_DIR}, will skip prediction step")
        cfg.STATUS.HAS_PREDICT_DATASET = False
    else:
        for p in predict_rasters:
            predict_r = Tile(fpth=p, output_path = cfg.IO.PREDICT,buffer_size=cfg.PREPROCESS.BUFFER_SIZE, tile_size=cfg.PREPROCESS.TILE_SIZE)
            predict_r.tile_image(mode=cfg.PREPROCESS.MODE)
            coco_path = predict_r.to_COCO(cfg.IO.TEMP + f'{p.stem}_coco.json')
            predict_coco_list.append(coco_path)
        merge_coco(predict_coco_list, cfg.IO.ANNOTATIONS+'/predict_coco.json')
    return cfg

def train_step(cfg, customize_transform=None): 
    coco_train = COCO(cfg.IO.ANNOTATIONS+'/train_coco.json')
    if Path(cfg.IO.ANNOTATIONS+'/predict_coco.json').is_file():
        coco_predict = COCO(cfg.IO.ANNOTATIONS+'/predict_coco.json')
    else: 
        coco_predict = None
    if cfg.TRAIN.DATAMODULE.TRANSOFRM == 'default':
        transform = get_transform
    else: 
        transform = customize_transform
    datamodule = LoadDataModule(train_coco=coco_train, predict_coco=coco_predict,
                                batch_size=cfg.TRAIN.DATAMODULE.BATCH_SIZE,
                                num_workers=cfg.TRAIN.DATAMODULE.NUM_WORKERS,
                                transform=transform
                                )
    logger = TensorBoardLogger(save_dir=cfg.IO.RESULT+'/logs', name='MyTrain')
    if cfg.TRAIN.MODEL == 'maskrcnn_rgb':
        model = maskrcnn_rgb_step(cfg.TRAIN.MASKRCNNRGB.NUM_CLASSES, float(cfg.TRAIN.MASKRCNNRGB.LEARNING_RATE))

    trainer = L.Trainer(logger=logger, 
                        accelerator=cfg.TRAIN.ACCELERATOR,
                        devices=cfg.TRAIN.DEVICES,
                        max_epochs=cfg.TRAIN.MAX_EPOCHS)
    trainer.fit(model, datamodule)
    cfg.STATUS.TRAINED = True
    return cfg

def maskrcnn_rgb_step(num_classes, learning_rate):
    model = MaskRCNN_RGB(num_classes=num_classes, learning_rate=learning_rate)
    return model

def main(iargs=None):
    _default_cfg_path = Path(__file__).parent / 'utils/config.toml'
    parser, preprocess_parser, train_parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if len(iargs) <1:
        parser.print_help()
        sys.exit(1)
    if inps.init:
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

if __name__ == '__main__':
    main(sys.argv[1:])
