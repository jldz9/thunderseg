#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point of thunderseg
"""
import argparse
import importlib.util
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path


import geopandas as gpd
import pandas as pd
import torch
torch.set_float32_matmul_precision('high')
from tensorboard import program
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
    parser = argparse.ArgumentParser(name, description=synopsis, add_help=True)
    parser.add_argument("-v", "--version", action='version', version=f'thunderseg {__version__}')
    parser.add_argument('-g', '--get_config', metavar='PATH', help='Get the default config file to given path')
    parser.add_argument('-e', '--example_data', metavar='PATH', help='Download example dataset to ~/thunderseg_example_data')
    subparser = parser.add_subparsers(dest='step')
    
    preprocess = subparser.add_parser('preprocess', help='Preprocess raster into tiles and register dataset')
    preprocess.add_argument('-c', '--config',metavar='PATH', help='Path to config')
    preprocess.set_defaults(func=preprocess_step)

    train = subparser.add_parser('train', help='Train models')
    train.add_argument('-c', '--config', metavar='PATH', help='Path to config')
    train.set_defaults(func=train_step)

    predict = subparser.add_parser('predict', help='Predict using trained model')
    predict.add_argument('-c','--config', metavar='PATH', help='Path to config')
    predict.add_argument('--ckpt', metavar='PATH', help='The saved checkpoint path, if not provided will try to search the latest one')
    predict.set_defaults(func=predict_step)
    return parser

def download_example_data(download_path=None):
    if importlib.util.find_spec('gdown') is None:
        print(f'{Fore.RED}gdown is not installed, trying to install')
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'gdown'])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install gdown. Error: {e}")
    import gdown
    url = 'https://drive.google.com/drive/folders/1Pu0oinH4SrOyVKjY5jxHqtUAg7pxoILj?usp=drive_link'
    if download_path is None:
        dest_folder = Path.home() / 'thunderseg_example_data'
        dest_folder.mkdir(exist_ok=True)
    else: 
        dest_folder = Path(download_path) / 'thunderseg_example_data'
        dest_folder.mkdir(exist_ok=True)
    print(f'{Fore.GREEN}downloading exampling data to {dest_folder}')
    gdown.download_folder(url, output=dest_folder.as_posix(), quiet=False)

def check_args(func):
    """Decorator to check if the arguments are correct
    """
    def wrapper(args):
        if args.config is None:
            print(f'{Fore.RED}A valid PATH is required after -c or --config, use --help for more information')
            sys.exit(1)
        return func(args)
    return wrapper

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
    
@check_args
def preprocess_step(args):
    """Step to prepare everything for project loop
    """
    cfg_path = find_config(args.config)
    cfg = Config(config_path=cfg_path)
    directories = create_project_structure(cfg.IO.WORKDIR)
    cfg.IO.update(directories)
    train_rasters = list(Path(cfg.IO.TRAIN_RASTER_DIR).glob('*.tiff')) + list(Path(cfg.IO.TRAIN_RASTER_DIR).glob('*.tif'))
    assert len(train_rasters) > 0, f"Not able to find any rasters under {cfg.IO.TRAIN_SHP_DIR}"
    train_shps = list(Path(cfg.IO.TRAIN_SHP_DIR).glob('*.shp')) #TODO might need to support more formats
    assert len(train_shps) > 0, f"Not able to find any shapefiles under {cfg.IO.TRAIN_SHP_DIR}"
    shps_t = [gpd.read_file(shp) for shp in train_shps]
    merged_shp = pd.concat(shps_t, ignore_index=True)
    merged_shp.drop_duplicates(inplace=True)
    merged_shp.to_file(cfg.IO.TRAIN_SHP+'/train_shp.shp')
    valid_shps = list(Path(cfg.IO.VALID_SHP_DIR).glob('*.shp')) #TODO might need to support more formats
    if len(valid_shps) > 0:
        print(f'{Fore.GREEN}Found valid shapefiles, will merge them')
        shps_v = [gpd.read_file(shp) for shp in valid_shps]
        merged_shp = pd.concat(shps_v, ignore_index=True)
        merged_shp.drop_duplicates(inplace=True)
        merged_shp.to_file(cfg.IO.VAL_SHP+'/valid_shp.shp')
        cfg.TRAIN.SEPE_VAL = True
    else:
        print(f'{Fore.YELLOW}No valid shapefiles found, will split porttion of train shapes for validation')
        cfg.TRAIN.SEPE_VAL = False
    train_coco_list = []
    valid_coco_list = []
    for r in train_rasters:
        print(f'{Fore.GREEN}Processing train raster {r.stem}')
        train_r = Tile(fpth=r, output_path=cfg.IO.TRAIN, buffer_size=cfg.PREPROCESS.BUFFER_SIZE, tile_size=cfg.PREPROCESS.TILE_SIZE)
        train_r.tile_image(mode=cfg.PREPROCESS.MODE, shp_path=cfg.IO.TRAIN_SHP+'/train_shp.shp')
        coco_path = train_r.to_COCO(cfg.IO.TEMP + f'/{r.stem}_coco.json', **cfg.PREPROCESS.COCO_INFO.to_dict())
        train_coco_list.append(coco_path)
        if cfg.TRAIN.SEPE_VAL:
            print(f'{Fore.GREEN}Processing valid raster {r.stem}')
            train_r.tile_shape(shp_path=cfg.IO.VAL_SHP+'/valid_shp.shp')
            coco_path_v = train_r.to_COCO(cfg.IO.TEMP + f'/{r.stem}_coco_valid.json', **cfg.PREPROCESS.COCO_INFO.to_dict())
            valid_coco_list.append(coco_path_v)
    merge_coco(train_coco_list, cfg.IO.ANNOTATIONS+'/train_coco.json')
    merge_coco(valid_coco_list, cfg.IO.ANNOTATIONS+'/valid_coco.json')
    cfg.to_file(cfg_path)

@check_args
def train_step(args): 
    cfg_path = find_config(args.config)
    cfg = Config(config_path=cfg_path)
    coco_train = COCO(cfg.IO.ANNOTATIONS+'/train_coco.json')
    if cfg.TRAIN.SEPE_VAL:
        coco_valid = COCO(cfg.IO.ANNOTATIONS+'/valid_coco.json')
    log_dir = cfg.IO.RESULT
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    try:
        success=webbrowser.open_new_tab(url)
    except webbrowser.Error as e:
        print(f"Failed to open TensorBoard in browser. Error: {e}")
    try: 
        subprocess.run(["xdg-open", url], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
         print(f"Failed to open TensorBoard in browser. xdg-open is might not be installed in the system")
    
    print(f"{Fore.GREEN}TensorBoard is running at {url}, \nIf you are using the headless environment, you may check logs at {log_dir} later")
    if cfg.TRAIN.MODEL == 'maskrcnn_rgb':
        train = Train(maskrcnn_rgb.MaskRCNN_RGB,
                      maskrcnn_rgb.LoadDataModule,
                      train_coco=coco_train,
                      validate_coco=coco_valid,
                      predict_coco=None,
                      batch_size=cfg.TRAIN.BATCH_SIZE,
                      num_workers=cfg.TRAIN.NUM_WORKERS,
                      num_classes=cfg.TRAIN.NUM_CLASSES,
                      learning_rate = cfg.TRAIN.LEARNING_RATE,
                      max_epochs = cfg.TRAIN.MAX_EPOCHS,
                      save_path = cfg.IO.RESULT,
                                                  )# TODO add path to checkpointpath and also add callbacks
        train.fit()

    cfg.to_file(cfg_path)

@check_args
def predict_step(args):
    cfg_path = find_config(args.config)
    cfg = Config(config_path=cfg_path)
    
    # If checkpoint is not provided, try to find the latest one
    if args.ckpt is not None:
            cfg.IO.CHECKPOINT = args.ckpt
    else:
            ckpt_paths = list(Path(cfg.IO.RESULT).rglob('*.ckpt'))
            assert ckpt_paths, f'not able to find any checkpoint output under {cfg.IO.RESULT}'
            latest_ckpt = max(ckpt_paths, key=lambda f: f.stat().st_mtime)
            cfg.IO.CHECKPOINT = latest_ckpt.resolve().as_posix()

    assert Path(cfg.IO.ANNOTATIONS+'/train_coco.json').is_file, f"Cannot find {cfg.IO.ANNOTATIONS+'/train_coco.json'}, did you run preprocess step?"
    coco_train = COCO(cfg.IO.ANNOTATIONS+'/train_coco.json')
    if Path(cfg.IO.ANNOTATIONS+'/predict_coco.json').is_file():
        predict_coco = COCO(cfg.IO.ANNOTATIONS+'/predict_coco.json')
    else:
        predict_rasters = list(Path(cfg.IO.PREDICT_RASTER_DIR).glob('*.tiff')) + list(Path(cfg.IO.PREDICT_RASTER_DIR).glob('*.tif'))
        assert len(predict_rasters) > 0, f"{Fore.RED}Can not find any rasters under {cfg.IO.PREDICT_RASTER_DIR}"
        predict_coco_list = []
        for p in predict_rasters:
            print(f'{Fore.GREEN}Processing predict raster {p.stem}')
            predict_r = Tile(fpth=p, output_path = cfg.IO.PREDICT, buffer_size=50, tile_size=cfg.PREPROCESS.TRANSFORM.RANDOM_CROP_HEIGHT-100, tile_mode='pixel')
            predict_r.tile_image(mode=cfg.PREPROCESS.MODE)
            coco_path = predict_r.to_COCO(cfg.IO.TEMP + f'/{p.stem}_coco_predict.json')
            predict_coco_list.append(coco_path)
        merge_coco(predict_coco_list, cfg.IO.ANNOTATIONS+'/predict_coco.json')
        predict_coco = COCO(cfg.IO.ANNOTATIONS+'/predict_coco.json')
        
    if cfg.TRAIN.MODEL == 'maskrcnn_rgb':
        predict = Train(maskrcnn_rgb.MaskRCNN_RGB,
                      maskrcnn_rgb.LoadDataModule,
                      coco_train,
                      validate_coco = None,
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
    cfg.to_file(cfg_path)
        
def main():
    _default_cfg_path = Path(__file__).parent / 'utils/config.toml'
    parser = create_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.get_config:
        Path(args.get_config).mkdir(exist_ok=True, parents=True)
        shutil.copy(_default_cfg_path, args.get_config)
        print(f'{Fore.GREEN}Config copied under {Path(args.get_config).resolve()}')
        sys.exit(0)

    if args.example_data:
        download_example_data(args.example_data)
        sys.exit(0)
    
    import torch
    torch.set_float32_matmul_precision('high')
    
    args.func(args)
    
if __name__ == '__main__':
    main()
