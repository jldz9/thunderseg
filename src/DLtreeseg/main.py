import argparse
import sys
import shutil
from pathlib import Path

from colorama import Fore, init
init(autoreset=True)
import geopandas as gpd
import pandas as pd
from DLtreeseg.core import Tile, create_project_structure, read_config, write_config
from DLtreeseg.utils import merge_coco
from DLtreeseg._version import __version__


def create_parser():
    synopsis = 'This is a python interface for DLtreeseg program'
    name = 'DLtreeseg'
    parser = argparse.ArgumentParser(name, description=synopsis, add_help=False)
    parser.add_argument("-v", "--version", action='version', version=f'DLtreeseg {__version__}')
    parser.add_argument('-c', '--config', metavar='PATH', help='Path to config for automate process')
    parser.add_argument('--init', metavar='PATH', help='Initiliza the working dir')
    subparser = parser.add_subparsers(dest='step')
    preprocess = subparser.add_parser('preprocess', help='Create file structure under given path')
    preprocess.add_argument('-c', '--config',metavar='PATH', help='Path to config')
    train = subparser.add_parser('train', help='Preprocess raster into tiles and register dataset')
    train.add_argument('-c', '--config', metavar='PATH', help='Path to config')
    return parser, preprocess, train

def preprocess_step(cfg):
    ori_train_rasters = list(Path(cfg['io']['train_raster_dir']).glob('*.tiff')) + list(Path(cfg['io']['train_raster_dir']).glob('*.tif'))
    assert len(ori_train_rasters) > 0, f"Not able to find any rasters under {cfg['io']['train_shp_dir']}"
    ori_train_shps = list(Path(cfg['io']['train_shp_dir']).glob('*.shp')) #TODO might need to support more formats
    assert len(ori_train_shps) > 0, f"Not able to find any shapefiles under {cfg['io']['train_shp_dir']}"
    shps = [gpd.read_file(shp) for shp in ori_train_shps]
    merged_shp = pd.concat(shps, ignore_index=True)
    merged_shp.drop_duplicates(inplace=True)
    merged_shp.to_file(cfg['io']['train_shp']+'/train_shp.shp')
    coco_list = []
    for r in ori_train_rasters:
        print(f'{Fore.GREEN}Processing {r.stem}')
        a = Tile(fpth=r, output_path=cfg['io']['train'], buffer_size=cfg['preprocess']['buffer_size'], tile_size=cfg['preprocess']['tile_size'])
        a.tile_image(mode=cfg['preprocess']['mode'], shp_path=cfg['io']['train_shp']+'/train_shp.shp')
        coco_path = a.to_COCO(cfg['io']['annotations'] + f'/{r.stem}_coco.json', **cfg['preprocess']['COCO_info'])
        coco_list.append(coco_path)
    merge_coco(coco_list, cfg['io']['annotations']+'/merged_coco.json')
   
def main(iargs=None):
    _default_cfg_path = Path(__file__).parent / 'utils/config.toml'
    _default_cfg = read_config(_default_cfg_path)
    parser, preprocess_parser, train_parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if len(iargs) <1:
        parser.print_help()
        sys.exit(1)
    if inps.init:
        shutil.copy(_default_cfg_path, inps.init)
        print(f'{Fore.GREEN} Config copied under {Path(inps.init).resolve()}')
    if inps.step == 'preprocess':   
        if inps.config is None:
            preprocess_parser.print_help()
            sys.exit(1)
        cfg = read_config(inps.config)
        directories = create_project_structure(cfg['io']['workdir'])
        cfg['io'].update(directories)
        write_config(cfg, inps.config)
        preprocess_step(cfg)
    if inps.step == 'train':
        train_cfg = []
if __name__ == '__main__':
    main(sys.argv[1:])