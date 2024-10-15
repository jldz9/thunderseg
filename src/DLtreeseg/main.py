import argparse
import sys
import shutil
from pathlib import Path
from types import SimpleNamespace

from DLtreeseg.core import Tile, create_project_structure
from DLtreeseg.utils import  read_toml
from DLtreeseg._version import __version__


def create_parser():
    synopsis = 'This is a python interface for DLtreeseg program'
    name = 'DLtreeseg'
    parser = argparse.ArgumentParser(name, description=synopsis)
    parser.add_argument("-v", "--version", action='version', version=f'DLtreeseg {__version__}')
    parser.add_argument('-c', '--config', metavar='PATH', help='Path to config for automate process')
    parser.add_argument('--init', metavar='PATH', help='Initiliza the working dir')
    subparser = parser.add_subparsers(dest='step')
    preprocess = subparser.add_parser('preprocess', help='Create file structure under given path')
    preprocess.add_argument('-c', '--config',metavar='PATH', help='Path to config')
    train = subparser.add_parser('train', help='Preprocess raster into tiles and register dataset')
    train.add_argument('-c', '--config', metavar='PATH', help='Path to config')
    return parser

def preprocess(cfg):
    ori_train_rasters = list(Path(cfg.io.train_rastrer_dir).glob('*.tiff')) + list(Path(cfg.io.train_rastrer_dir).glob('*.tif'))
    ori_train_shps = list(Path(cfg.io.shape_file_dir).glob('*.shp'))
    for r in ori_train_rasters:
        a = Tile(fpth=r, output_path=cfg.train, buffer_size=cfg.preprocess.buffer_size, tile_size=cfg.preprocess.tile_size)
        a.tile_image()
        for s in ori_train_shps:
            a.tile_shape(s)
        a.to_COCO(cfg.annotations / 'train_coco.json', 'train_coco', **cfg.preprocess.COCO_info.__dict__)
    return
def main(iargs=None):
    _default_cfg_path = Path(__file__).parent / 'utils/config.toml'
    _default_cfg = read_toml(_default_cfg_path)
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if inps.init:
        ns = create_project_structure(inps.init)
        shutil.copy(_default_cfg_path, inps.init)
        _default_cfg.append(ns)
    if inps.step == 'preprocess':   
        cfg = _default_cfg.append(read_toml(inps.config))
        preprocess(cfg)
    if inps.step == 'train':
        cfg = _default_cfg.append(read_toml(inps.config))
        train_cfg = () 
if __name__ == '__main__':
    main(sys.argv[1:])