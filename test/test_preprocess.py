
import os
import sys
from pathlib import Path
import rasterio as rio
import numpy as np
from rasterio.windows import Window, transform
from rasterio.transform import from_origin
sys.path.append((Path(__file__).parents[1] / 'src').as_posix())
from DLtreeseg.core.preprocess import Tile


fpath = Path('/home/vscode/remotehome/DL_drake/Drake/Ref/Drake20220928_MS.tif')
shp_path = Path('/home/vscode/remotehome/DL_drake/shp/shurbcrown_train.shp')
output_path = Path('/home/vscode/remotehome/DL_drake/output')

a = Tile(fpth=fpath, output_path=output_path, buffer_size=20, tile_size=100)
a.tile_image()
a.tile_shape(shp_path)
a.to_COCO('/home/vscode/remotehome/DL_drake/shurbcoco.json')
print()
