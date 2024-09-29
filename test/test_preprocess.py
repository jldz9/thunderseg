
import os
os.environ['PROJ_LIB'] = '/home/jldz9/miniconda3/envs/DL/share/proj'
import sys
sys.path.append('/home/jldz9/DL/DL_packages/DLtreeseg/src')
from pathlib import Path
import rasterio as rio
import numpy as np
from rasterio.windows import Window, transform
from rasterio.transform import from_origin
from DLtreeseg.core.preprocess import Tile

fpath = Path('/home/jldz9/DL/DL_drake/Drake/Ref/Drake20220928_MS.tif')
shp_path = Path('/home/jldz9/DL/DL_drake/shp/shurbcrown.shp')
output_path = Path('/home/jldz9/DL/output')

a = Tile(fpth=fpath, output_path=output_path, buffer_size=20, tile_size=100)
a.tile_image()
a.tile_shape(shp_path)
a.parse_COCO('/home/jldz9/DL/output/shurbcoco.json', 'shurbcoco')
print()
