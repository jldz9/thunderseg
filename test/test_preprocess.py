
import os
import sys
from pathlib import Path
import rasterio as rio
import numpy as np
from rasterio.windows import Window, transform
from rasterio.transform import from_origin
sys.path.append((Path(__file__).parents[1] / 'src').as_posix())
from thunderseg.core.preprocess import Preprocess
from thunderseg.utils import merge_coco
from pycocotools.coco import COCO

fpath = Path('/home/jldz9/DL/DL_drake/traintaster.tif')
shp_path = Path('/home/jldz9/DL/DL_drake/shp/shp_20220928.shp')
output_path = Path('/home/jldz9/DL/DL_drake')

a = Preprocess(fpth=fpath, output_path=output_path, buffer_size=0, tile_size=100)
a.tile_shape(shp_path)
a.to_COCO('/home/vscode/remotehome/DL_drake/demo.json')
print()


#coco1 = '/home/vscode/remotehome/DL_drake/shurbcoco.json'
#coco2 = '/home/vscode/remotehome/DL_drake/Drake20220928_MS_coco.json'
#coco3 = '/home/vscode/remotehome/DL_drake/Drake20220928_MS_row5742_col5742_coco.json'

#merge_coco([coco1, coco2],'/home/vscode/remotehome/DL_drake/combined.json')