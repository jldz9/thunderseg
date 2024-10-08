import sys
sys.path.append('/home/jldz9/DL/DL_packages/detectree2')
sys.path.append('/home/jldz9/DL/DL_packages/DLtreeseg/src')
from pathlib import Path
from detectree2.preprocessing.tiling import load_class_mapping
from DLtreeseg.core import Tile, create_project_structure


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