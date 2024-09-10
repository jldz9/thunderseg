import sys
sys.path.append('/home/jldz9/DL/DL_packages/detectree2')
sys.path.append('/home/jldz9/DL/DL_packages/detectree2/detectree2')
sys.path.append('/home/jldz9/DL/DL_packages/DLtreeseg/src')
from pathlib import Path
import glob
from DLtreeseg.core.io import setup_cfg
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectree2.models.predict import predict_on_data
from detectree2.preprocessing.tiling import tile_data
from detectree2.models.train import get_tree_dicts, get_filenames
from detectree2.models.outputs import project_to_geojson, polygon_from_mask, stitch_crowns, clean_crowns
import rasterio as rio
fpath = Path('/home/jldz9/DL/DL_drake/Drake/Ref/Drake20220928_MS.tif')
data = rio.open(fpath)
model = Path('/home/jldz9/DL/models/230729_05dates.pth')
tiles_path = '/home/jldz9/DL/DL_drake/tiles/'


buffer = 20
tile_width = 100
tile_height = 100
tile_data(data, tiles_path, buffer, tile_width, tile_height)

cfg = setup_cfg(update_model = str(model), out_dir='/home/jldz9/DL/result')
predict_on_data(tiles_path+"/", predictor=DefaultPredictor(cfg))
project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")
crowns = stitch_crowns(tiles_path + "predictions_geo/", 1)
print()

