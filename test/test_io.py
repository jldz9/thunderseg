import sys
sys.path.append('/home/jldz9/DL/DL_packages/detectree2')
sys.path.append('/home/jldz9/DL/DL_packages/detectree2/detectree2')
sys.path.append('/home/jldz9/DL/DL_packages/thunderseg/src')
from pathlib import Path
import glob
import rasterio as rio
import geopandas as gpd 
"""
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
"""
shp_path = Path('/home/jldz9/DL/DL_drake/shp/shurbcrown.shp')
shp = gpd.read_file(shp_path)
print()
