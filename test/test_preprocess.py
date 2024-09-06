from pathlib import Path
import rasterio as rio
import numpy as np
from rasterio.windows import Window, transform
from rasterio.transform import from_origin
from DLtreeseg.core.preprocess import Tile

fpath = Path('/home/jldz9/DL/DL_drake/Drake/Ref/Drake20220928_MS.tif')
output_path = Path('/home/jldz9/DL/output')

a = Tile(fpth=fpath, output_path=output_path, buffer_size=20, tile_size=100)
print(())