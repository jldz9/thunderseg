import sys
sys.path.append('/home/jldz9/DL/DL_packages/DLtreeseg/src/DLtreeseg')
from pathlib import Path
import rasterio as rio
import numpy as np
from rasterio.windows import Window, transform
from rasterio.transform import from_origin
import DLtreeseg.core.preprocess as preprocess

fpath = Path('/home/jldz9/DL/DL_drake/Drake/Ref/Drake20220928_MS.tif')
output_path = Path('/home/jldz9/DL/output')

a = preprocess.Preprocess(fpth=fpath, output_path=output_path)
a.save_tiles('/home/jldz9/DL/Drake20220928_MS.h5')