import sys
import time
sys.path.append('/home/jldz9/DL/DL_packages/thunderseg/src')

#from thunderseg.core.preprocess import Preprocess
#from pycocotools.coco import COCO
'''
raster = '/home/jldz9/thunderseg_test/train/Drake20220928_MS.tif'
shp = '/home/jldz9/thunderseg_test/shp/train_shp/shp_20220928.shp'
output = '/home/jldz9/thunderseg_test/train/'
coco = '/home/jldz9/thunderseg_test/shp/train_shp/instances_default.json'
a = Preprocess(fpth=raster, output_path=output, buffer_size=40, tile_size=304, tile_mode='pixel', band_mode='BGR')
'''
from pathlib import Path
#from pycocotools import coco
#from thunderseg.utils.tool import coco_to_binary_mask, shp_to_binary_mask

#coco = COCO(r'/home/jldz9/thunderseg_test/annotation/after/maskrcnn after/annotations/instances_Validation.json')
#img_ids = [1, 2, 3]
#binary_13 = coco_to_binary_mask(coco, image_dir='/home/jldz9/thunderseg_test/annotation/images/Train',category_name='shrub', img_ids=img_ids[0], debug='/home/jldz9/thunderseg_test/annotation/after/')
#binary_21 = coco_to_binary_mask(coco, image_dir='/home/jldz9/thunderseg_test/annotation/images/Train',category_name='shrub', img_ids=img_ids[1], debug='/home/jldz9/thunderseg_test/annotation/after/')
#binary_25 = coco_to_binary_mask(coco, image_dir='/home/jldz9/thunderseg_test/annotation/images/Train',category_name='shrub', img_ids=img_ids[2], debug='/home/jldz9/thunderseg_test/annotation/after/')
#print(coco)

shp = r'/home/jldz9/thunderseg_test/annotation/shp/Drake20220928_MS_row6880_col9632.shp'
output = r'/home/jldz9/thunderseg_test/annotation/shp/'
raster = r'/home/jldz9/thunderseg_test/train/Drake20220928_MS/rasters/Drake20220928_MS_row6880_col9632.tif'
import rasterio as rio
import numpy as np
clip = Path('/home/jldz9/thunderseg_test/clip/')
files = list(clip.glob('*.tif'))
from PIL import Image
for file in files:
    print(file)
    with rio.open(file) as src:
        red = src.read(3)
        green = src.read(2)
        blue = src.read(1)
    img = np.stack((red, green, blue), axis=-1)
    bgr_image = (255 * (img / img.max())).astype(np.uint8)
    img = Image.fromarray(bgr_image, mode ='RGB')
    img.save(file.with_suffix('.png'))


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
from scipy.stats import skewnorm
a = -5
plt.boxplot(data, vert=True, patch_artist=True,
            boxprops=dict(facecolor="lightcoral", color="darkred"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="darkred"),
            capprops=dict(color="darkred"),
            flierprops=dict(marker='o', color='black', markersize=2))