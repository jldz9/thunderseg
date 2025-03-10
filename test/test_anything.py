import sys
import time
sys.path.append('/home/jldz9/DL/DL_packages/thunderseg/src')

from thunderseg.core.preprocess import Preprocess
from pycocotools.coco import COCO
'''
raster = '/home/jldz9/thunderseg_test/train/Drake20220928_MS.tif'
shp = '/home/jldz9/thunderseg_test/shp/train_shp/shp_20220928.shp'
output = '/home/jldz9/thunderseg_test/train/'
coco = '/home/jldz9/thunderseg_test/shp/train_shp/instances_default.json'
a = Preprocess(fpth=raster, output_path=output, buffer_size=40, tile_size=304, tile_mode='pixel', band_mode='BGR')
'''

from pycocotools import coco
from thunderseg.utils.tool import coco_to_binary_mask, shp_to_binary_mask

#coco = COCO(r'/home/jldz9/thunderseg_test/annotation/after/maskrcnn after/annotations/instances_Validation.json')
#img_ids = [1, 2, 3]
#binary_13 = coco_to_binary_mask(coco, image_dir='/home/jldz9/thunderseg_test/annotation/images/Train',category_name='shrub', img_ids=img_ids[0], debug='/home/jldz9/thunderseg_test/annotation/after/')
#binary_21 = coco_to_binary_mask(coco, image_dir='/home/jldz9/thunderseg_test/annotation/images/Train',category_name='shrub', img_ids=img_ids[1], debug='/home/jldz9/thunderseg_test/annotation/after/')
#binary_25 = coco_to_binary_mask(coco, image_dir='/home/jldz9/thunderseg_test/annotation/images/Train',category_name='shrub', img_ids=img_ids[2], debug='/home/jldz9/thunderseg_test/annotation/after/')
#print(coco)

shp = r'/home/jldz9/thunderseg_test/annotation/shp/Drake20220928_MS_row6880_col9632.shp'
output = r'/home/jldz9/thunderseg_test/annotation/shp/'
raster = r'/home/jldz9/thunderseg_test/train/Drake20220928_MS/rasters/Drake20220928_MS_row6880_col9632.tif'

shp_to_binary_mask(shp, raster, output)