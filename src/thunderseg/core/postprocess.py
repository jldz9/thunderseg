#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import geopandas as gpd
import pandas as pd
from colorama import Fore, init
init(autoreset=True)
from rasterio.transform import Affine
from rasterio.windows import transform as window_transform
from rasterio.features import shapes
from shapely.geometry import shape, box

from thunderseg.utils import float_to_binary_mask, windowdict_to_window
from pycocotools.coco import COCO
"""
Module include post-process functions for thunderseg, including image mosaic
"""
class Postprocess:
    """
    Handle georeference for predictions
    """
     
    def __init__(self, predict_coco, predict_result, output_path, predict_result_parser = None):
        """The predict_coco should be combined from export from merge_coco function under utils.tool"""  #TODO add more compatibility to regular COCO format in the furture
        if isinstance(predict_coco, COCO):
            self.coco = predict_coco
        else:
            self.coco = COCO(predict_coco)
        self.predict = [p for batch in predict_result for p in batch]
        self.output = output_path
        self.predict_parser = predict_result_parser

    def _parser_artichture_type(self):
        keys = {key for d in self.predict for key in d.keys()}
        if keys == {'masks', 'boxes', 'scores', 'labels'}:
            self.predict_parser = 'mask_rcnn'

    def mask_rcnn_postprocess(self):
        print(f'{Fore.GREEN} Working on post-processing for Mask-RCNN')
        for info in self.coco.dataset['info']:
            affine = Affine(*info['affine'])
            img_range = info['image_ids'].split('-')
            imgs = self.coco.loadImgs(range(int(img_range[0]),int(img_range[1])+1))
            polygon_gdfs_list = []
            bbox_gdfs_list = []
            for idx,img in enumerate(imgs):
                window = windowdict_to_window(img['window'])
                window_affine = window_transform(window, affine)
                predict = self.predict[idx]
                masks = [float_to_binary_mask(mask.to('cpu').numpy(), threshold=0.7) for mask in predict['masks']] #TODO threshold in config file
                if not masks:
                     continue
                labels = [int(label.to('cpu').numpy()) for label in predict['labels']]
                scores = [float(score.to('cpu').numpy()) for score in predict['scores']]
                shapes_generator = [shapes(binary_mask, transform=window_affine) for binary_mask in masks]
                polygons = []
                bboxes = []
                for i, shape_generator in enumerate(shapes_generator):
                    for geom, value in shape_generator:
                        if value == 1: 
                            polygon = shape(geom)
                            bounding_box = polygon.bounds
                            polygons.append({
                                "geometry": polygon,
                                "category": self.coco.loadCats(labels[i])[0]['name'],
                                'score': scores[i]
                            })
                            bboxes.append({
                                "bounding_box": box(*bounding_box),
                                "category": self.coco.loadCats(labels[i])[0]['name'],
                                'score': scores[i]
                            })
                polygon_gdfs_list.append(gpd.GeoDataFrame(polygons, geometry='geometry'))
                bbox_gdfs_list.append(gpd.GeoDataFrame(bboxes, geometry='bounding_box'))
            polygon_gdf = pd.concat(polygon_gdfs_list)
            polygon_gdf.set_crs(f'EPSG:{info["crs"]}', inplace=True)
            polygon_gdf.drop_duplicates(subset='geometry', inplace=True)
            polygon_gdf.to_file(f'{self.output}/{Path(info["file_name"]).stem + "_polygon.gpkg"}', layer='polygon', driver='GPKG') #TODO make export name in config
            bbox_gdf = pd.concat(bbox_gdfs_list)
            bbox_gdf.set_crs(f'EPSG:{info["crs"]}', inplace=True)
            bbox_gdf.drop_duplicates(subset='bounding_box', inplace=True) # TODO: drop duplicates won't work for slightly different box and polygon, find a way to figure this out
            bbox_gdf.to_file(f'{self.output}/{Path(info["file_name"]).stem + "_bounding_box.gpkg"}', layer='bounding_box', driver='GPKG')
            print(f'{Fore.GREEN} Prediction saved under {self.output}/{Path(info["file_name"]).stem + "_bounding_box.gpkg"}')
            print(f'{Fore.GREEN} Prediction saved under {self.output}/{Path(info["file_name"]).stem + "_polygon.gpkg"}')

    
                
