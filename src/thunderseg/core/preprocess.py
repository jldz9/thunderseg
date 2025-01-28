#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image preprocess module for thunderseg, include image IO, tilling
"""
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.io import DatasetReader, MemoryFile
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely import box

from thunderseg.utils import to_pixelcoord, COCO_parser, window_to_dict, get_mean_std, assert_json_serializable

class Tile: 
    """Preprocess module for image IO, tilling.
    
    Use to tile input raster into certain give size tiles, add buffer around tiles to reduce edge effect. 
    """
    def __init__(self,
                 fpth: str,
                 output_path: str = '.',
                 tile_size : int = 236,
                 buffer_size : int = 10,
                 tile_mode = 'meter',
                 ):
        """ Initializes parameters
        Args:
            fpth: Path to single raster file.
            output_path: Output file path, default is current directory.
            tile_size: Tile size, all tiles will be cut into squares. Unit: meter
            buffer_size: Buffer size around tiles. Unit:meter
            tile_mode: 'meter' or 'pixel' #TODO add more unit support in the furture
            debug: Switch to turn on debug
        """
        self.fpth = Path(fpth).absolute()
        self._output_path = Path(output_path).absolute()
        self._output_path.mkdir(exist_ok=True)
        self._tile_size = tile_size
        if not isinstance(self._tile_size, int):
            raise TypeError(f"Tile_size tuple should be Int, not {type(self.tile_width)}")
        self._buffer_size = buffer_size
        if not isinstance(self._buffer_size, int) :
            raise TypeError(f"Buffer size should be Int, not {type(self._buffer_size_pixel_x)}")
        self._images = {'id':[],
                        "file_name":[],
                        "width":[],
                        "height": [],
                        "date_captured":[]
        }
        self._annotations = {'id':[],
                             'image_id':[],
                             'category_id':[],
                             'bbox':[],
                             'area':[],
                             'iscrowd':[],
                             'segmentation':[]} 
        self.tile_mode = tile_mode

    def _get_window(self):
        """Make rasterio windows for raster tiles, pad original dataset to makesure all tiles size looks the same.
        Tiles will overlap on right and bottom buffer.         
        """
        oridataset = rio.open(self.fpth)
        profile = oridataset.profile.copy()
        y = profile['height']
        x = profile['width']
        transform = oridataset.profile['transform']
        # Convert meter tile size to pixel tile size
        if self.tile_mode == 'meter':
            tile_size_pixel_x = int(np.ceil(self._tile_size / oridataset.res[0]))
            tile_size_pixel_y = int(np.ceil(self._tile_size / oridataset.res[1]))
            buffer_size_pixel_x = int(np.ceil(self._buffer_size / oridataset.res[0]))
            buffer_size_pixel_y = int(np.ceil(self._buffer_size / oridataset.res[1]))
        elif self.tile_mode == 'pixel':
            tile_size_pixel_x = self._tile_size
            tile_size_pixel_y = self._tile_size
            buffer_size_pixel_x = self._buffer_size
            buffer_size_pixel_y = self._buffer_size
        # Calculate number of tiles along height and width with buffer.
        n_tiles_x = int(np.ceil((x + buffer_size_pixel_x) / (buffer_size_pixel_x + tile_size_pixel_x)))
        n_tiles_y = int(np.ceil((y + buffer_size_pixel_y) / (buffer_size_pixel_y + tile_size_pixel_y)))
        
        # Add buffer to original raster to make sure every tiles has same size.
        data = oridataset.read()
        data = np.where(data<0, 0, data)
        pad = ((0,0),
                (buffer_size_pixel_y, n_tiles_y * (tile_size_pixel_y + buffer_size_pixel_y) - y),
                (buffer_size_pixel_x, n_tiles_x * (tile_size_pixel_x + buffer_size_pixel_x) - x)
                )
        padded_data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
        self._profile = profile.update({
            'height': padded_data.shape[1],
            'width': padded_data.shape[2],
            'transform': Affine(transform[0],transform[1], transform[2]- buffer_size_pixel_x*transform[0],
                                transform[3],transform[4], transform[5]- buffer_size_pixel_x*transform[4])
        })

        # Make meshgrid to create 2d (x,y) index of all tiles
        tile_index_x, tile_index_y = np.meshgrid(np.arange(n_tiles_x), np.arange(n_tiles_y))
        flat_tile_x = tile_index_x.flatten()
        flat_tile_y = tile_index_y.flatten()

        # Make windows for all tiles.
        self._windows = [
            Window(
            max(((start_x * (tile_size_pixel_x + (2 * buffer_size_pixel_x)) - start_x * buffer_size_pixel_x), 0)),
            max(((start_y * (tile_size_pixel_y + (2 * buffer_size_pixel_y)) - start_y * buffer_size_pixel_x), 0)),
            tile_size_pixel_x + 2 * buffer_size_pixel_x,
            tile_size_pixel_y + 2 * buffer_size_pixel_y,
            ) 
            for start_x, start_y in zip(flat_tile_x, flat_tile_y)
            ]
        oridataset.close()
        memfile = MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(padded_data)
        self._dataset = memfile.open()

    def resample(self, new_resolution: float):
        """Resample original raster to certain resolution (meter).
        Args: 
            new_resolution: the resolution of new raster
        """
        print(f'Resampling raster to {new_resolution} m')
        with rio.open(self.fpth) as ori_dataset:
            ori_dataset = rio.open(self.fpth)
            profile = ori_dataset.profile.copy()
            old_transform = profile['transform']
            new_width= int(np.round((ori_dataset.bounds.right - ori_dataset.bounds.left) / new_resolution))
            new_height = int(np.round((ori_dataset.bounds.top - ori_dataset.bounds.bottom) / new_resolution))
            profile.update({
                'height': new_height,
                'width': new_width,
                'transform': Affine(new_resolution, old_transform.b, old_transform.c, 
                                    old_transform.d, -new_resolution, old_transform.f)

            })
            data = ori_dataset.read(
                out_shape = (ori_dataset.count, new_height, new_width),
                resampling = Resampling.gauss
            )
        memfile = MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(data)
        self._dataset = memfile.open()

    def tile_image(self, mode='BGR', shp_path: str = None):
        """Cut input image into square tiles with buffer and preserver geoinformation for each tile."""         
        if mode == 'BGR':
            band = 3
        elif mode == 'MS':
            band = self._dataset.count
        self._get_window()
        tiles_list = []
        self._profiles = []
        num_tiles = len(self._windows)
        for idx, window in enumerate(self._windows):
            tile_profile = self._dataset.profile
            tile_data = self._dataset.read(window=window)
            tile_profile.update({
            'transform': self._dataset.window_transform(window),
            'height': window.height,
            'width': window.width,
            'count': band
        })
            tiles_list.append(tile_data)
            self._profiles.append(tile_profile)
            sys.stdout.write(f'\rWorking on: {idx+1}/{num_tiles} image tile')
            sys.stdout.flush()
            self._images['id'].append(idx+1)
            filename = f'{self._output_path}/{self.fpth.stem}_row{window.row_off}_col{window.col_off}.tif'
            self._images['file_name'].append(filename)
            self._images['width'].append(window.width)
            self._images['height'].append(window.height)
            self._images['date_captured'].append(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            self.to_file(filename, tile_data, tile_profile, mode=mode)
            self.to_png(tile_data, Path(filename).with_suffix('.png').as_posix())
        print()
        self._stack_tiles = np.stack(tiles_list, axis=0)
        self._stack_tiles = self._stack_tiles[:,0:band, :,:] # Make sure if use BGR mode will export only frist 3 bands
        mean, std = get_mean_std(self._stack_tiles)
        self._report = {'file_name': self.fpth.as_posix(),
                        'output_path': self._output_path.as_posix(),
                        'tile_size': self._tile_size,
                        'buffer_size': self._buffer_size,
                        'tile_numbers': num_tiles,
                        'original_size': str(rio.open(self.fpth).shape),
                        'buffed_size': str(self._dataset.shape),
                        'crs': str(self._dataset.crs.to_epsg()),
                        'band': band,
                        'affine': (self._dataset.transform.a, 
                                self._dataset.transform.b,
                                self._dataset.transform.c,
                                self._dataset.transform.d,
                                self._dataset.transform.e,
                                self._dataset.transform.f),
                        'driver': self._dataset.profile['driver'],
                        'dtype': self._dataset.profile['dtype'],
                        'nodata':self._dataset.profile['nodata'],
                        'pixel_mean' : [float(i) for i in mean],
                        'pixel_std' : [float(i) for i in std],
                        'mode': mode
                        }
        if shp_path is not None and Path(shp_path).is_file():
            self.tile_shape(shp_path)
            self._no_shp = False
        else: 
            self._no_shp = True
   
    def tile_shape(self, shp_path: str):
        """Use raster window defined by _get_window to tile input shapefiles.
        Args:
            shp_path: Path to single shapefile
        """
        if not hasattr(self, "_windows"):
            self._get_window()
        self._shp_path = Path(shp_path)
        self._shpdataset = gpd.read_file(self._shp_path)
        self._shpdataset = self._shpdataset.fillna(0).astype({'category':int, 'iscrowd':int})
        if self._shpdataset.crs.to_epsg() != self._dataset.crs.to_epsg():
            self._shpdataset = self._shpdataset.to_crs(epsg=self._dataset.crs.to_epsg())
        annotation_id = 1
        for idx, window in enumerate(self._windows):
            geobounds = self._dataset.window_bounds(window)
            bbox = box(*geobounds)
            window_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=self._dataset.crs.to_epsg())
            intersection = gpd.overlay(self._shpdataset, window_gdf, how='intersection')
            if len(intersection) > 0:
                _num_of_intersect = len(intersection)
                print(f'found {_num_of_intersect} polygons in tile {idx}')
                for _, row in intersection.iterrows():
                    pixel_coord = to_pixelcoord(self._dataset.transform, window, row.geometry)
                    area = row.geometry.area/(self._dataset.res[0]*self._dataset.res[1])
                    bbox = [min(pixel_coord[0::2]), 
                            min(pixel_coord[1::2]), 
                            max(pixel_coord[0::2]) - min(pixel_coord[0::2]),
                            max(pixel_coord[1::2]) - min(pixel_coord[1::2])]
                    self._annotations['id'].append(annotation_id)
                    annotation_id += 1
                    self._annotations['image_id'].append(idx+1)
                    self._annotations['category_id'].append(row.category)
                    self._annotations['bbox'].append(bbox)
                    self._annotations['area'].append(area)
                    self._annotations['iscrowd'].append(row.iscrowd)
                    self._annotations['segmentation'].append(pixel_coord)

    def to_COCO(self, output_path: str = None, **kwargs) -> str:
        """Convert input images and annotations to COCO format.
        Args:
            kwargs: Meta data provided to this method that store in "Info" section. Needs to be json serializable. 
        Return: 
            self._coco_path: The path to COCO json file
        
        """
        kwargs.update(self._report)
        kwargs["date_created"] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        assert_json_serializable(**kwargs)
        self._coco = COCO_parser(kwargs)
        self._coco.add_categories(id = [1], name=['shurb'], supercategory=['plant'])
        self._coco.add_licenses(license_id=[1],license_url=[''],license_name=[''])
        self._coco.add_images(id = self._images['id'], 
                        file_name = self._images['file_name'],
                        width = self._images['width'],
                        height = self._images['height'],
                        date_captured = self._images['date_captured'],
                        window = [window_to_dict(w) for w in self._windows]
                        )
        if self._no_shp:
            self._coco.add_annotations()
        else:
            self._coco.add_annotations(id = self._annotations['id'],
                                image_id = self._annotations['image_id'],
                                category_id = self._annotations['category_id'],
                                bbox = self._annotations['bbox'],
                                area = self._annotations['area'],
                                iscrowd = self._annotations['iscrowd'],
                                segmentation = self._annotations['segmentation'],
                                bbox_mode='xywh'
                                )
        if output_path is not None:
            self._coco_path = Path(output_path)
        else:
            self._coco_path = f'{self._output_path}/{self.fpth.stem}_coco.json'
        self._coco.save_coco(self._coco_path)
        print(f'COCO saved at {self._coco_path}')
        return self._coco_path
    
    @property
    def data(self):
        return self._stack_tiles
    
    @property
    def window(self):
        return self._windows
    
    @property
    def profile(self):
        return self._profiles
    
    @property
    def summary(self):
        return self._report
    
    @property
    def ori_data(self) -> DatasetReader:
        """
        This returns the rasterio dataset for the padded original raster
        """
        return self._dataset
    
    def clear(self):
        for attr in vars(self): 
            setattr(self, attr, None)
    
    def to_file(self, path_to_file:str, data:np.ndarray, profile=None, mode:str='BGR'):
        path_to_file = Path(path_to_file)
        if mode.upper() == 'BGR':
            with rio.open(path_to_file,'w', **profile) as dst:
                for i in range(0,3):
                    dst.write(data[i], i+1)
        if mode.upper() == 'MS':
            with rio.open(path_to_file, 'w', **profile) as dst:
                dst.write(data)

    def to_png(self, data: np.ndarray, path_to_file:str):
        band1 = data[0] # B
        band2 = data[1] # G
        band3 = data[2] # R
        stack = np.stack([band1, band2, band3], axis=0)
        stack = np.transpose(stack, (1,2,0))
        min_val = np.min(stack)  # Minimum value in the array
        max_val = np.max(stack)
        if max_val == min_val:
            normalized_array = np.zeros_like(stack, dtype=np.float32)  # or np.ones_like(array, dtype=np.float32)
        else:
            normalized_array = (stack - min_val) / (max_val - min_val) * 255
        
        array_rgb = cv2.cvtColor(normalized_array, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_to_file, array_rgb)

