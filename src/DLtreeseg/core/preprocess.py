#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image preprocess module for DLtreeseg, include image IO, tilling
"""
import io
import json
import sys
from pathlib import Path
from datetime import datetime, timezone


import geopandas as gpd
import numpy as np
import rasterio as rio
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from rasterio.io import DatasetReader, MemoryFile
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely import box

from DLtreeseg.utils import pack_h5_list, to_pixelcoord, COCO_parser, window_to_dict, get_mean_std
from DLtreeseg.core import save_h5, to_file


class Tile: 
    """Preprocess module for image IO, tilling.
    
    Use to tile input raster into certain give size tiles, add buffer around tiles to reduce edge effect. 
    """
    def __init__(self,
                 fpth: str,
                 output_path: str = '.',
                 tile_size : int = 256,
                 buffer_size : int = 10,
                 debug = False
                 ):
        """ Initializes parameters
        Args:
            fpth: Path to single raster file.
            output_path: Output file path, default is current directory.
            tile_size: Tile size, all tiles will be cut into squares. Unit: meter
            buffer_size: Buffer size around tiles. Unit:meter
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
        
        if debug is True:
            print('debug')

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
        tile_size_pixel_x = int(np.ceil(self._tile_size / oridataset.res[0]))
        tile_size_pixel_y = int(np.ceil(self._tile_size / oridataset.res[1]))
        buffer_size_pixel_x = int(np.ceil(self._buffer_size / oridataset.res[0]))
        buffer_size_pixel_y = int(np.ceil(self._buffer_size / oridataset.res[1]))
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

    def tile_image(self, mode='rgb'):
        """Cut input image into square tiles with buffer and preserver geoinformation for each tile."""         
        if mode == 'rgb':
            suffix = 'png'
        elif mode == 'tig':
            suffix = 'tif'
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
        })
            tiles_list.append(tile_data)
            self._profiles.append(tile_profile)
            sys.stdout.write(f'\rWorking on: {idx+1}/{num_tiles} image tile')
            sys.stdout.flush()
            self._images['id'].append(idx+1)
            filename = f'{self._output_path}/{self.fpth.stem}_row{window.row_off}_col{window.col_off}.{suffix}'
            self._images['file_name'].append(filename)
            self._images['width'].append(window.width)
            self._images['height'].append(window.height)
            self._images['date_captured'].append(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            #to_file(filename, tile_data, tile_profile, suffix=suffix)
        print()
        self._stack_tiles = np.stack(tiles_list, axis=0)
        mean, std = get_mean_std(self._stack_tiles)
        self._report = {'file_name': self.fpth.as_posix(),
                        'output_path': self._output_path.as_posix(),
                        'tile_size': self._tile_size,
                        'buffer_size': self._buffer_size,
                        'tile_numbers': num_tiles,
                        'original_size': str(rio.open(self.fpth).shape),
                        'buffed_size': str(self._dataset.shape),
                        'crs': str(self._dataset.crs.to_epsg()),
                        'band': self._dataset.count,
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
                        'band_order': 'BGREN'
                        }
        print()
        
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
            sys.stdout.write(f'\rClipping shapfile for: {idx+1}/{len(self._windows)} tile')
            sys.stdout.flush()
            geobounds = self._dataset.window_bounds(window)
            bbox = box(*geobounds)
            window_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=self._dataset.crs.to_epsg())
            intersection = gpd.overlay(self._shpdataset, window_gdf, how='intersection')
            print(f'{idx}:{len(intersection)}')
            if len(intersection) > 0:
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
        print()

    def to_COCO(self, cocopath:str, name:str, **kwargs):
        """Convert input images and annotations to COCO format.
        Args:
            info: Information of dataset for COCO json file, default is an empty dict
            output_path: Output path to save COCO json file, default is annotation folder under current directory
            kwargs: Other meta data information for COCO json file "Info" section, could include "description", "url", "version", "year", "contributor"
        """
        kwargs.update(self._report)
        kwargs["date_created"] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
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
        self._coco.add_annotations(id = self._annotations['id'],
                             image_id = self._annotations['image_id'],
                             category_id = self._annotations['category_id'],
                             bbox = self._annotations['bbox'],
                             area = self._annotations['area'],
                             iscrowd = self._annotations['iscrowd'],
                             segmentation = self._annotations['segmentation']
                             )
        self._coco.save_json(cocopath)
        register_coco_instances(name,{}, cocopath, self._output_path)
        return self._coco.data
    
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

    def to_h5(self, save_path:str):
        """
        Save tiles into HDF5 dataset
        """
        save_h5(save_path=save_path,
                data = self._stack_tiles,
                attrs= self._report,
                windows = pack_h5_list(self._windows),
                profiles = pack_h5_list(self._profiles)
                )
        
    

    