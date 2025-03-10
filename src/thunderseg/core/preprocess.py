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
from colorama import Fore, init
import fiftyone as fo
import geopandas as gpd
import numpy as np
from PIL import Image
import rasterio as rio
from rasterio.io import DatasetReader, MemoryFile
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely import box

from thunderseg.utils import to_pixelcoord, COCO_parser, window_to_dict, get_mean_std, assert_json_serializable

class Preprocess_v2:
    """Preprocess module for image IO, tilling.
    
    Use to tile input raster into certain give size tiles, add buffer around tiles to reduce edge effect. 

    version 2:
    - Use fiftyone for image and annotation management
    """
    def __init__(self,
                 fpth: str,
                 output_path: str = '.',
                 tile_size : int = 236,
                 buffer_size : int = 10,
                 tile_mode = 'meter',
                 band_mode = 'BGR', 
                 ):
        """ Initializes parameters"""
        self.fpth = Path(fpth).absolute()
        self.fodataset = fo.Dataset(name=self.fpth.stem)
        self.output_path = Path(output_path).absolute() / self.fpth.stem
        self.tile_size = tile_size
        self.buffer_size = buffer_size
        self.tile_mode = tile_mode
        self.band_mode = band_mode

    def _get_window(self):
        """Make rasterio windows for raster tiles, pad original dataset to makesure all tiles size looks the same.
        Tiles will overlap on right and bottom buffer.         
        """
        with rio.open(self.fpth) as oridataset: 
            profile = oridataset.profile.copy()
            y = profile['height']
            x = profile['width']
            transform = oridataset.profile['transform']
            # Convert meter tile size to pixel tile size
            if self.tile_mode == 'meter':
                tile_size_pixel_x = int(np.ceil(self.tile_size / oridataset.res[0]))
                tile_size_pixel_y = int(np.ceil(self.tile_size / oridataset.res[1]))
                buffer_size_pixel_x = int(np.ceil(self.buffer_size / oridataset.res[0]))
                buffer_size_pixel_y = int(np.ceil(self.buffer_size / oridataset.res[1]))
            elif self.tile_mode == 'pixel':
                tile_size_pixel_x = self.tile_size
                tile_size_pixel_y = self.tile_size
                buffer_size_pixel_x = self.buffer_size
                buffer_size_pixel_y = self.buffer_size

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
            self.padded_data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
            self.profile = profile.update({
                'height': self.padded_data.shape[1],
                'width': self.padded_data.shape[2],
                'transform': Affine(transform[0],transform[1], transform[2]- buffer_size_pixel_x*transform[0],
                                    transform[3],transform[4], transform[5]- buffer_size_pixel_x*transform[4])
            })

            # Make meshgrid to create 2d (x,y) index of all tiles
            tile_index_x, tile_index_y = np.meshgrid(np.arange(n_tiles_x), np.arange(n_tiles_y))
            flat_tile_x = tile_index_x.flatten()
            flat_tile_y = tile_index_y.flatten()

            # Make windows for all tiles.
            self.windows = [
                Window(
                max(((start_x * (tile_size_pixel_x + (2 * buffer_size_pixel_x)) - start_x * buffer_size_pixel_x), 0)),
                max(((start_y * (tile_size_pixel_y + (2 * buffer_size_pixel_y)) - start_y * buffer_size_pixel_x), 0)),
                tile_size_pixel_x + 2 * buffer_size_pixel_x,
                tile_size_pixel_y + 2 * buffer_size_pixel_y,
                ) 
                for start_x, start_y in zip(flat_tile_x, flat_tile_y)
                ]
            
    def _raster_gt_tile_size(self):
        """Check if raster size is larger than tile size + 2*buffer_size on both height and width, if not, return False to prevent tilling."""
        with rio.open(self.fpth) as dataset:
            x = dataset.width * dataset.res[0]
            y = dataset.height * dataset.res[1]
            if x < self._tile_size + 2 * self._buffer_size or y < 2 * self._buffer_size + self._tile_size:
                print(f'{Fore.YELLOW}Raster size ({x} x{y})is smaller than input tile + 2*buffer size ({self._tile_size}, 2*{self._buffer_size}), \
                      please change your tile and buffer size')
                return False
            else:
                return True
        
    



class Preprocess: 
    """Preprocess module for image IO, tilling.
    
    Use to tile input raster into certain give size tiles, add buffer around tiles to reduce edge effect. 
    """
    def __init__(self,
                 fpth: str,
                 output_path: str = '.',
                 tile_size : int = 236,
                 buffer_size : int = 10,
                 tile_mode = 'meter',
                 band_mode = 'BGR', 
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
        self.fodataset = fo.Dataset(name=self.fpth.stem)
        self._output_path = Path(output_path).absolute()
        self._output_path = self._output_path / self.fpth.stem
        self._annotations_path = self._output_path / 'annotations'
        self._img_path = self._output_path / 'imgs'
        self._raster_path = self._output_path / 'rasters'
        self._output_path.mkdir(exist_ok=True, parents=True)
        self._annotations_path.mkdir(exist_ok=True, parents=True)
        self._img_path.mkdir(exist_ok=True, parents=True)
        self._raster_path.mkdir(exist_ok=True, parents=True)
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
        self.band_mode = band_mode
        self.tile_mode = tile_mode
        # ------
        # run image preprocessing
        # ------
        if self._raster_gt_tile_size():
            self.tile_image()
        else:
            oridataset = rio.open(self.fpth)
            profile = oridataset.profile.copy()
            if self.band_mode == 'BGR':
                band = 3
            elif self.band_mode == 'MS':
                band = oridataset.count
            sample = fo.Sample(filepath=self.fpth.as_posix(),
                               metadata=fo.ImageMetadata(width=profile['width'], height=profile['height'],num_channels=band),
                               ground_truth=fo.Detections(
                                   detections=[
                                       
                                   ]
                                   
                               ))
            y = profile['height']
            x = profile['width']
             
            profile.update({'count': band})
            transform = oridataset.profile['transform']
            self._images['id'].append(1)
            filename = f'{self._output_path}/{self.fpth.stem}_preprocess.tif'
            self._images['file_name'].append(filename)
            self._images['width'].append(x)
            self._images['height'].append(y)
            self._images['date_captured'].append(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            data = oridataset.read()[0:band,:,:]
            self.to_file(filename, data, profile, mode=self.band_mode)
            self.to_png(data, Path(filename).with_suffix('.png').as_posix())
            data = data.reshape(1, band, y, x)
            mean, std = get_mean_std(data)
            self._report = {'file_name': self.fpth.as_posix(),
                            'output_path': self._output_path.as_posix(),
                            'tile_size': x,
                            'buffer_size': 0,
                            'tile_numbers': 1,
                        'original_size': str(rio.open(self.fpth).shape),
                        'buffed_size': str(oridataset.shape),
                        'crs': str(oridataset.crs.to_epsg()),
                        'band': band,
                        'affine': (transform.a, 
                                transform.b,
                                transform.c,
                                transform.d,
                                transform.e,
                                transform.f),
                        'driver': oridataset.profile['driver'],
                        'dtype':  oridataset.profile['dtype'],
                        'nodata': oridataset.profile['nodata'],
                        'pixel_mean' : [float(i) for i in mean],
                        'pixel_std' : [float(i) for i in std],
                        'mode': self.band_mode
                        }

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

    def _raster_gt_tile_size(self):
        """Check if raster size is larger than tile size, if not, raise error."""
        with rio.open(self.fpth) as dataset:
            x = dataset.width * dataset.res[0]
            y = dataset.height * dataset.res[1]
            if x < self._tile_size + 2 * self._buffer_size or y < 2 * self._buffer_size + self._tile_size:
                print(f'{Fore.YELLOW}Raster size ({x} x{y})is smaller than input tile + 2*buffer size ({self._tile_size}, 2*{self._buffer_size}), \
                      please change your tile and buffer size')
                return False
            else:
                return True
        
    def tile_image(self, shp_path: str = None):
        """Cut input image into square tiles with buffer and preserver geoinformation for each tile."""         
        if self.band_mode == 'BGR':
            band = 3
        elif self.band_mode == 'MS':
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
            filename_raster = f'{self._raster_path}/{self.fpth.stem}_row{window.row_off}_col{window.col_off}.tif'
            filename_img = f'{self._img_path}/{self.fpth.stem}_row{window.row_off}_col{window.col_off}.png'
            self._images['file_name'].append(filename_raster)
            self._images['width'].append(window.width)
            self._images['height'].append(window.height)
            self._images['date_captured'].append(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            self.to_file(filename_raster, tile_data, tile_profile, mode=self.band_mode)
            self.to_png(tile_data, filename_img)
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
                        'mode': self.band_mode
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

    def to_COCO(self, **kwargs) -> str:
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
        self._coco.add_categories(id = [1], name=['shrub'], supercategory=['plant']) #TODO No hard coded for category in the furture
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
        self._coco_img = self._coco
        for image in self._coco_img.dict['images']:
            image['file_name'] = image['file_name'].replace('.tif', '.png')
        
       
      
        coco_path = f'{self._annotations_path}/{self.fpth.stem}_coco_raster.json'
        coco_img_path = f'{self._annotations_path}/{self.fpth.stem}_coco_img.json'
        self._coco.save_coco(coco_path)
        self._coco_img.save_coco(coco_img_path)
        print(f'COCO_img saved at {coco_img_path}\nCOCO_raster saved at {coco_path}')
              
        return coco_path, coco_img_path
    
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
        stack = np.stack([band3, band2, band1], axis=2)
        stack[stack <-1000] = 0 # Remove negative value in the raster that represent empty value 
        min_val = np.min(stack)  # Minimum value in the array
        max_val = np.max(stack)
        if max_val == min_val:
            normalized_array = np.zeros_like(stack, dtype=np.uint8)  # or np.ones_like(array, dtype=np.float32)
        else:
            normalized_array = ((stack - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        image = Image.fromarray(normalized_array, mode='RGB')
        image.save(path_to_file, dpi=(300,300))
       

 