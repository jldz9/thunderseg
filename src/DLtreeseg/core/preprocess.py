#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image preprocess module for DLtreeseg, include image IO, tilling
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.io import DatasetReader, MemoryFile
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling

from DLtreeseg.utils import pack_h5_list
from DLtreeseg.core import save_h5, save_gis

class Tile: 
    """
    Preprocess module for image IO, tilling.
    """
    def __init__(self,
                 fpth: str,
                 output_path: str = '.',
                 tile_size : int = 256,
                 buffer_size : int = 10,
                 debug = False
                 ):
        """
        tile_size: The tile size of individual tiles, all tiles will be cut into squares. Unit: meter
        buffer_size: Buffer around tiles. Unit:meter
        """
        self.fpth = Path(fpth).absolute()
        self._dataset = rio.open(self.fpth)
        self.output_path = Path(output_path).absolute()
        self.output_path.mkdir(exist_ok=True)
        self._tile_size = tile_size
        if not isinstance(self._tile_size, int):
            raise TypeError(f"Tile_size tuple should be Int, not {type(self.tile_width)}")
        self._buffer_size = buffer_size
        if not isinstance(self._buffer_size, int) :
            raise TypeError(f"Buffer size should be Int, not {type(self._buffer_size_pixel_x)}")
        if debug is True:
            print('debug')
        # Run when init

    @staticmethod
    def _get_window(dataset: DatasetReader, tile_size, buffer_size, ):
        profile = dataset.profile.copy()
        y = profile['height']
        x = profile['width']
        transform = dataset.profile['transform']
        # Convert meter tile size to pixel tile size
        tile_size_pixel_x = int(np.ceil(tile_size / dataset.res[0]))
        tile_size_pixel_y = int(np.ceil(tile_size / dataset.res[1]))
        buffer_size_pixel_x = int(np.ceil(buffer_size / dataset.res[0]))
        buffer_size_pixel_y = int(np.ceil(buffer_size / dataset.res[1]))
        # Calculate number of tiles along height and width with buffer.
        n_tiles_x = int(np.ceil((x + buffer_size_pixel_x) / (buffer_size_pixel_x + tile_size_pixel_x)))
        n_tiles_y = int(np.ceil((y + buffer_size_pixel_y) / (buffer_size_pixel_y + tile_size_pixel_y)))
        
        # Add buffer to original raster to make sure every tiles has same size.
        data = dataset.read()
        data = np.where(data<0, 0, data)
        pad = ((0,0),
                (buffer_size_pixel_y, n_tiles_y * (tile_size_pixel_y + buffer_size_pixel_y) - y),
                (buffer_size_pixel_x, n_tiles_x * (tile_size_pixel_x + buffer_size_pixel_x) - x)
                )
        padded_data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
        profile.update({
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
        windows = np.array([
            Window(
            max(((start_x * (tile_size_pixel_x + (2 * buffer_size_pixel_x)) - start_x * buffer_size_pixel_x), 0)),
            max(((start_y * (tile_size_pixel_y + (2 * buffer_size_pixel_y)) - start_y * buffer_size_pixel_x), 0)),
            tile_size_pixel_x + 2 * buffer_size_pixel_x,
            tile_size_pixel_y + 2 * buffer_size_pixel_y,
            ) 
            for start_x, start_y in zip(flat_tile_x, flat_tile_y)
            ])
        dataset.close()
        return padded_data, profile, windows

    def resample(self, new_resolution):
        print(f'Resampling raster to {new_resolution} m')
        profile = self._dataset.profile.copy()
        old_transform = profile['transform']
        new_width= int(np.round((self._dataset.bounds.right - self._dataset.bounds.left) / new_resolution))
        new_height = int(np.round((self._dataset.bounds.top - self._dataset.bounds.bottom) / new_resolution))
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': Affine(new_resolution, old_transform.b, old_transform.c, 
                                old_transform.d, -new_resolution, old_transform.f)

        })
        data = self._dataset.read(
            out_shape = (self._dataset.count, new_height, new_width),
            resampling = Resampling.gauss
        )
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(data)
            self._dataset = memfile.open()

    def tile_image(self):
        """
        Cut input image into square tiles with buffer and preserver geoinformation for each tile. 
        """
        padded_data, profile, self._windows = Tile._get_window(self._dataset, self._tile_size, self._buffer_size)
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(padded_data)

            self._dataset = memfile.open()
                   
        # loop stack all tiles into one (tiles, channels, rows, columns) matrix
            tiles_list = []
            self._profiles = []
            self._coco_dicts = []
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
                sys.stdout.write(f'\rWorking on: {idx+1}/{num_tiles} tile')
                sys.stdout.flush()
            print()
            self._stack_tiles = np.stack(tiles_list, axis=0)
            self._report = {'file name': self.fpth.name,
                    'tile size': self._tile_size,
                    'buffer size': self._buffer_size,
                    'total tiles': num_tiles,
                    'original size': str(rio.open(self.fpth).shape),
                    'buffed size': str(self._dataset.shape),
                    'crs': str(self._dataset.crs),
                    'band': self._dataset.count
                    }
    
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
        return print()
    
    def to_gis(self, path_to_dir:str):
        """
        Save individual tiles into raster
        """
        path_to_dir = Path(path_to_dir)
        path_to_dir.mkdir(parents=True, exist_ok=True)
        tiles = [self._stack_tiles[i] for i in range(self._stack_tiles.shape[0])]
        for data, window, profile in zip(tiles, self._windows, self._profiles):
            tile_path = Path(f'{path_to_dir}/{self.fpth.stem}_{window.row_off}_{window.col_off}{self.fpth.suffix}')
            save_gis(tile_path, data, profile)
        return print()

    