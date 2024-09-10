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
from rasterio.io import DatasetReader
from rasterio.windows import Window
from rasterio.transform import Affine

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
        self.tile_size = tile_size
        if not isinstance(self.tile_size, int):
            raise TypeError(f"Tile_size tuple should be Int, not {type(self.tile_width)}")
        self.buffer_size = buffer_size
        if not isinstance(self.buffer_size, int) :
            raise TypeError(f"Buffer size should be Int, not {type(self._buffer_size_pixel_x)}")
        if debug is True:
            print('debug')
        # Run when init
        self._tile_image()
    
    def _tile_image(self):
        """
        Cut input image into square tiles with buffer and preserver geoinformation for each tile. 
        """
        profile = self._dataset.profile.copy()
        y = profile['height']
        x = profile['width']
        transform = self._dataset.profile['transform']
        # Convert meter tile size to pixel tile size
        self._tile_size_pixel_x = int(np.ceil(self.tile_size / self._dataset.res[0]))
        self._tile_size_pixel_y = int(np.ceil(self.tile_size / self._dataset.res[1]))
        self._buffer_size_pixel_x = int(np.ceil(self.buffer_size / self._dataset.res[0]))
        self._buffer_size_pixel_y = int(np.ceil(self.buffer_size / self._dataset.res[1]))
        # Calculate number of tiles along height and width with buffer.
        self._n_tiles_x = int(np.ceil((x + self._buffer_size_pixel_x) / (self._buffer_size_pixel_x + self._tile_size_pixel_x)))
        self._n_tiles_y = int(np.ceil((y + self._buffer_size_pixel_y) / (self._buffer_size_pixel_y + self._tile_size_pixel_y)))
        
        # Add buffer to original raster to make sure every tiles has same size.
        data = self._dataset.read()
        data = np.where(data<0, 0, data)
        pad = ((0,0),
                (self._buffer_size_pixel_y, self._n_tiles_y * (self._tile_size_pixel_y + self._buffer_size_pixel_y) - y),
                (self._buffer_size_pixel_x, self._n_tiles_x * (self._tile_size_pixel_x + self._buffer_size_pixel_x) - x)
                )
        padded_data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
        profile.update({
            'height': padded_data.shape[1],
            'width': padded_data.shape[2],
            'transform': Affine(transform[0],transform[1], transform[2]- self._buffer_size_pixel_x*transform[0],
                                transform[3],transform[4], transform[5]- self._buffer_size_pixel_x*transform[4])
        })

        # write new dataset into RAM
        with io.BytesIO() as memfile:
            with rio.open(memfile, 'w', **profile) as dst:
                dst.write(padded_data)
            memfile.seek(0)
            self._dataset_padded = rio.open(memfile) 

        # Make meshgrid to create 2d (x,y) index of all tiles
        tile_index_x, tile_index_y = np.meshgrid(np.arange(self._n_tiles_x), np.arange(self._n_tiles_y))
        flat_tile_x = tile_index_x.flatten()
        flat_tile_y = tile_index_y.flatten()
        num_tiles = len(flat_tile_x)

        # Make windows for all tiles.
        self._windows = np.array([
            Window(
            max(((start_x * (self._tile_size_pixel_x + (2 * self._buffer_size_pixel_x)) - start_x * self._buffer_size_pixel_x), 0)),
            max(((start_y * (self._tile_size_pixel_y + (2 * self._buffer_size_pixel_y)) - start_y * self._buffer_size_pixel_x), 0)),
            self._tile_size_pixel_x + 2 * self._buffer_size_pixel_x,
            self._tile_size_pixel_y + 2 * self._buffer_size_pixel_y,
            ) 
            for start_x, start_y in zip(flat_tile_x, flat_tile_y)
            ])
            
        # loop stack all tiles into one (tiles, channels, rows, columns) matrix
        tiles_list = []
        self._profiles = []
        self._coco_dicts = []
        for idx, window in enumerate(self._windows):
            tile_profile = self._dataset_padded.profile
            tile_data = self._dataset_padded.read(window=window)
            tile_profile.update({
            'transform': self._dataset_padded.window_transform(window),
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
                'tile size': self._tile_size_pixel_x,
                'buffer size': self._buffer_size_pixel_x,
                'total tiles': num_tiles,
                'original size': str(self._dataset.shape),
                'buffed size': str(self._dataset_padded.shape),
                'crs': str(self._dataset_padded.crs),
                'band': self._dataset_padded.count
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
        return self._dataset_padded
    
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

    