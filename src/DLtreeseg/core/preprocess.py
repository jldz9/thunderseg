#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image preprocess module for DLtreeseg, include all necessary image IO, tilling
"""
import io
import sys
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.io import DatasetReader
from rasterio.windows import Window
from rasterio.transform import Affine

from utils.tool import pack_h5_list, save_h5
class Preprocess: 
    """
    Preprocess class for image IO, tilling.
    """
    def __init__(self,
                 fpth: str,
                 output_path: str = '.',
                 tile_size : int = 256,
                 buffer_size : int = 10,
                 debug = False
                 ):
        self.fpth = Path(fpth).absolute()
        self.dataset = rio.open(self.fpth)
        self.output_path = Path(output_path).absolute()
        self.output_path.mkdir(exist_ok=True)
        self.tile_size = tile_size
        if not isinstance(self.tile_size, int):
            raise TypeError(f"Tile_size tuple should be Int, not {type(self.tile_width)}")
        self.buffer_size = buffer_size
        if not isinstance(self.buffer_size, int) :
            raise TypeError(f"Buffer size should be Int, not {type(self.buffer_size)}")
        if debug is True:
            print('debug')
        # Run when init
        self._tile_image()
    
    
    def _tile_image(self):
        """
        Cut input image into square tiles with buffer and preserver geoinformation for each tile. 
        """
        profile = self.dataset.profile.copy()
        height = profile['height']
        width = profile['width']
        transform = self.dataset.profile['transform']
        # Calculate number of tiles along height and width with buffer.
        n_tiles_x = int(np.ceil((width + self.buffer_size) / (self.tile_size + self.buffer_size)))
        n_tiles_y = int(np.ceil((height + self.buffer_size) / (self.tile_size + self.buffer_size)))
        
        # Add buffer to original raster to make sure every tiles has same size.
        data = self.dataset.read()
        data = np.where(data<0, 0, data)
        pad = ((0,0),
                (self.buffer_size, n_tiles_y * (self.tile_size + self.buffer_size) - height),
                (self.buffer_size, n_tiles_x * (self.tile_size + self.buffer_size) - width)
                )
        padded_data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
        profile.update({
            'height': padded_data.shape[1],
            'width': padded_data.shape[2],
            'transform': Affine(transform[0],transform[1], transform[2]- self.buffer_size*transform[0],
                                transform[3],transform[4], transform[5]- self.buffer_size*transform[4])
        })
        with io.BytesIO() as memfile:
            with rio.open(memfile, 'w', **profile) as dst:
                dst.write(padded_data)
            memfile.seek(0)
            self.dataset_padded = rio.open(memfile) 

        # Make meshgrid to create 2d (x,y) index of all tiles
        tile_indices_x = np.arange(n_tiles_x)
        tile_indices_y = np.arange(n_tiles_y)
        tile_x, tile_y = np.meshgrid(tile_indices_x, tile_indices_y)
        flat_tile_x = tile_x.flatten()
        flat_tile_y = tile_y.flatten()
        num_tiles = len(flat_tile_x)

        # Make windows for all tiles.
        windows = np.array([
            Window(
            max(((start_x * (self.tile_size + (2 * self.buffer_size)) - start_x * self.buffer_size), 0)),
            max(((start_y * (self.tile_size + (2 * self.buffer_size)) - start_y * self.buffer_size), 0)),
            self.tile_size + 2 * self.buffer_size,
            self.tile_size + 2 * self.buffer_size,
            ) 
            for start_x, start_y in zip(flat_tile_x, flat_tile_y)
            ])
            
        # loop stack all tiles into one (tiles, channels, rows, columns) matrix
        tile_stack = []
        profile_stack = []
        for idx, window in enumerate(windows):
            tile_profile = self.dataset_padded.profile
            tile_data = self.dataset_padded.read(window=window)
            tile_profile.update({
            'transform': self.dataset_padded.window_transform(window),
            'height': window.height,
            'width': window.width,
        })
            tile_stack.append(tile_data)
            profile_stack.append(tile_profile)
            sys.stdout.write(f'\rWorking on: {idx+1}/{num_tiles} tile')
            sys.stdout.flush()
        print()
        tile_total = np.stack(tile_stack, axis=0)
        report = {'file name': self.fpth.name,
                'tile size': self.tile_size,
                'buffer size': self.buffer_size,
                'total tiles': num_tiles,
                'original size': str(self.dataset.shape),
                'buffed size': str(self.dataset_padded.shape),
                'crs': str(self.dataset_padded.crs),
                'band': self.dataset_padded.count
                }
        self._stack_tiles = tile_total
        self._windows = windows
        self._profiles = profile_stack
        self._report = report
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
        return self.dataset_padded
    
    def write_h5(self, save_path:str):
        """
        Save tiles into HDF5 dataset
        """
        save_h5(save_path=save_path,
                data = self._stack_tiles,
                attrs= self._report,
                windows = pack_h5_list(self._windows),
                profiles = pack_h5_list(self._profiles)
                )
    
    def write_gis(self, path_to_dir:str):
        """
        Save individual tiles into raster
        """
        path_to_dir = Path(path_to_dir)
        path_to_dir.mkdir(parents=True, exist_ok=True)
        tiles = [self._stack_tiles[i] for i in range(self._stack_tiles.shape[0])]
        for data, window, profile in zip(tiles, self._windows, self._profiles):
            tile_path = f'{path_to_dir}/{self.fpth.stem}_{window.row_off}_{window.col_off}.{self.fpth.suffix}'
            with rio.open(tile_path, 'w', **profile) as dst:
                dst.write(data)
        print(f'Tiles exported in {path_to_dir}')

    