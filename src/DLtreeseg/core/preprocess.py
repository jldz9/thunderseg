#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image preprocess module for DLtreeseg, include all necessary image IO, tilling
"""
import io
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio as rio
from rasterio.io import DatasetReader
from rasterio.windows import Window
from rasterio.transform import Affine

from utils.tool import h5_list
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
        self.tile_image()
        if debug is True:
            print('debug')
    
    
    def tile_image(self):
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

        # Make meshgrid for x and y label of all tiles
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
            
        # Save tiles to h5
        
        tile_stack = []
        profile_stack = []
        tile_profile = self.dataset_padded.profile
        for idx, window in enumerate(windows):
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
        tile_total = np.stack(tile_stack, axis=0)
        windows_h5 = h5_list(windows)
        profiles_h5 = h5_list(np.array(profile_stack))
        report = {'file name': self.fpth.name,
                'tile size': self.tile_size,
                'buffer size': self.buffer_size,
                'total tiles': num_tiles,
                'original size': str(self.dataset.shape),
                'buffed size': str(self.dataset_padded.shape),
                'crs': str(self.dataset_padded.crs),
                'band': self.dataset_padded.count
                }
        with h5py.File(f'{self.output_path}/{self.fpth.stem}_tiles.h5', 'w') as hf:
            dset = hf.create_dataset(f'tiles', data=tile_total)
            hf.create_dataset(f'windows', data=windows_h5)
            hf.create_dataset(f'profiles', data=profiles_h5)
            for key, value in report.items():
                dset.attrs[key] = value
        print(f'\n Tiles saved to {self.output_path}')
        
    