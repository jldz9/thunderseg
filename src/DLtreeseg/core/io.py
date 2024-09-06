#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IO related module for DLtreeseg
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio as rio
import torch

# save hdf5 to local drive
def save_h5(save_path:str, data:np.ndarray, attrs:dict = None, **kwarg):
    """
    Use of h5py to storage data to local disk. **kwarg should contains packed binary data from
    function pack_h5_list.
    """
    save_path = Path(save_path)
    if save_path.is_file():
        while True:
            rmfile = input(f'File {save_path.name} exist, do you want to open this dataset? Y(es)/O(verwrite)/C(ancel) ')
            if rmfile.lower() == 'y':
                break
            if rmfile.lower() == 'o':
                save_path.unlink()
                break
            if rmfile.lower() == 'c':
                sys.exit()
    with h5py.File(save_path, 'r+') as hf:
        if save_path.stem in hf:
            while True:
                rmgroup = input(f'Group {save_path.stem} exist, do you want to overwrite? Y(es)/N(o) ')
                if rmgroup.lower() == 'y':
                    del hf[save_path.stem]
                    break
                if rmgroup.lower() == 'n':
                    sys.exit()
        grp = hf.create_group(save_path.stem)
        grp.create_dataset('data', data=data)
        if attrs is not None:
            for key, value in attrs.items():
                grp.attrs[key] = value
        if kwarg:
            for key, value in kwarg.items():
                grp.create_dataset(key, data=value)
    print(f'{save_path} saved!')

# save file to local gis
def save_gis(path_to_file:str, data:np.ndarray, profile):
    with rio.open(path_to_file, 'w', **profile) as dst:
        dst.write(data)
        sys.stdout.write(f'\rExport {path_to_file.name}')
        sys.stdout.flush()
