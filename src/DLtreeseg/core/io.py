#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IO related module for DLtreeseg
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import geopandas as gpd
from geopandas import GeoDataFrame
import h5py
import numpy as np
import rasterio as rio
import torch


def create_project_structure(workdir: str) -> SimpleNamespace:
    """
    Create the necessary file structure for detectron2.
    Args: 
        Workdir: The root dir use to create directory strucutre
    Returns:
        A SimpleNamespace with .annotations, .train, .train_shp, .val, .val_shp, .test, .result
    """
    workdir = Path(workdir)
    directories = dict(
            annotations = workdir / "datasets" / "annotations",
            train = workdir / "datasets" / "train",
            train_shp = workdir / "datasets" / "train" / "shp",
            val = workdir / "datasets" / "val",
            val_shp = workdir / "datasets" / "val" / "shp",
            test = workdir / "datasets" / "test",
            result = workdir / "results"
    )
    for key, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created {key} at {directory}")
    return SimpleNamespace(**directories)

def save_h5(save_path:str, data:np.ndarray = [], attrs:dict = None, **kwarg):
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

def to_file(path_to_file:str, data:np.ndarray, profile=None, mode:str='rgb'):
    path_to_file = Path(path_to_file)
    if mode.lower() == 'rgb':
        with rio.open(path_to_file,'w', **profile) as dst:
            for i in range(0,3):
                dst.write(data[i], i+1)
    if mode == 'ms':
        with rio.open(path_to_file, 'w', **profile) as dst:
            dst.write(data)

def to_png(data: np.ndarray, path_to_file:str):
        band1 = data[0] # B
        band2 = data[1] # G
        band3 = data[2] # R
        stack = np.stack([band1, band2, band3], axis=0)
        min_val = np.min(stack)  # Minimum value in the array
        max_val = np.max(stack)
        if max_val == min_val:
            normalized_array = np.zeros_like(stack, dtype=np.float32)  # or np.ones_like(array, dtype=np.float32)
        else:
            normalized_array = (stack - min_val) / (max_val - min_val) * 255
        
        array_bgr = cv2.cvtColor(normalized_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path_to_file, array_bgr)