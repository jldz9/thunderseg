#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IO related module for DLtreeseg
"""
import json
import sys
from pathlib import Path
import tomllib
import tomli_w

import cv2
import h5py
import numpy as np
import rasterio as rio



def create_project_structure(workdir: str) -> dict:
    """
    Create the necessary file structure for detectron2.
    Args: 
        Workdir: The root dir use to create directory strucutre
    Returns:
        A SimpleNamespace with .annotations, .train, .train_shp, .val, .val_shp, .test, .result
    """
    workdir = Path(workdir).resolve()
    directories = dict(
            ANNOTATIONS = workdir / "datasets" / "annotations",
            TRAIN = workdir / "datasets" / "train",
            TRAIN_SHP = workdir / "datasets" / "train" / "shp",
            VAL = workdir / "datasets" / "val",
            VAL_SHP = workdir / "datasets" / "val" / "shp",
            PREDICT = workdir / "datasets" / "predict",
            RESULT = workdir / "results",
            TEMP = workdir / "temp"
    )
    for key, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created {key} folder at {directory}")
    return {k:v.as_posix() for k, v in directories.items()}

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
