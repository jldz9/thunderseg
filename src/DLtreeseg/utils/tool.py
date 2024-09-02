#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tools use by DLtreeseg 
"""
import sys

import pickle
from pathlib import Path
import numpy as np
import h5py

# pack list into numpy scalar data which can be storage into hdf5 dataset
def pack_h5_list(data: list):
    pickle_object = pickle.dumps(data)
    return np.void(pickle_object)
# unpack numpy scalar data to list
def unpack_h5_list(data: np.void):
    return pickle.loads(data.tobytes())

