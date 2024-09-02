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