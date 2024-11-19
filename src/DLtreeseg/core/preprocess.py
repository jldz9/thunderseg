#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image preprocess module for DLtreeseg, include image IO, tilling
"""

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import geopandas as gpd
import lightning as L
import numpy as np
from pycocotools.coco import COCO
import rasterio as rio
from rasterio.io import DatasetReader, MemoryFile
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely import box
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from DLtreeseg.utils import pack_h5_list, to_pixelcoord, COCO_parser, window_to_dict, get_mean_std, assert_json_serializable, bbox_from_mask
from DLtreeseg.core import save_h5, to_file



class Tile: 
    """Preprocess module for image IO, tilling.
    
    Use to tile input raster into certain give size tiles, add buffer around tiles to reduce edge effect. 
    """
    def __init__(self,
                 fpth: str,
                 output_path: str = '.',
                 tile_size : int = 236,
                 buffer_size : int = 10,
                 debug = False
                 ):
        """ Initializes parameters
        Args:
            fpth: Path to single raster file.
            output_path: Output file path, default is current directory.
            tile_size: Tile size, all tiles will be cut into squares. Unit: meter
            buffer_size: Buffer size around tiles. Unit:meter
            debug: Switch to turn on debug
        """
        self.fpth = Path(fpth).absolute()
        self._output_path = Path(output_path).absolute()
        self._output_path.mkdir(exist_ok=True)
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
        
        if debug is True:
            print('debug')

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
        tile_size_pixel_x = int(np.ceil(self._tile_size / oridataset.res[0]))
        tile_size_pixel_y = int(np.ceil(self._tile_size / oridataset.res[1]))
        buffer_size_pixel_x = int(np.ceil(self._buffer_size / oridataset.res[0]))
        buffer_size_pixel_y = int(np.ceil(self._buffer_size / oridataset.res[1]))
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

    def resample(self, new_resolution: float):
        """Resample original raster to certain resolution (meter).
        Args: 
            new_resolution: the resolution of new raster
        """
        print(f'Resampling raster to {new_resolution} m')
        with rio.open(self.fpth) as ori_dataset:
            ori_dataset = rio.open(self.fpth)
            profile = ori_dataset.profile.copy()
            old_transform = profile['transform']
            new_width= int(np.round((ori_dataset.bounds.right - ori_dataset.bounds.left) / new_resolution))
            new_height = int(np.round((ori_dataset.bounds.top - ori_dataset.bounds.bottom) / new_resolution))
            profile.update({
                'height': new_height,
                'width': new_width,
                'transform': Affine(new_resolution, old_transform.b, old_transform.c, 
                                    old_transform.d, -new_resolution, old_transform.f)

            })
            data = ori_dataset.read(
                out_shape = (ori_dataset.count, new_height, new_width),
                resampling = Resampling.gauss
            )
        memfile = MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(data)
        self._dataset = memfile.open()

    def tile_image(self, mode='BGR', shp_path: str = None):
        """Cut input image into square tiles with buffer and preserver geoinformation for each tile."""         
        if mode == 'BGR':
            band = 3
        elif mode == 'MS':
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
            filename = f'{self._output_path}/{self.fpth.stem}_row{window.row_off}_col{window.col_off}.tif'
            self._images['file_name'].append(filename)
            self._images['width'].append(window.width)
            self._images['height'].append(window.height)
            self._images['date_captured'].append(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            to_file(filename, tile_data, tile_profile, mode=mode)
        print()
        self._stack_tiles = np.stack(tiles_list, axis=0)
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
                        'mode': mode
                        }
        if shp_path is not None and Path(shp_path).is_file():
            self.tile_shape(shp_path)
   
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

    def to_COCO(self, output_path: str = None, **kwargs) -> str:
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
        self._coco.add_categories(id = [1], name=['shurb'], supercategory=['plant'])
        self._coco.add_licenses(license_id=[1],license_url=[''],license_name=[''])
        self._coco.add_images(id = self._images['id'], 
                        file_name = self._images['file_name'],
                        width = self._images['width'],
                        height = self._images['height'],
                        date_captured = self._images['date_captured'],
                        window = [window_to_dict(w) for w in self._windows]
                        )
        self._coco.add_annotations(id = self._annotations['id'],
                             image_id = self._annotations['image_id'],
                             category_id = self._annotations['category_id'],
                             bbox = self._annotations['bbox'],
                             area = self._annotations['area'],
                             iscrowd = self._annotations['iscrowd'],
                             segmentation = self._annotations['segmentation']
                             )
        if output_path is not None:
            self._coco_path = Path(output_path)
        else:
            self._coco_path = f'{self._output_path}/{self.fpth.stem}_coco.json'
        self._coco.save_json(self._coco_path)
        print(f'COCO saved at {self._coco_path}')
        return self._coco_path
    
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

def get_transform(image:np.ndarray, target:dict={}, train = True, mean:list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
    """
    Apply transform to both image and target using Albumentations, 
    Args:
        image: should be a numpy array with shape of (Height,Width,Channel)
        target: should be a dict contains bbox, mask, 
    """

    three_channel_image_only_transform = A.Compose(
        [A.SomeOf([ 
        A.PlanckianJitter(),
        A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5)),
        A.RandomToneCurve(),
        ], n=1, p=0.5)
        ])
    
    image_only_transform = A.Compose([A.SomeOf([
        #A.Downscale(scale_range=(0.5, 1)),
        A.GaussNoise(noise_scale_factor=0.5),
        A.Sharpen(),
        A.AdvancedBlur(),
        A.Defocus(),
        A.MotionBlur(allow_shifted=False)
    ], n=2, p=0.5),
    A.Normalize(mean=mean, std=std, max_pixel_value=1)])

    image_and_target_transform = A.Compose([A.SomeOf([
        A.PixelDropout(),
        
        A.HorizontalFlip(),
        A.RandomRotate90(),
    ], n=2, p=0.5),
    A.RandomCrop(height=512, width=512),
    ToTensorV2()])
    if train is True:
        if image.shape[2] == 3 and image.shape[0] > image.shape[2] and image.shape[1] > image.shape[2]: 
            temp = three_channel_image_only_transform(image=image)
            image = temp['image']
        temp = image_only_transform(image=image)
        image = temp['image']
        temp = image_and_target_transform(image=image, 
                                        masks=target['masks']
                                        )
        image = temp['image']
        target['area'] = torch.tensor([int(np.sum(mask.numpy())) for mask in temp['masks']])
        drop_index = np.where(target['area'].numpy()<10)[0]
        target['area'] = [j for i, j in enumerate(target['area']) if i not in list(drop_index)]
        if len(target['area']) >1:
            target['area'] = torch.tensor(target['area'])
            target['annotation_id'] = [j for i, j in enumerate(target['annotation_id']) if i not in list(drop_index)]
            target['masks'] = torch.stack([j for i, j in enumerate(temp['masks']) if i not in list(drop_index)])
            target['boxes'] = torch.tensor([bbox_from_mask(mask.numpy()) for mask in target['masks']])
            target['bbox_mode'] = ['xyxy']* len(target['area'])
            target['iscrowd'] = [int(j) for i, j in enumerate(target['iscrowd'].numpy()) if i not in list(drop_index)]
            target['labels'] = torch.tensor([int(j) for i, j in enumerate(target['labels'].numpy()) if i not in list(drop_index)])
            return image, target
        else:
            target['area'] = torch.zeros((0,),dtype=torch.int64)
            target['annotation_id'] = []
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            target['bbox_mode'] = ['xyxy']
            target['iscrowd'] = []
            return image, target
        
    elif train is False:
        predict_transform = A.Compose([A.Normalize(mean=mean, std=std, max_pixel_value=1),
                                       ToTensorV2()])
        temp = predict_transform(image=image)
        image = temp['image']
        return image
    
class TrainDataset(Dataset):
    def __init__(self, coco:str, transform=get_transform):
        """
        Args:
            coco : The single json file path exported from Tile.to_COCO represent the image dataset
            img_dir : Not required, normally obtained from COCO file, will find image under it if specified
            transform : transform fron torchvision.transforms
        """
        self._coco = COCO(coco)
        # Get parent path of the first availiable image in the dataset
        self._img_dir = Path(self._coco.imgs[self._coco.getImgIds()[0]]['file_name']).parent.as_posix()
        self._transform = transform
    def __len__(self):
        return len(self._coco.imgs)

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = len(self._coco.imgs)
        while attempts< max_attempts:
            image_info = self._coco.imgs[idx+1] # pycocotools use id number which starts from 1.
            annotation_ids = self._coco.getAnnIds(imgIds=idx+1)
            if len(annotation_ids) > 0:
                image, target = self._load_image_target(image_info, annotation_ids)
                if self._transform is not None:
                    image, target= self._transform(image, target, train = True, 
                                                   mean=self._coco.dataset['info']['pixel_mean'],
                                                   std=self._coco.dataset['info']['pixel_std'])
                    return image, target
            else:
                idx = (idx+1)% len(self._coco.imgs)
                attempts += 1

    def _load_image_target(self, image_info, annotation_ids):
        with rio.open(image_info['file_name']) as f:
            image = f.read()
        image_hwc = np.transpose(image, (1,2,0))
        anns = self._coco.loadAnns(annotation_ids)
        target = {}
        # ID
        target["image_id"] = image_info['id']
        target['annotation_id'] = [ann['id'] for ann in anns]
        # Bboxes
        target["boxes"] = [ann['bbox'] for ann in anns]
        target['bbox_mode'] = [ann['bbox_mode'] for ann in anns]

        # Masks
        masks = [self._coco.annToMask(ann) for ann in anns]
        target['masks'] = masks
        
        # Labels
        labels = [ann['category_id'] for ann in anns]
        target['labels'] = torch.tensor(labels)

        # Area
        areas = [ann['area'] for ann in anns]
        target['area'] = areas

        # Iscrowd
        iscrowd = [ann['iscrowd'] for ann in anns]
        target['iscrowd'] = torch.tensor(iscrowd)
        
        return image_hwc, target
 
    def _xywh_to_xyxy(self, single_bbox:list):
        """Convert [x, y, width, height] to [xmin, ymin, xmax, ymax]"""
        
        return [single_bbox[0], single_bbox[1], single_bbox[0]+single_bbox[2], single_bbox[1]+single_bbox[3]]

class PreditDataset(Dataset):
    """Predict Dataset with no target export"""
    def __init__(self, coco_train:str, coco_predict:str, transform=get_transform):
        """ 
        Args:
            coco_train
            coco_predict : The single json file path exported from Tile.to_COCO represent the predict image dataset
            img_dir : Not required, normally obtained from COCO file, will find image under it if specified
            transform : transform fron torchvision.transforms
        """
        self._train_coco = COCO(coco_train)
        self._coco = COCO(coco_predict)
        self._img_dir = Path(self._coco.imgs[self._coco.getImgIds()[0]]['file_name']).parent.as_posix()
        self._transform = transform
    
    def __len__(self):
        return len(self._coco.imgs)

    def __getitem__(self, idx):
        with rio.open(self._coco.imgs[idx+1]['file_name']) as f:
            image = f.read()
        
        if self._transform:
            image = self._transform(image, train=False, 
                                    mean=self._coco_train.dataset['info'][0]['total_mean'],
                                    std=self._coco_train.dataset['info'][0]['total_std'])

        return image

class LoadDataModule(L.LightningDataModule):
    def __init__(self, train_coco, 
                 predict_coco = None,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 transform=get_transform):
        super().__init__()
        self.train_coco = train_coco
        self.predict_coco = predict_coco
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def prepare_data(self):
        pass

    def setup(self, stage=None, train_pct:float=0.8, val_pct:float=0.1, test_pct:float=0.1):
        dataset = TrainDataset(coco = self.train_coco, transform=self.transform)
        if train_pct+val_pct+test_pct != 1.0: 
            test_pct = 1 - (train_pct + val_pct)
            warnings.warn(f'The sum of train, validate, and test percent are greater than %100, setting test set to {test_pct*100}%')
        train_size = int(train_pct*len(dataset))
        val_size = int(val_pct*len(dataset))
        test_size = len(dataset) - (train_size + val_size)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        if stage == 'fit':
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        if stage == 'test':
            self.test_dataset = test_dataset
        if stage == 'predict':
            self.predict_dataset = PreditDataset(coco=self.predict_coco, transform=self.transform)
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          collate_fn=LoadDataModule.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=LoadDataModule.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=LoadDataModule.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=LoadDataModule.collate_fn)
    
