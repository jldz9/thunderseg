#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tools use by DLtreeseg 
"""
import json
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

import numpy as np
from colorama import Fore, init
from pycocotools.coco import COCO
from rasterio.windows import Window
from shapely import Polygon

# Colorama init for color output
init(autoreset=True)

def assert_json_serializable(**kwargs):
    for key, value in kwargs.items():
        try:
            # Try to serialize the value
            json.dumps(value)
        except (TypeError, ValueError) as e:
            # Raise an assertion error if the value is not serializable
            assert False, f"Value for key '{key}' is not JSON serializable: {e}"

def bbox_from_mask(mask: np.ndarray) -> list:
    """Generate a (x_min, y_min, x_max, y_max) bbox from imput binary mask
    Args:
        mask: A single objective binary mask
    Return:
        bbox: A bbox list [x_min, y_min, x_max, y_max]
    """
    rows = np.any(mask, axis=1)
    if not rows.any():
        return [-1, -1, -1, -1]
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [x_min, y_min, x_max+1, y_max+1]

def float_to_binary_mask(mask: np.ndarray,  threshold: float = 0.5) -> np.ndarray:
    '''Convert float mask from prediction into binary mask to create polygons'''
    return (mask > threshold).astype(np.uint8)

def recal_pixel_mean_std(n1:int, 
                         n2:int, 
                         mean1: np.ndarray, 
                         std1: np.ndarray,
                         mean2: np.ndarray,
                         std2:np.ndarray):
    """Recalculate mean and standard division of combined image dataset across bands
        Args:
    n1: number of images containes in dataset1
    n2: number of images containes in dataset2
    mean1: mean of dataset 1 across bands
    std1: std of dataset 1 across bands
    mean2: mean of dataset 1 across bands
    std2: std of dataset 2 across bands
    """
    if not all([isinstance(i, np.ndarray) for i in [mean1, std1, mean2, std2]]):
        raise TypeError(f'Mean and std should be np.ndarray')
    combined_mean = (n1 * mean1 + n2 * mean2) / (n1 + n2)
    variance1 = std1**2
    variance2 = std2**2
    combined_variance = (
        (n1 - 1) * variance1 +
        (n2 - 1) * variance2 +
        (n1 * n2 / (n1 + n2)) * (mean1 - mean2) ** 2
        ) / (n1 + n2 - 1)
    combined_std = np.sqrt(combined_variance)
    return combined_mean, combined_std

def get_mean_std(data: np.ndarray) -> tuple:
        """Calculate mean and std for input raster data.
        Args:
            data: should be stack of data (images, band, height, width)
        Returns:
            mean: Mean value of input data
            std: Standard deviation of input data
        """
        mean = np.mean(data, axis=(0,2,3))
        std = np.std(data, axis=(0,2,3))
        return mean, std

def merge_coco(coco_fpths: tuple | list, output_path: str = None):
    """Merge multiple coco files into one coco file, update corresponding ids
    Args:
        coco_fpths: A tuple contains multiple coco file paths to merge. 
        output_path: A pathlike string for the path to save merged coco file
    """
    coco_fpths = sorted([Path(coco) for coco in coco_fpths], key=lambda p: p.name)
    merged_coco = {
        "info":[],
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": []
    }
    for idx, file_path in enumerate(coco_fpths):
        with open(file_path, 'r') as f:
            coco = json.load(f)
        validate_coco(coco)
        print(f'{Fore.GREEN}Merging {file_path}')
        coco['info']['dataset_id'] = idx+1
        # Check there are new licenses present in coco, optional, is okay if no licenses in coco file   
        if coco.get('licenses', False):
            license_id_offset = len(merged_coco['licenses'])
            count = 0
            for license in coco['licenses']:
                if license not in merged_coco['licenses'] and not any(d.get('name') == license['name'] for d in merged_coco['licenses']):
                    count+=1
                    merged_coco['licenses'].extend([{"id": count+ license_id_offset, **license}])
        ## This method group the image section (one list of dicts) into lists of dicts depends on license_id
            split_data = defaultdict(list)
            list(map(lambda item: split_data[item['license']].append(item), coco['images']))
            image_group = split_data.values()
            image_append_list = []
         ## Change groups of image' license id
            for group in image_group:
                old_license_id = group[0]['license']
                license_name = next((c['name'] for c in coco['licenses'] if c['id'] == old_license_id), None)
                new_license_id = next((c['id'] for c in merged_coco['licenses'] if c['name'] == license_name),None)
                new_group = [{**image, "license":new_license_id} for image in group]
                image_append_list.append(new_group)
            coco['images'] = [item for sublist in image_append_list for item in sublist]
        else:
            print(f'{Fore.YELLOW}no license section found in {file_path}, will ignore license section')
            
        # Check if there are new categories present in coco
        category_id_offset = len(merged_coco["categories"])
        count = 0
        for category in coco['categories']:
            if category not in merged_coco['categories'] and not any(d.get('name') == category['name'] for d in merged_coco['categories']):
                count+=1
                new_id = category_id_offset+count
                merged_coco['categories'].extend([{"id":new_id, **category}])
        
        ## This method group the annotation section (one list of dicts) into lists of dicts depends on category_id
        split_data = defaultdict(list)
        list(map(lambda item: split_data[item['category_id']].append(item), coco['annotations']))
        anno_group = split_data.values()

        ## Change groups of annotations' category id
        anno_append_list = []
        for group in anno_group:
            old_category_id = group[0]['category_id']
            category_name = next((c['name'] for c in coco['categories'] if c['id'] == old_category_id), None)
            assert category_name is not None, f"Category id {old_category_id} does not exist in categories section"
            new_category_id = next((c['id'] for c in merged_coco['categories'] if c['name'] == category_name),None)
            assert new_category_id is not None, f'Not able to find {category_name} in the categories pool'
            new_group = [{**annotation, "category_id":new_category_id} for annotation in group]
            anno_append_list.append(new_group)
        coco['annotations'] = [item for sublist in anno_append_list for item in sublist]
                
        # Merge cocos image section
        image_id_offset = len(merged_coco["images"])
        new_image_ids = [image["id"]+ image_id_offset for image in coco['images']]
        coco_images = []
        for image_dict, new_id in zip(coco['images'], new_image_ids):
            image_dict['id'] = new_id
            coco_images.append(image_dict)
        merged_coco["images"].extend(coco_images)
        coco['info']['image_ids'] = f'{min(new_image_ids)}-{max(new_image_ids)}'

        ## Change annotations ids
        annotation_id_offset = len(merged_coco["annotations"])
        new_annotations_ids = [anno["id"]+ annotation_id_offset for anno in coco['annotations']]
        coco_annotations=[]
        for anno_dict, new_id in zip(coco['annotations'], new_annotations_ids):
            anno_dict['id'] = new_id
            coco_annotations.append(anno_dict)
        merged_coco['annotations'].extend([
                    {**annotation,"image_id": annotation["image_id"] + image_id_offset} 
                    for annotation in coco_annotations
                ])
        coco['info']['annotation_ids'] = f'{min(new_annotations_ids)}-{max(new_annotations_ids)}'

        # Add info in merged_coco['info'] to be add capability to split datasets
        if len(merged_coco['info']) == 0:
            merged_coco['info'].extend([{"total_mean":coco['info']['pixel_mean'],
                                        "total_std":coco['info']['pixel_std'], 
                                        'number_of_datasets':1}])
            merged_coco['info'].extend([coco['info']])

        else: 
            n1 = len(merged_coco['images'])
            n2 = len(coco['images'])
            mean1 = np.array(merged_coco['info'][0]['total_mean'])
            mean2 = np.array(coco['info']['pixel_mean'])
            assert len(mean1) == len(mean2), f"The dataset {file_path} has {len(mean2)} bands, does not match the rest dataset with {len(mean1)} bands"
            std1 = np.array(merged_coco['info'][0]['total_std'])
            std2 = np.array(coco['info']['pixel_std'])
            total_mean, total_std = recal_pixel_mean_std(n1, n2, mean1, std1, mean2, std2)
            merged_coco['info'].extend([coco['info']])
            merged_coco['info'][0]=({"total_mean":total_mean.tolist(),
                                     "total_std":total_std.tolist(),
                                     'number_of_datasets':len(merged_coco['info'])-1})
    if output_path is None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as temp_file:
            json.dump(merged_coco, temp_file, indent=4)
            temp_file_path = temp_file.name
            print(f'{Fore.YELLOW}Output path is None, will export as tempfile')
        coco = COCO(temp_file_path)
        return coco
    else:
        with open(output_path, 'w') as f:
            json.dump(merged_coco, f, indent=4)
            print(f'{Fore.GREEN}Merged COCO file saved under {output_path}!')
        
def pack_h5_list(data: list):
    """pack list into numpy scalar data which can be storage into hdf5 dataset"""
    pickle_object = pickle.dumps(data)
    return np.void(pickle_object)

def to_pixelcoord(transform, window, polygon: Polygon) -> list :
    """ Convert geographic coords in polygon to local pixel-based coords by bound box.
    Args:
        window: The rasterio window with (col_off, row_off, width, height)
        polygon: Single polygon from shaply
    """
    # apply ~transform, you are computing the inverse of an affine transformation matrix, which 
    # reverse the mapping from spatial coordinates to pixel coordinates.
    
    polylist = []
    if polygon.geom_type == 'MultiPolygon': 
        for p in polygon.geoms:
            polylist.append(p)
    elif polygon.geom_type == 'Polygon':
        polylist.append(polygon)
    polygon = polylist[0]
    if polygon.has_z:
        coord_list = [(x, y) for x, y, _ in polygon.exterior.coords]
    else:
        coord_list = [(x, y) for x, y in polygon.exterior.coords]
    pixel_coord = [~transform*coord for coord in coord_list]
    pixelcoord = [(x - window.col_off,  y - window.row_off) 
                           for x, y in pixel_coord]
    pixelcoord_list = [point for coord in pixelcoord for point in coord]
    return pixelcoord_list

def unpack_h5_list(data: np.void):
    return pickle.loads(data.tobytes())

def window_to_dict(window: Window) -> dict:
    """Convert rasterio Window into dict, in order to save to COCO 
    Args:
        window: The single rasterio Window data
    """
    window_dict  = {
        "col_off": int(window.col_off),
        "row_off": int(window.row_off),
        "width": int(window.width),
        "height": int(window.height)
    }
    return window_dict

def windowdict_to_window(window_dict:dict)-> Window:
    return Window(window_dict['col_off'], window_dict['row_off'],window_dict['width'],window_dict['height'])

def validate_coco(coco: dict):
    """ Validate COCO structure to ensure required fields are present. """
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in coco:
            raise ValueError(f"Missing required key: {key}")
    return True

class COCO_parser_backup:

    """Make COCO Json format (https://github.com/levan92/cocojson/blob/main/docs/coco.md) for images
    """
    def __init__(self, 
                 info: dict = {}):
        self.info = info
        self.COCO = {"info": self.info}


    @staticmethod
    def template(licenses_update:list = [], 
                 images_update:list = [], 
                 categories_update:list = [], 
                 annotations_update:list = []):
        
        if len(licenses_update) == 3:
            licenses = {"id": int(licenses_update[0]), # License ID, must be unique
                        "name":str(licenses_update[1]), # Optional
                        "url": str(licenses_update[2]) # Optional           
                    }
            return licenses
        elif len(licenses_update) != 0: 
            raise ValueError(f'licenses section has 3 catagories, the input has {len(licenses_update)}')

        if len(images_update) == 9:
            images = {"id": int(images_update[0]), # Image ID, Required
                      "file_name": str(images_update[1]), #Required. Better to be the full path to the image
                      "width": int(images_update[2]), # Required
                      "height": int(images_update[3]), # Required
                      "license": int(images_update[4]), # Optional
                      "flickr_url": str(images_update[5]), # Optional                    
                      "coco_url": str(images_update[6]), # Optional
                      "date_captured": str(images_update[7]), # Optional     
                      "window": images_update[8]    # rasterio window, convert by window_to_dict
            }
            return images
        elif len(images_update) != 0: 
            raise ValueError(f'Images section has 9 catagories, the input has {len(images_update)}')
        
        if len(categories_update) == 3:
            categories = {
                "id": int(categories_update[0]), # The categories id, must be unique
                "name": str(categories_update[1]),
                "supercategory": str(categories_update[2]), # Optional
                
            }
                
            return categories
        elif len(categories_update) != 0: 
            raise ValueError(f'Categories section has 3 catagories, the input has {len(categories_update)}')
        
        if len(annotations_update) == 10:
            annotations = {
                "id": int(annotations_update[0]), # Each annotation also has an id (unique to all other annotations in the dataset)
                "image_id": int(annotations_update[1]), #corresponds to a specific image in the dataset
                "category_id": int(annotations_update[2]), # corresponds to a single category specified in the categories section
                "bbox": list(annotations_update[3]), #bounding box, required, default use XYXY absolute coordinate
                "area": int(annotations_update[4]), # measured in pixels (10px * 20px = 200px)
                "iscrowd": int(annotations_update[5]), # specifies whether the segmentation is for a single object (0) or  not possible to delineate the individually from other similar objects. (1)
                "segmentation": list(annotations_update[6]), # A list of list of points [[x1,y1, x2,y2 ...], [x11, y11, x22, y22, ...]] define the shape of the object, optional                
                "keypoints" : list(annotations_update[7]), #Optional, list of key points with format [x1,y1,v1, ...,xn,yn,vn] visibility (0(no label),1(label, visable),2(label, not visable))
                "num_keypoints": int(annotations_update[8]), # number of key points
                "bbox_mode": annotations_update[9] # Bounding box mod, use (x,y) of top left and (x,y) of bottom right, required
            } # Each instance should have it's own bbox and segmentation even is in same image and same category
            return annotations
        elif len(annotations_update) != 0: 
            raise ValueError(f'Annotations section has 10 catagories , the input has {len(annotations_update)}')

    
    @staticmethod   
    def _add_licenses(license_id: list,
                    license_url: list =[],
                    license_name: list = []):
        
        if len(license_url) != len(license_id) or len(license_name) != len(license_id):
            raise AssertionError(f"All input lists should have same length, use 'N/A' or '' in the list for missing value ")
        # Check if any empty lists, if so, fill with default value
        named_vars = {k: v for k, v in locals().items() if k not in ['args', 'kwargs']}
        empty_named_vars = {k: v for k, v in named_vars.items() if v == []}
        if len(empty_named_vars) > 0:
            for key, _ in empty_named_vars.items():
                new_list = [''] * len(license_id)
                empty_named_vars[key] = new_list
        named_vars.update(empty_named_vars)
        license_list = []
        for  id, name, url in zip(*list(named_vars.values())):
            tempdict = COCO_parser.template(licenses_update=[id, name, url])
            license_list.append(tempdict)
        return {'licenses':license_list}

    @staticmethod
    def _add_images(id: list,
                   file_name: list,
                   width: list ,
                   height: list ,
                   license: list= [],
                   flickr_url: list = [], 
                   coco_url: list = [],
                   date_captured: list= [], 
                   window: list = []                  
                   ):
        if len(id) != len(file_name) or len(id) != len(width) or len(id) != len(height):
            raise AssertionError (f'Length of id, fil_name, width, and height should be the same')
        named_vars = {k: v for k, v in locals().items() if k not in ['args', 'kwargs']}
        empty_named_vars = {k: v for k, v in named_vars.items() if v == []}
        if len(empty_named_vars) > 0:
            for key, _ in empty_named_vars.items():
                if key == 'license':
                    new_list = [1] * len(id)
                elif key == 'date_captured':
                    new_list = ["1949-10-01T07:43:32Z"] * len(id)  
                elif key == 'window':
                    new_list = [{}]*len(id)
                else:
                    new_list = [''] * len(id)
                empty_named_vars[key] = new_list
        named_vars.update(empty_named_vars)
        image_list = []
        for i, fn, w, h, lic, fl, cocol, dc, w in zip(*list(named_vars.values())):
            tempdict = COCO_parser.template(images_update=[i, fn, w, h, lic, fl, cocol, dc, w])
            image_list.append(tempdict)
        return {"images":image_list}
    
    @staticmethod
    def _add_categories(id: list,
                       name: list = [],
                       supercategory:list = [], 
                       ):
        named_vars = {k: v for k, v in locals().items() if k not in ['args', 'kwargs']}
        empty_named_vars = {k: v for k, v in named_vars.items() if v == []}
        if len(empty_named_vars) > 0:
            for key, _ in empty_named_vars.items():
                new_list = [''] * len(id)
                empty_named_vars[key] = new_list
            named_vars.update(empty_named_vars)
        categories_list = []
        for i, n, s in zip(*list(named_vars.values())):
            tempdict = COCO_parser.template(categories_update=[i ,n, s])
            categories_list.append(tempdict)
        return {"categories": categories_list}
    
    @staticmethod
    def _add_annotations(id:list,
                        image_id:list,
                        category_id:list, 
                        bbox:list,                              
                        area:list,
                        iscrowd:list,
                        segmentation:list = [],
                        keypoints :list = [],
                        num_keypoints :list = [],
                        bbox_mode = 'xywh'):
        if bbox_mode == 'xyxy':
            bbox_mode = ['xyxy']*len(id)
        elif bbox_mode == 'xywh':
            bbox_mode = ['xywh']*len(id)
        elif isinstance(bbox_mode, list):
            bbox_mode = bbox_mode
        else:
            raise ValueError('Only "xyxy" ,"xywh" or customized list are currently supported') 
        named_vars = {k: v for k, v in locals().items() if k not in ['args', 'kwargs']}
        empty_named_vars = {k: v for k, v in named_vars.items() if v == []}
        if len(empty_named_vars) > 0:
            for key, _ in empty_named_vars.items():
                if key == "keypoints" or key == "segmentation":
                    new_list = [[]] * len(id)
                elif key == "num_keypoints":
                    new_list = [0] * len(id)
                empty_named_vars[key] = new_list
            named_vars.update(empty_named_vars)
        annotations_list = []
        for id, ii, ci, bb,  a, ic, se, kp, nk, bbm in zip(*list(named_vars.values())):
            tempdict = COCO_parser.template(annotations_update=[id, ii, ci, bb, a, ic, [se], kp, nk, bbm])
            annotations_list.append(tempdict)
        return {"annotations": annotations_list}
        
    def add_licenses(self,
                    license_id: list,
                    license_url: list =[],
                    license_name: list = []):
        licenses = self._add_licenses(license_id, license_url, license_name)
        if "licenses" in self.COCO:
            self.COCO['licenses'] = self.COCO['licenses'] + licenses['licenses']
        else:
            self.COCO.update(licenses)
    
    def add_images(self, id: list,
                   file_name: list,
                   width: list ,
                   height: list ,
                   license: list= [],
                   flickr_url: list = [], 
                   coco_url: list = [],
                   date_captured: list= [],
                   window: list = []                   
                   ):
        images = self._add_images(id, file_name,  width, height,license, 
                                                  coco_url,  
                                                  date_captured, flickr_url, window)
        if "images" in self.COCO: 
            self.COCO['images'] = self.COCO['images'] + images['images']
        else: 
            self.COCO.update(images)

    def add_categories(self, id: list,
                       name: list = [],
                       supercategory:list = []):
        categories = self._add_categories(id, name, supercategory)
        if "categories" in self.COCO:
            self.COCO["categories"] = self.COCO['categories'] + categories['categories']
        else: 
            self.COCO.update(categories)
    
    def add_annotations(self, id:list,
                        image_id:list,
                        category_id:list, 
                        bbox:list,                              
                        area:list,
                        iscrowd:list,
                        segmentation:list = [],
                        keypoints :list = [],
                        num_keypoints :list = [],
                        bbox_mode = 'xyxy'):
        annotations = self._add_annotations(id, image_id, category_id, bbox, 
                                                area, iscrowd, segmentation,
                                                keypoints,
                                                num_keypoints,
                                                bbox_mode)
        if 'annotations' in self.COCO: 
            self.COCO['annotations'] = self.COCO['annotations'] + annotations['annotations']
        else:
            self.COCO.update(annotations)
    @property
    def data(self):
        return SimpleNamespace(**self.COCO)
    
    def save_json(self, save_path:str):
        with open(Path(save_path).absolute(), 'w') as f:
            json.dump(self.COCO, f)

class COCO_parser:
    """COCO JSON format parser for images.

    Attributes:
        info (dict): Metadata about the dataset.
        COCO (dict): Structure to hold the COCO formatted data.
    """
    def __init__(self, info: dict = None):
        self.info = info or {}
        self.COCO = {"info": self.info}

    @staticmethod
    def template(licenses_update: Optional[List[Any]] = None,
                 images_update: Optional[List[Any]] = None,
                 categories_update: Optional[List[Any]] = None,
                 annotations_update: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Creates a COCO template for licenses, images, categories, and annotations."""
        licenses_update = licenses_update or []
        images_update = images_update or []
        categories_update = categories_update or []
        annotations_update = annotations_update or []

        if licenses_update and len(licenses_update) != 3:
            raise ValueError(f'licenses section must have 3 categories, received {len(licenses_update)}.')

        if images_update and len(images_update) != 9:
            raise ValueError(f'Images section must have 9 categories, received {len(images_update)}.')

        if categories_update and len(categories_update) != 3:
            raise ValueError(f'Categories section must have 3 categories, received {len(categories_update)}.')

        if annotations_update and len(annotations_update) != 10:
            raise ValueError(f'Annotations section must have 10 categories, received {len(annotations_update)}.')

        licenses = {"id": int(licenses_update[0]), "name": str(licenses_update[1]), "url": str(licenses_update[2])} if licenses_update else {}
        images = {
            "id": int(images_update[0]),
            "file_name": str(images_update[1]),
            "width": int(images_update[2]),
            "height": int(images_update[3]),
            "license": int(images_update[4]),
            "flickr_url": str(images_update[5]),
            "coco_url": str(images_update[6]),
            "date_captured": str(images_update[7]),
            "window": images_update[8]
        } if images_update else {}
        categories = {
            "id": int(categories_update[0]),
            "name": str(categories_update[1]),
            "supercategory": str(categories_update[2]),
        } if categories_update else {}
        annotations = {
            "id": int(annotations_update[0]),
            "image_id": int(annotations_update[1]),
            "category_id": int(annotations_update[2]),
            "bbox": list(annotations_update[3]),
            "area": int(annotations_update[4]),
            "iscrowd": int(annotations_update[5]),
            "segmentation": list(annotations_update[6]),
            "keypoints": list(annotations_update[7]),
            "num_keypoints": int(annotations_update[8]),
            "bbox_mode": annotations_update[9],
        } if annotations_update else {}

        return {"licenses": licenses, "images": images, "categories": categories, "annotations": annotations}

    @staticmethod
    def _add_section(section_name: str, *args: List[List[Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generates a section of the COCO JSON."""
        section_list = []
        for values in zip(*args):
            section_data = COCO_parser.template(**{section_name: list(values)})
            section_list.append(section_data)
        return section_list

    def add_licenses(self, license_id: List[int], license_url: List[str] = None, license_name: List[str] = None):
        """Add licenses to the COCO structure."""
        license_url = license_url or [''] * len(license_id)
        license_name = license_name or [''] * len(license_id)
        licenses = self._add_section('licenses_update', license_id, license_url, license_name)

        self.COCO.setdefault('licenses', []).extend([d[key] for d in licenses for key in d if key == 'licenses'])

    def add_images(self, id: List[int], file_name: List[str], width: List[int], height: List[int],
                   license: List[int] = None, flickr_url: List[str] = None, coco_url: List[str] = None,
                   date_captured: List[str] = None, window: List[dict] = None):
        """Add images to the COCO structure."""
        license = license or [1] * len(id)
        flickr_url = flickr_url or [''] * len(id)
        coco_url = coco_url or [''] * len(id)
        date_captured = date_captured or ["1949-10-01T07:43:32Z"] * len(id)
        window = window or [{}] * len(id)

        images = self._add_section('images_update', id, file_name, width, height, license, flickr_url, coco_url, date_captured, window)

        self.COCO.setdefault('images', []).extend([d[key] for d in images for key in d if key == 'images'])

    def add_categories(self, id: List[int], name: List[str] = None, supercategory: List[str] = None):
        """Add categories to the COCO structure."""
        name = name or [''] * len(id)
        supercategory = supercategory or [''] * len(id)

        categories = self._add_section('categories_update', id, name, supercategory)

        self.COCO.setdefault('categories', []).extend([d[key] for d in categories for key in d if key == 'categories'])

    def add_annotations(self, id: List[int], image_id: List[int], category_id: List[int],
                        bbox: List[List[float]], area: List[int], iscrowd: List[int],
                        segmentation: List[List[float]] = None, keypoints: List[List[float]] = None,
                        num_keypoints: List[int] = None, bbox_mode: str = 'xywh'):
        """Add annotations to the COCO structure."""
        segmentation = segmentation or [[]] * len(id)
        keypoints = keypoints or [[]] * len(id)
        num_keypoints = num_keypoints or [0] * len(id)

        annotations = self._add_section('annotations_update', id, image_id, category_id, bbox, area, iscrowd, segmentation, keypoints, num_keypoints, [bbox_mode] * len(id))

        self.COCO.setdefault('annotations', []).extend([d[key] for d in annotations for key in d if key == 'annotations'])
        for annotation in self.COCO['annotations']:
            annotation['segmentation'] = [annotation['segmentation']]

    @property
    def data(self) -> SimpleNamespace:
        """Return the COCO data as a SimpleNamespace."""
        return SimpleNamespace(**self.COCO)

    def save_json(self, save_path: str):
        """Save the COCO format data to a JSON file.

        Args:
            save_path (str): The path to save the JSON file.
        """
        with open(save_path, 'w') as f:
            json.dump(self.COCO, f, indent=4)

class SimpleNamespace_ext(SimpleNamespace):
    """ 
    An extension of SimpleNamespace, make sure all levels of keys in dict are converted into namespace not
    only the first level by using SimpleNamespace_ext(**your_dict)
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                kwargs[key] = self._nested_dict(value)
        super().__init__(**kwargs)
        Warning('This funciton is Deprecated')

    def _nested_dict(self, d:dict) -> SimpleNamespace:
        if not isinstance(d, dict):
            return d
        ns = SimpleNamespace()
        for key, value in d.items():
            # If the value is a dictionary, convert it recursively
            if isinstance(value, dict):
                setattr(ns, key, self._nested_dict(value))
            else:
                setattr(ns, key, value)  # Otherwise, just set the attribute
        return ns
    
    def append(self, other:SimpleNamespace):
        if isinstance(other, SimpleNamespace):
            self.__dict__.update(other.__dict__)
        else: 
            raise TypeError('Only able to append SimpleNamespace')

class Suppressor:
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')  # Redirect stdout to null

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout  # Restore stdout
