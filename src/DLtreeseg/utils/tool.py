#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tools use by DLtreeseg 
"""
import json
import pickle
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

import numpy as np
from rasterio.windows import Window
from shapely import Polygon, unary_union

# pack list into numpy scalar data which can be storage into hdf5 dataset
def pack_h5_list(data: list):
    pickle_object = pickle.dumps(data)
    return np.void(pickle_object)
# unpack numpy scalar data to list
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

def read_toml(config_path) -> dict:
    """Read toml config file used by DLtreeseg"""
    with open(config_path, 'rb') as f:
        tmp = tomllib.load(f)
    toml = SimpleNamespace_ext(**tmp)
    return toml

def get_mean_std(data: np.ndarray):
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
def read_json(json_path: str):
        with open(json_path, 'r') as f:
            COCO = json.load(f)
        return COCO
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
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                kwargs[key] = self._nested_dict(value)
        super().__init__(**kwargs)

    def _nested_dict(self, d:dict):
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
    