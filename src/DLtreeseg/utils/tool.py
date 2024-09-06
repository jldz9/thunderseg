#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tools use by DLtreeseg 
"""
import json
import pickle
from pathlib import Path
from dateutil import parser

from detectron2.structures import BoxMode
import numpy as np

# pack list into numpy scalar data which can be storage into hdf5 dataset
def pack_h5_list(data: list):
    pickle_object = pickle.dumps(data)
    return np.void(pickle_object)
# unpack numpy scalar data to list
def unpack_h5_list(data: np.void):
    return pickle.loads(data.tobytes())

class COCO_format:
    """
    Make COCO Json format (https://cocodataset.org/#home) for 
    """
    def __init__(self, 
                 metadata: dict = {}):
        
        self.info = metadata
        #self.COCO = {"info": self.info}


    @staticmethod
    def template(licenses_update:list = [], 
                 images_update:list = [], 
                 categories_update:list = [], 
                 annotations_update:list = []):
        
        if len(licenses_update) == 3:
            licenses = {"url": str(licenses_update[0]), # Optional
                    "id": int(licenses_update[1]), # License ID, must be unique
                    "name":str(licenses_update[2]) # Optional
                    }
            return licenses
        elif len(licenses_update) != 0: 
            raise ValueError(f'licenses section has 3 catagories, the input has {len(licenses_update)}')

        if len(images_update) == 8:
            images = {"license": int(images_update[0]), # Optional
                    "file_name": str(licenses_update[1]), #Required. Better to be the full path to the image
                    "coco_url": str(licenses_update[2]), # Optional
                    "height": int(images_update[3]), # Required
                    "width": int(images_update[4]), # Required
                    "date_captured": parser.parse(images_update[5]), # Optional
                    "flickr_url": str(images_update[6]), # Optional
                    "id": int(images_update[7]) # Image ID, Required
            }
            return images
        elif len(images_update) != 0: 
            raise ValueError(f'Images section has 7 catagories, the input has {len(images_update)}')
        
        if len(categories_update) == 3:
            categories = {
                "supercategory": str(categories_update[0]), # Optional
                "id": int(categories_update[1]), # The categories id, must be unique
                "name": str(categories_update[2])
            }
            return categories
        elif len(categories_update) != 0: 
            raise ValueError(f'Categories section has 3 catagories, the input has {len(categories_update)}')
        
        if len(annotations_update) == 10:
            annotations = {
                "bbox": list(annotations_update[0]), #bounding box, required, default use XYXY absolute coordinate
                "segmentation": list(annotations_update[1]), # A list of list of points [[x1,y1, x2,y2 ...]] define the shape of the object, optional
                "area": int(annotations_update[2]), # measured in pixels (10px * 20px = 200px)
                "iscrowd": int(annotations_update[3]), # specifies whether the segmentation is for a single object (0) or  not possible to delineate the individually from other similar objects. (1)
                "image_id": int(annotations_update[4]), #corresponds to a specific image in the dataset
                "category_id": int(annotations_update[5]), # corresponds to a single category specified in the categories section
                "id": int(annotations_update[6]), # Each annotation also has an id (unique to all other annotations in the dataset)
                "keypoints" : list(annotations_update[7]), #Optional, list of key points with format [x1,y1,v1, ...,xn,yn,vn] visibility (0(no label),1(label, visable),2(label, not visable))
                "num_keypoints": int(annotations_update[8]), # number of key points
                "bbox_mode": annotations_update[9], # Bounding box mod, use (x,y) of top left and (x,y) of bottom right, required
            } # Each instance should have it's own bbox and segmentation even is in same image and same category
            return annotations
        elif len(annotations_update) != 0: 
            raise ValueError(f'Annotations section has 10 catagories , the input has {len(annotations_update)}')

    
    @staticmethod   
    def add_license(license_id: list,
                    license_url: list =[],
                    license_name: list = []):
        
        image_count = len(license_id)
        if len(license_url) != image_count or len(license_name) != image_count:
            raise AssertionError(f"All input lists should have same length, use 'N/A' or '' in the list for missing value ")
        
        # Check if any empty lists, if so, fill with default value
        named_vars = {k: v for k, v in locals().items() if k not in ['args', 'kwargs']}
        empty_named_vars = {k: v for k, v in named_vars.items() if v == []}
        if len(empty_named_vars) > 0:
            for key, _ in empty_named_vars.items():
                new_list = [''] * image_count
                exec(f'{key} = {new_list}')
        license_list = [tempdict]
        for url, id, name in zip(license_url, license_id, license_name):
            tempdict = COCO_format.template(licenses_update=[url,id,name])
            license_list.append(tempdict)
        return {'licenses':license_list}

    @staticmethod
    def add_images(id: list,
                   file_name: list ,
                   license_id: list= [], 
                   coco_url: list = [],
                   height: list = [],
                   width: list = [],
                   date_captured: list= [],
                   flickr_url: list = []
                   ):
        if len(id) != len(file_name):
            raise AssertionError(f"id and filename should have same length")
        named_vars = {k: v for k, v in locals().items() if k not in ['args', 'kwargs']}
        empty_named_vars = {k: v for k, v in named_vars.items() if v == []}
        if len(empty_named_vars) > 0:
            for key, _ in empty_named_vars.items():
                if key == 'height' or key == 'width' or key == 'license_id':
                    new_list = [0] * len(id)
                elif key == 'data_captured':
                    new_list = ["19491001-15:43:32"]
                else:
                    new_list = [''] * len(id)
                exec(f'{key} = {new_list}')
        image_list = []
        for i, fn, lid, cocol, h, w, dc, fl in zip(id, file_name, license_id, 
                                                  coco_url, height, width, 
                                                  date_captured, flickr_url):
            tempdict = COCO_format.template(images_update=[lid, fn, cocol, h, w, dc, fl, i])
            image_list.append(tempdict)
        return {"images":image_list}
    
    @staticmethod
    def add_categories(id: list,
                       supercategory:list = [], 
                       name: list = []):
        named_vars = {k: v for k, v in locals().items() if k not in ['args', 'kwargs']}
        empty_named_vars = {k: v for k, v in named_vars.items() if v == []}
        if len(empty_named_vars) > 0:
            for key, _ in empty_named_vars.items():
                new_list = [''] * len(id)
                exec(f'{key} = {new_list}')
        categories_list = []
        for i, s, n in zip(id, supercategory, name):
            tempdict = COCO_format.template(categories_update=[s, i ,n])
            categories_list.append(tempdict)
        return {"categories": categories_list}
    """
    @staticmethod
    def add_annotations(bbox:list,
                        image_id:list, 
                        category_id:list,
                        id:list,
                        segmentation,
                        area,
                        iscrowd,
                        keypoints,
                        num_keypoints,
                        bbox_mode = 'xyxy'):
        named_vars = {k: v for k, v in locals().items() if k not in ['args', 'kwargs']}
        empty_named_vars = {k: v for k, v in named_vars.items() if v == []}
        if len(empty_named_vars) > 0:
            for key, _ in empty_named_vars.items():
                new_list = [''] * len(id)
                exec(f'{key} = {new_list}')
        if bboxmode == 'xyxy':
            bm = BoxMode.XYXY_ABS
        elif bboxmode == 'xywh':
            bm = BoxMode.XYWH_ABS
            """