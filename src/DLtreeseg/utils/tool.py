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

        if len(images_update) == 8:
            images = {"id": int(images_update[0]), # Image ID, Required
                      "file_name": str(images_update[1]), #Required. Better to be the full path to the image
                      "width": int(images_update[2]), # Required
                      "height": int(images_update[3]), # Required
                      "license": int(images_update[4]), # Optional
                      "flickr_url": str(images_update[5]), # Optional                    
                      "coco_url": str(images_update[6]), # Optional
                      "date_captured": str(images_update[7]), # Optional         
            }
            return images
        elif len(images_update) != 0: 
            raise ValueError(f'Images section has 7 catagories, the input has {len(images_update)}')
        
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
                "segmentation": list(annotations_update[6]), # A list of list of points [[x1,y1, x2,y2 ...]] define the shape of the object, optional                
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
            tempdict = COCO_format.template(licenses_update=[id, name, url])
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
                   ):
        if len(id) != len(file_name) or len(id) != len(width) or len(id) != len(height):
            raise AssertionError (f'Length of id, fil_name, width, and height should be the same')
        named_vars = {k: v for k, v in locals().items() if k not in ['args', 'kwargs']}
        empty_named_vars = {k: v for k, v in named_vars.items() if v == []}
        if len(empty_named_vars) > 0:
            for key, _ in empty_named_vars.items():
                if key == 'license':
                    new_list = [0] * len(id)
                elif key == 'date_captured':
                    new_list = ["1949-10-01T07:43:32Z"] * len(id)  
                else:
                    new_list = [''] * len(id)
                empty_named_vars[key] = new_list
        named_vars.update(empty_named_vars)
        image_list = []
        for i, fn, w, h, lic, fl, cocol, dc in zip(*list(named_vars.values())):
            tempdict = COCO_format.template(images_update=[i, fn, w, h, lic, fl, cocol, dc])
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
            tempdict = COCO_format.template(categories_update=[i ,n, s])
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
                        bbox_mode = 'xyxy'):
        if bbox_mode == 'xyxy':
            bbox_mode = [BoxMode.XYXY_ABS]*len(id)
        elif bbox_mode == 'xywh':
            bbox_mode = [BoxMode.XYWH_ABS]*len(id)
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
            tempdict = COCO_format.template(annotations_update=[id, ii, ci, bb, a, ic, se, kp, nk, bbm])
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
                   ):
        images = self._add_images(id, file_name,  width, height,license, 
                                                  coco_url,  
                                                  date_captured, flickr_url)
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
    
    def data(self):
        return self.COCO
    
    def save_json(self, save_path:str):
        with open(save_path, 'w') as f:
            json.dump(self.COCO, f)