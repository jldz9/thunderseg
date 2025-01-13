#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug module for thunderseg package
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.patches import Rectangle

def check_image_target(image, target=None, savepath=None):
    """ Plot a single image with target to check the quality of transform
    Args: 
        image: (torch.Tensor, torchvision.tv_tensors._image.Image, or np.array) the single image with an array-like format
        target: (dict) The annotation info relate to this image, keys should at least contains image_id, boxes, labels, and masks
    """
    if not isinstance(image, np.ndarray):
        image = image.cpu().permute(1,2,0).numpy()
        image = (image - np.min(image))/(np.max(image) - np.min(image))
    
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    ax = plt.gca()
    if target is not None:
        target = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
        for box, label, mask, area in zip(target['boxes'], target['labels'], target['masks'], target['area']):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = Rectangle((x_min, y_min), width, height, linewidth=0.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            #fontsize = int(np.sqrt(area))//3
            #plt.text(x_min, y_min, f'Label: {label}', color='white', fontsize=fontsize, backgroundcolor='yellow')
            masked_image = np.ma.masked_where(mask == 0, mask)
            #masked_image = np.transpose(masked_image, (1,2,0))
            plt.imshow(masked_image, cmap='jet', alpha=0.5)
    if savepath is not None: 
        plt.savefig(savepath, bbox_inches='tight',pad_inches=0)
        
def find_cuda_tensors(data, parent_key=""):
    """
    Recursively search for CUDA tensors in a nested data structure and print their keys/paths.
    """
    if torch.is_tensor(data):
        if data.is_cuda:
            print(f"CUDA tensor found at key: {parent_key}, shape: {data.shape}")
    elif isinstance(data, dict):
        for key, value in data.items():
            find_cuda_tensors(value, f"{parent_key}.{key}" if parent_key else key)
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            find_cuda_tensors(value, f"{parent_key}[{idx}]")


