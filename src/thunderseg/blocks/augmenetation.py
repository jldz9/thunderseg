import cv2 
import numpy as np
import random
import albumentations as A


def get_valid_crop(image, mask, crop_size=(256, 256)):
    """Similiar to  Albumentations' CropNonEmptyMaskIfExists method but 
    ensuring that every crop contains at least one mask and does not cut objects
    Args:
        image (np.array): Image to crop, image shape (H, W, C)
        mask (np.array): Binary Mask (H, W)
        crop_size (tuple): Size of the crop
        """
    H, W = mask.shape
    crop_h, crop_w = crop_size

    # Find connected components (objects) in the mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Remove background label (0)
    obj_indices = np.where(stats[:, 4] > 0)[0][1:]  # Ignore background

    if len(obj_indices) == 0:
        
        raise ValueError("No objects found in the mask!")

   
    # Randomly choose an object
    obj_idx = random.choice(obj_indices)
    x, y, w, h, area = stats[obj_idx]

    # Ensure the crop fully contains the object
    x_start = max(0, x + w - crop_w) if x + crop_w > W else x
    y_start = max(0, y + h - crop_h) if y + crop_h > H else y

    # Extract the crop
    cropped_image = image[y_start:y_start+crop_h, x_start:x_start+crop_w]
    cropped_mask = mask[y_start:y_start+crop_h, x_start:x_start+crop_w]

    return cropped_image, cropped_mask

 