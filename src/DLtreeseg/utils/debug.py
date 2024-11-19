import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
from matplotlib.patches import Rectangle


def check_image_target(image, target, savepath=None):
    """ Plot a single image with target to check the quality of transform
    Args: 
        image: (torch.Tensor, torchvision.tv_tensors._image.Image, or np.array) the single image with an array-like format
        target: (dict) The annotation info relate to this image, keys should at least contains image_id, boxes, labels, and masks
    """
    if not isinstance(image, np.ndarray):
        image = image.cpu().permute(1,2,0).numpy()
        image = (image - np.min(image))/(np.max(image) - np.min(image))
    target = {k: v.cpu().numpy() for k, v in target.items()}
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    ax = plt.gca()

    for box, label, mask in zip(target['boxes'], target['labels'], target['masks']):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min, f'Label: {label}', color='white', fontsize=12, backgroundcolor='red')
        masked_image = np.ma.masked_where(mask == 0, mask)
        masked_image = np.transpose(masked_image, (1,2,0))
        plt.imshow(masked_image, cmap='jet', alpha=0.5)
    plt.axis('off')

    if savepath is not None: 
        plt.savefig(savepath, bbox_inches='tight',pad_inches=0)
        




