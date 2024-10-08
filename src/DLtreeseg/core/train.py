import copy

import numpy as np
import rasterio as rio
import torch
import torch.nn as nn
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.modeling.backbone import build_resnet_backbone, BACKBONE_REGISTRY, ResNet
from detectron2.modeling import  ShapeSpec
from detectron2.modeling.backbone.resnet import BasicStem, BottleneckBlock
from detectron2.config import get_cfg


class MultiBand_DatasetMapper(DatasetMapper):
    """
    A DatasetMapper extended from detectron2 DefaultMapper to support multi-band images.
    """
    def __init__(self, cfg, is_train=True):
        augmentations = [
            T.RandomRotation(angle=[90, 90], expand=False),
            T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
        ]
        super().__init__(
                         augmentations=augmentations,
                         image_format=cfg.INPUT.FORMAT,
                         instance_mask_format=cfg.INPUT.MASK_FORMAT,
                         is_train=is_train,
                         keypoint_hflip_indices= None,
                         precomputed_proposal_topk= None,
                         recompute_boxes=False,
                         use_instance_mask=cfg.MODEL.MASK_ON,
                         use_keypoint=cfg.MODEL.KEYPOINT_ON,
                         )
        self.cfg = cfg
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        
    
    def __call__(self, dataset_dict):
        """Use for instance segmentation, does not consider semantic segmentation for now"""
        dataset_dict = copy.deepcopy(dataset_dict)
        with rio.open(dataset_dict['file_name']) as src:
            image = src.read() # output (Count, Height, Width) ndarray
        # Apply augmentations
        image = np.transpose(image, (1, 2, 0))
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image_transformed = aug_input.image
        # convert to torch tensor
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_transformed.transpose(2,0,1)))
        # transform annotations
        image_transformed_shape = image_transformed.shape[:2]
        if 'annotations' in dataset_dict:
            dataset_dict = self._transform_annotations(dataset_dict, transforms, image_transformed_shape)
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        return dataset_dict

class MultiBand_ResNetStem(BasicStem):
    def __init__(self, cfg, in_channels):
        super().__init__(
            in_channels=in_channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            stride = 4,
            norm=cfg.MODEL.RESNETS.STEM_NORM,
        )
"""
@BACKBONE_REGISTRY.register()
def multiband_resnet_backbone(cfg, input_shape):
    in_channels = cfg.INPUT.IN_CHANNELS
    #input_shape = ShapeSpec(5,1578,1578)
    backbone = ResNet(cfg, lambda: MultiBand_ResNetStem(cfg, in_channels), BottleneckBlock)
    return backbone
"""
class CustomResNetStem(BasicStem):
    def __init__(self, in_channels=5, out_channels=64, norm="BN"):
        super().__init__(in_channels, out_channels, norm)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)

@BACKBONE_REGISTRY.register()
def build_custom_resnet_backbone(cfg, input_shape):
    return ResNet(cfg, input_shape, CustomResNetStem, BottleneckBlock)

# In your training script, use the new backbone in the config




class MultiBand_Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MultiBand_DatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = MultiBand_DatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

        return optimizer

    
def modify_conv1_weights(model, num_input_channels):
    """
    Modify the weights of the first convolutional layer (conv1) to accommodate a different number of input channels.

    This function adjusts the weights of the `conv1` layer in the model's backbone to support a custom number
    of input channels. It creates a new weight tensor with the desired number of input channels,
    and initializes it by repeating the weights of the original channels.

    Args:
        model (torch.nn.Module): The model containing the convolutional layer to modify.
        num_input_channels (int): The number of input channels for the new conv1 layer.

    """
    with torch.no_grad():
        # Retrieve the original weights of the conv1 layer
        old_weights = model.backbone.bottom_up.stem.conv1.weight

        # Create a new weight tensor with the desired number of input channels
        # The shape is (out_channels, in_channels, height, width)
        new_weights = torch.zeros((old_weights.size(0), num_input_channels, *old_weights.shape[2:]))

        # Initialize the new weights by repeating the original weights across the new channels
        # This example repeats the first 3 channels if num_input_channels > 3
        for i in range(num_input_channels):
            new_weights[:, i, :, :] = old_weights[:, i % 3, :, :]

        # Create a new conv1 layer with the updated number of input channels
        model.backbone.bottom_up.stem.conv1 = nn.Conv2d(
            num_input_channels, old_weights.size(0), kernel_size=7, stride=2, padding=3, bias=False
        )

        # Copy the modified weights into the new conv1 layer
        model.backbone.bottom_up.stem.conv1.weight.copy_(new_weights)
