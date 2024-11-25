import copy
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import warnings
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L
import numpy as np
import rasterio as rio
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
torch.set_float32_matmul_precision('high')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from thunderseg.utils import bbox_from_mask, Config

cfg = Config((Path(__file__).parents[1] / 'utils/config.toml').resolve())

def get_transform(image:np.ndarray, target:dict={}, train = True, mean:list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
    """
    Apply transform to both image and target using Albumentations, 
    Args:
        image: should be a numpy array with shape of (Height,Width,Channel)
        target: should be a dict contains bbox, mask, 
    """

    three_channel_image_only_transform = A.Compose(
        [A.SomeOf([ 
        #A.PlanckianJitter(),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)),
        A.RandomToneCurve(),
        ], n=1, p=0.5)
        ])
    
    image_only_transform = A.Compose([A.SomeOf([
        A.Downscale(scale_range=(0.5, 1)),
        #A.GaussNoise(noise_scale_factor=0.5),
        #A.Sharpen(),
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
    A.RandomCrop(height=cfg.PREPROCESS.TRANSFORM.RANDOM_CROP_HEIGHT, width=cfg.PREPROCESS.TRANSFORM.RANDOM_CROP_WIDTH),
    ToTensorV2()])
    if train:
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
            #check_image_target(image, target, f'/workspaces/DLtreeseg/test/image_debug/debug{target["image_id"]}.png')
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
        
    elif not train:
        predict_transform = A.Compose([A.Normalize(mean=mean, std=std, max_pixel_value=1),
                                       ToTensorV2()])
        temp = predict_transform(image=image)
        image = temp['image']
        return image
    
class TrainDataset(Dataset):
    def __init__(self, coco:str | COCO, transform=get_transform):
        """
        Args:
            coco : The merged json file path exported from merged_coco represent the image dataset
            transform : transform method use for image agumentation
        """
        if isinstance(coco, COCO):
            self._coco = coco
        else:
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
            # This make sure filter out empty annotations
            image_info = self._coco.imgs[idx+1] # pycocotools use id number which starts from 1.
            annotation_ids = self._coco.getAnnIds(imgIds=idx+1)
            if len(annotation_ids) > 0:
                image, target = self._load_image_target(image_info, annotation_ids)
                if self._transform is not None:
                    image, target= self._transform(image, target, train = True, 
                                                   mean=self._coco.dataset['summary']['total_mean'],
                                                   std=self._coco.dataset['summary']['total_std'])
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
    def __init__(self, train_coco:str, predict_coco:str, transform=get_transform):
        """ 
        Args:
            coco_train: The merged train coco json file, we need to use the mean and std from the train dataset
            coco_predict : The merged predict coco json file path exported from merge_coco represent the predict image dataset
            transform : transform use to transfrom the dataset
        """
        if isinstance(train_coco, COCO):
            self._train_coco = train_coco
        else:
            self._train_coco = COCO(train_coco)
        if isinstance(predict_coco, COCO):
            self._coco = predict_coco
        else:
            self._coco = COCO(predict_coco)
        self._img_dir = Path(self._coco.imgs[self._coco.getImgIds()[0]]['file_name']).parent.as_posix()
        self._transform = transform
    
    def __len__(self):
        return len(self._coco.imgs)

    def __getitem__(self, idx):
        with rio.open(self._coco.imgs[idx+1]['file_name']) as f:
            image = f.read()
            image = np.transpose(image, (1, 2, 0))
        if self._transform:
            image = self._transform(image, train=False, 
                                    mean=self._train_coco.dataset['summary']['total_mean'],
                                    std=self._train_coco.dataset['summary']['total_std'])
  
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
            self.predict_dataset = PreditDataset(train_coco=self.train_coco, predict_coco=self.predict_coco, transform=self.transform)
    
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
                          num_workers=self.num_workers)
    
class MaskRCNN_RGB(L.LightningModule):
    def __init__(self, num_classes: int = 2, learning_rate: float = 1e-3):
        super().__init__()
        # Load maskrcnn model from torchvision
        self.model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

        # Replace the pre-trained head with a new one
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # Replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
    def forward(self, images, targets=None):
        if self.training:
            train_targets = [{key: target[key] for key in ['boxes', 'labels', 'masks'] if key in target} for target in targets]
            return self.model(images, train_targets)
        else:
            return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.forward(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.forward(images)
        score_threshold = 0.5  # TODO: Add this to config file 
        mask_threshold = 0.5 # TODO: Add this to config file 
        self.val_results_bbox = []
        self.val_results_mask = []
        self.coco_gt = COCO()
        coco_gt = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        for i, prediction in enumerate(predictions):
            image_id = targets[i]['image_id']
            for j, _ in enumerate(prediction['masks']):
                if prediction['scores'][j].item() < score_threshold:
                    continue
                binary_mask = (prediction['masks'][j, 0] >mask_threshold).cpu().numpy().astype(np.uint8)
                if np.sum(binary_mask) == 0:
                    continue
                rle_mask = maskUtils.encode(np.asfortranarray(binary_mask))
                self.val_results_mask.append({
                    'image_id': image_id,
                    'category_id': int(prediction['labels'][j]),
                    'segmentation': rle_mask,
                    'score': float(prediction['scores'][j])
                })
                x1, y1, x2, y2 = prediction['boxes'][j].cpu().numpy()
                self.val_results_bbox.append({
                'image_id': image_id,
                'category_id': int(prediction['labels'][j]),
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(prediction['scores'][j])
                })
        for i, target in enumerate(targets):
            coco_gt['images'].append(
                {'id': target['image_id'],
                 'width':images[i].shape[1],
                 'height': images[i].shape[2]}
            )
            coco_gt['categories'].append(
                {"id":1, "name": "shurb", "supercategory": "plant"}
            )
            if len(target['annotation_id']) == 0:
                coco_gt['annotations'].append(
                    {
                        "id": 0,
                        "image_id": target['image_id'],
                        "category_id": 1,
                        "bbox": [0,0,0,0],
                        "segmentation": maskUtils.encode(np.asfortranarray(np.zeros((images[i].shape[1], 
                                                                                    images[i].shape[2])).astype(np.uint8))),
                        'area':0,
                        'iscrowd':0
                    })
            else:
                for j, ann_id in enumerate(target['annotation_id']):
                    x1, y1, x2, y2 = target['boxes'][j].cpu().numpy().astype(np.float32)
                    coco_gt['annotations'].append(
                        {
                            "id": ann_id,
                            "image_id": target['image_id'],
                            "category_id": target['labels'][j],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "segmentation": maskUtils.encode(np.asfortranarray(target['masks'][j].cpu().numpy())),
                            'area':target['area'][j].item(),
                            'iscrowd':target['iscrowd'][j]
                        }
                ) 
        self.coco_gt.dataset = copy.deepcopy(coco_gt)
        self.coco_gt.createIndex()

    def predict_step(self, batch, batch_idx):
        images = batch
        predictions = self.forward(images)
        return predictions

    def on_validation_epoch_end(self):
        if len(self.val_results_bbox) > 0:
            coco_dt_bbox = self.coco_gt.loadRes(self.val_results_bbox)
            coco_eval_bbox = COCOeval(self.coco_gt, coco_dt_bbox, iouType='bbox')
            coco_eval_bbox.evaluate()
            coco_eval_bbox.accumulate()
            coco_eval_bbox.summarize()
            self.log_dict({
                "bbox_AP": coco_eval_bbox.stats[0],
                "bbox_AP50": coco_eval_bbox.stats[1],
                "bbox_AP75": coco_eval_bbox.stats[2]
            }, on_epoch=True)
        
        if len(self.val_results_mask) > 0:
            coco_dt_mask = self.coco_gt.loadRes(self.val_results_mask)
            coco_eval_mask = COCOeval(self.coco_gt, coco_dt_mask, iouType='segm')
            coco_eval_mask.evaluate()
            coco_eval_mask.accumulate()
            coco_eval_mask.summarize()
            self.log_dict({
                "mask_AP": coco_eval_mask.stats[0],
                "mask_AP50": coco_eval_mask.stats[1],
                "mask_AP75": coco_eval_mask.stats[2]
            }, on_epoch=True)
            
            # Reset results storage for the next epoch
            self.val_results_bbox.clear()
            self.val_results_mask.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005) # TODO make this flexiable
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
    
