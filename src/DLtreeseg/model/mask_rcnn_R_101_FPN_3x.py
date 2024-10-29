import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# Equivalent Model Construction
def mask_rcnn_R_101_FPN_3x(number_classes, weight, threshold):
# Step 1: ResNet101 with FPN as the backbone
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)

    # Step 2: Create MaskRCNN model using the backbone
    # The number of classes is 91 for COCO (including background)
    model = MaskRCNN(backbone, num_classes=number_classes)

    # Step 3: Load the pre-trained weights for Mask R-CNN R_101_FPN_3x
    if weight is not None: 

        checkpoint = torch.load(weight, map_location='cpu')

    # Load the weights into the model
    model.load_state_dict(checkpoint['model'])

    # Step 4: Set the threshold for object detection score (like Detectron2)
    model.roi_heads.score_thresh = threshold

    # Step 5: Transfer the model to GPU (if available)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    return model

# Example Inference
