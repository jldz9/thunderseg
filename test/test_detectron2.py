from detectron2.model_zoo import get_config
from detectron2.modeling import build_model
base_model: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
weight = '/home/jldz9/DL/models/230729_05dates.pth'

cfg = get_config(base_model)
cfg.MODEL.WEIGHTS = weight
cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.GAMMA = 0.1
cfg.MODEL.BACKBONE.FREEZE_AT = 3
cfg.SOLVER.WARMUP_ITERS = 120
cfg.SOLVER.MOMENTUM = 0.9
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 1024
cfg.SOLVER.BASE_LR = 0.0003389
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.TEST.EVAL_PERIOD = 100
cfg.RESIZE = "fixed"
cfg.INPUT.MIN_SIZE_TRAIN = 1000
