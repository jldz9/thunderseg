import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS

config_file = '/home/jldz9/DL/test/mmtest/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
checkpoint_file = '/home/jldz9/DL/test/mmtest/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
register_all_modules()
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
image = mmcv.imread('/home/jldz9/DL/DL_packages/DLtreeseg/test/demo.jpg',channel_order='rgb')
inference_detector(model, image)
result = inference_detector(model, image)
print(result)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
visualizer.add_datasample(
    'result',
    image,
    data_sample=result,
    draw_gt = None,
    wait_time=0,
)
visualizer.show()