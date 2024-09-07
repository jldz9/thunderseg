from DLtreeseg.utils.tool import COCO_format

coco = COCO_format({"description": "test"})
coco.add_licenses(license_id = [1,2,3], license_url = ['b','c','d'], license_name = ['aa','bb', 'cc'])
coco.add_images(id=[1,2,3,4], file_name=['a','b','c','d'], width=[3,4,5,6], height=[2,3,4,5],license=[], flickr_url=[], coco_url=[], date_captured=[])
coco.add_categories(id=[1,2,3,4], name=['a','b','c','d'], supercategory=['a','a','a','a'])
coco.add_annotations(id=[1,2,3,4], 
                     image_id=[1,1,2,2],
                     category_id=[1,1,1,1],
                     bbox=[[1,2,3,4],[2,2,3,4],[3,2,3,4],[4,2,3,4]],
                     area=[2,3,4,5],
                     iscrowd=[1,1,1,1],
                     segmentation=[[1,2,1,3,1,4],[1,2,1,4,1,5],[2,1,2,2,2,5],[3,2]],
                     keypoints=[],
                     num_keypoints=[],
                     bbox_mode='xywh')
coco.data
coco.save_json('/home/jldz9/DL/COCO.json')