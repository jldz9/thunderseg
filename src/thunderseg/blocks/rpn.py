from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork
#from thunderseg.utils.tool import flexible_RPNHead

def anchor_lt_small_obj():
    """The original anchor generator used in Maskrcnn does not works for small objects.
    This function return a anchor generator with smaller anchor sizes and aspect ratios specific for smaller than 50x50 pixels objects.
    """
    anchor_sizes = ((4, 8, 16),(8, 16,24), (32,),(32,))
    aspect_ratios = ((0.25, 0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    return anchor_generator
'''
def rpn_head_small_obj(in_channels: int):
    """The original RPN head used in Maskrcnn does not works for small objects.
    This function return a RPN head with smaller anchor sizes and aspect ratios specific for smaller than 50x50 pixels objects.
    """
    rpn_head = flexible_RPNHead(
        anchorgenerator=anchor_lt_small_obj(),
        in_channels=in_channels,
        conv_depth=1
    )
    return rpn_head'
'''

def rpn_small_obj(in_channels: int):
    """The original RPN used in Maskrcnn does not works for small objects.
    This function return a RPN with smaller anchor sizes and aspect ratios specific for smaller than 50x50 pixels objects.
    """
    rpn = RegionProposalNetwork(
        anchor_generator=anchor_lt_small_obj(),
        head=rpn_head_small_obj(in_channels),
        fg_iou_thresh=0.5,       # 
        bg_iou_thresh=0.3,       # 
        batch_size_per_image=256, # 
        positive_fraction=0.5,   # 
  
        pre_nms_top_n=2000,      # 
        post_nms_top_n=1000,     #
        nms_thresh=0.7           # 
    )
    return rpn