import torch
import torch.nn as nn

from torchvision.ops import FeaturePyramidNetwork

from thunderseg.blocks.attention import SEBlock, CBAM

class SE_FPN(FeaturePyramidNetwork):
    """A simple attention FPN implementation use SqueezeExcitation. Add SEBlock to each FPN output for more accurate feature selection.
    """
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int = 256,
        extra_blocks: str = None,  # keep the same input as torchvision.ops.FeaturePyramidNetwork
        norm_layer: nn.Module = None,
        attention_channels: int = None
    ):
        super().__init__(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
                norm_layer=norm_layer
            )
        # Add se block for each FPN output
        self.se_blocks = nn.ModuleList([
            SEBlock(out_channels, attention_channels)
            for _ in in_channels_list
        ])
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Use original FPN forward
        results = super().forward(inputs)
        # Apply SEBlock to each FPN output
        for key, se in zip(results.keys(), self.se_blocks):
            results[key] = se(results[key])
        return results

class CBAM_FPN(FeaturePyramidNetwork):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__(in_channels_list, out_channels)
        self.cbam_modules = nn.ModuleList([
            CBAM(out_channels) for _ in in_channels_list
        ])
    def forward(self, inputs):
        features = super().forward(inputs)
        for key, cbam in zip(features.keys(), self.cbam_modules):
            features[key] = cbam(features[key])
        return features