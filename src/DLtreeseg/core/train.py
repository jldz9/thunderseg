import torch
import lightning as L


class DrakeDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()

class MaskRCNN(L.LightningModule):
