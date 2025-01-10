#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train module for thunderseg
"""
import torch
import torch.utils.data
from lightning import LightningModule, LightningDataModule
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
torch.set_float32_matmul_precision('high')

class Train():
    def __init__(self, model: LightningModule, 
                 datamodule: LightningDataModule,
                 train_coco,
                 predict_coco,
                 batch_size:int=5, 
                 num_workers:int=0,
                 num_classes:int=2,
                 learning_rate:float=0.001,
                 max_epochs:int=100,
                 save_path = None,
                 callbacks = None,
                 ckpt_path = None
                 ):
        """The main entry point of the training process.
            Args:
                model: Lightningmodule, find more detail in https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
                datamodule: LightningDataModule, find more detail in https://lightning.ai/docs/pytorch/stable/data/datamodule.html
                batch_size: used in 
        """
        self.model = model(num_classes, learning_rate)
        self.datamodule = datamodule(train_coco, predict_coco, batch_size, num_workers)
        self.logger = TensorBoardLogger(save_dir=save_path, name='Train')
        self.trainer = Trainer(logger=self.logger,
                               max_epochs=max_epochs,
                               callbacks=callbacks,
                               log_every_n_steps=10,
                               default_root_dir=save_path
                               )
        self.ckpt_path = ckpt_path
        
    def fit(self):
        return self.trainer.fit(self.model, self.datamodule, ckpt_path=self.ckpt_path)
    
    def test(self):
        return self.trainer.test(self.model, self.datamodule, ckpt_path=self.ckpt_path)
    
    def validate(self):
        return self.trainer.validate(self.model, self.datamodule, ckpt_path=self.ckpt_path)
    
    def predict(self):
        return self.trainer.predict(self.model, self.datamodule, ckpt_path=self.ckpt_path)
    
        

