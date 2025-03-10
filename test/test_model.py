
import sys
import lightning as l
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


sys.path.append('/home/jldz9/DL/DL_packages/thunderseg/src')

from thunderseg.model import attemtion_unet



checkpoint_callback = ModelCheckpoint(dirpath='/home/jldz9/thunderseg_test/result', 
                                      filename='model-{epoch:03d}-{val_f1:.2f}', 
                                      save_top_k=5, 
                                      monitor='val_f1',
                                      mode='max',
                                      every_n_epochs=20)
model = attemtion_unet.ATTUNet()
data_module = attemtion_unet.ATTUNetDataModule(img, shp)
logger = TensorBoardLogger(save_dir='/home/jldz9/thunderseg_test/result', name='logs')
trainer = l.Trainer(max_epochs=500, accelerator='auto', logger=logger, log_every_n_steps=50, callbacks=[checkpoint_callback])

trainer.fit(model, data_module)




