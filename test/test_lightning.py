import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
    

class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    
transform = transforms.ToTensor()  
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform) 
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)


train_loader = DataLoader(train_set, num_workers=16)
valid_loader = DataLoader(valid_set, num_workers=16)
#autoencoder = LitAutoEncoder(Encoder(), Decoder())
model = LitAutoEncoder.load_from_checkpoint("/workspaces/thunderseg/lightning_logs/version_5/checkpoints/epoch=2-step=144000.ckpt")

# train model
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = L.Trainer(accelerator='gpu', devices=1)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path="/workspaces/thunderseg/lightning_logs/version_5/checkpoints/epoch=2-step=144000.ckpt")