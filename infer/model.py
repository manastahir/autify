import torch, torchvision
import pytorch_lightning as pl
import os

class Model(pl.LightningModule):
    def __init__(self, model_name="resnet18", num_classes=3):
        super().__init__()
        self.model = getattr(torchvision.models, model_name)(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, batch):
        x = self.model(batch)
        return x