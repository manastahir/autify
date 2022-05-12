import torch, torchvision
import pytorch_lightning as pl
import torchmetrics
import os
class Model(pl.LightningModule):
    def __init__(self, model_name="resnet18", num_classes=3, lr=1e-3, optimizer_name="Adam", freeze_until=None, weights=None):
        super().__init__()
        self.model = getattr(torchvision.models, model_name)(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        if freeze_until is not None:
            for l in self.model.named_children():
                if l[0] == freeze_until:
                  break
                for params in l[1].parameters():
                    params.requires_grad = False
        if weights:
            weights=torch.tensor(weights, dtype=torch.float)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weights, 
                                                  label_smoothing=0.1)
        self.train_accuracy = torchmetrics.F1Score()
        self.valid_accuracy = torchmetrics.F1Score()
        self.epoch = 0
        self.lr = lr
        self.optimizer_name = optimizer_name

    def forward(self, batch):
        x = self.model(batch)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.ce_loss(y_hat, y)
        
        accuracy = self.train_accuracy(torch.nn.functional.softmax(y_hat, dim=-1), y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, on_epoch=True, prog_bar=True, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.ce_loss(y_hat, y)
        
        accuracy = self.valid_accuracy(torch.nn.functional.softmax(y_hat, dim=-1), y)
        
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_accuracy", accuracy, on_epoch=True, prog_bar=True, on_step=True)
    
    def validation_epoch_end(self, outputs):
        accuracy = self.valid_accuracy.compute()
        self.log('valid_accuracy_epoch', accuracy)
        self.valid_accuracy.reset()
        # print(f'{self.epoch}: Validation accuracy: ', accuracy.cpu().numpy())

    def training_epoch_end(self, outputs):
        accuracy = self.train_accuracy.compute()
        self.log('train_accuracy_epoch', accuracy)
        self.train_accuracy.reset()
        # print(f'{self.epoch}: Training accuracy: ', accuracy.cpu().numpy())
        self.epoch += 1  

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), lr=self.lr)
        return optimizer