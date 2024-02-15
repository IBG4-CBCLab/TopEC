import logging
from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

log = logging.getLogger(__name__)

class EnzymeLitModule(pl.LightningModule):
    r"""Training module with a DimeNet++ module as feature extractor.
    DimeNet++ is modified to be used for classification.
    Also embedding layer is changed from lookup to linear layer.

    Attributes:

        num_classes (int, optional): Total number of enzymes classes. Defaults to 7.
        lr (float, optional): Starting learning rate. Defaults to 0.001.

    """

    def __init__(
        self,
        net,
        num_classes: int = 96,
        class_weights: list = [1, 1, 1, 1, 1, 1, 1],
        lr: float = 0.001,
        n_gpus: int = 1,
        cutoff: int = 10,
        out_dim: int = 128,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_layers: int = 6,
        num_gaussians: int = 50,
        max_num_neighbors: int = 32,
        num_interactions: int = 6,
        dropout: int = 0.25,
        readout: str = "mean",
        resolution: str = 'residue',
    ):

        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        
        self.num_classes = num_classes
        print('num classes = ', num_classes)
        self.lr = lr
       
        if n_gpus > 1:
            self.sync_dist = True
        else:
            self.sync_dist = False

        class_weights = np.ones(num_classes) #change to balanced class weights
        self.class_weights = torch.FloatTensor(class_weights)

        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        self.train_macroACC = Accuracy(
            task='multiclass', num_classes=num_classes, average="macro"
        )
        self.val_macroACC = Accuracy(
            task='multiclass', num_classes=num_classes, average="macro"
        )
        self.test_macroACC = Accuracy(
            task='multiclass', num_classes=num_classes, average="macro"
        )

        self.train_ACC = Accuracy(task='multiclass', num_classes=num_classes, average="micro")
        self.val_ACC = Accuracy(task='multiclass', num_classes=num_classes, average="micro")
        self.test_ACC = Accuracy(task='multiclass', num_classes=num_classes, average="micro")

        
        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_macroacc_best = MaxMetric()
        
    def forward(self, x, pos, edge_index=None,  batch=None):
        
        batch = torch.zeros(x.size()[0], dtype=torch.long, device=self.device) if batch is None else batch
        
        return self.net(x, pos, edge_index, batch)


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_macroacc_best.reset()

    def step(self, batch: Any):
        
        _, logits = self.forward(x=batch.x, pos=batch.pos, edge_index=batch.edge_index, batch=batch.batch)
        loss = self.criterion(logits, batch.y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, batch.y


    def training_step(self, batch: Any, batch_idx: int):

        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_ACC(preds, targets)
        macroacc = self.train_macroACC(preds, targets)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,sync_dist=self.sync_dist)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True,logger=True,sync_dist=self.sync_dist)
        self.log("train/macroacc", macroacc, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=self.sync_dist)
       
        return {"loss": loss, "preds": preds, "targets": targets}

    # def training_epoch_end(self, outputs: List[Any]):
    #     # `outputs` is a list of dicts returned from `training_step()`
    #     pass


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_ACC(preds, targets)
        macroacc = self.val_macroACC(preds, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=self.sync_dist)
        
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):#, outputs: List[Any]):
        acc = self.val_ACC.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        
        macroacc = self.val_macroACC.compute()  # get val accuracy from current epoch
        self.val_macroacc_best.update(macroacc)
        self.log("val/macroacc_best", self.val_macroacc_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=self.sync_dist)
    
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_ACC(preds, targets)
        macroacc = self.test_macroACC(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        
        
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        
        acc = self.test_ACC.compute()  # get test accuracy from current epoch
        macroacc = self.test_macroACC.compute()  # get test accuracy from current epoch
        self.log("test/acc", acc, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log("test/macroacc", macroacc, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        
    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_ACC.reset()
        self.test_ACC.reset()
        self.val_ACC.reset()
        
        self.train_macroACC.reset()
        self.test_macroACC.reset()
        self.val_macroACC.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr)
   
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, logits = self.forward(x=batch.x, pos=batch.pos, edge_index=batch.edge_index, batch=batch.batch)
 
        return logits, F.softmax(logits, dim=1)

