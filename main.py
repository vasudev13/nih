from pathlib import Path

from data import *
from model import *
from transformations import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def cli_hydra(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    wandb_logger = instantiate(cfg.wandb)
    wandb_logger.log_hyperparams(cfg)

    train_transforms = instantiate(cfg.train_transforms)
    val_transforms = instantiate(cfg.val_transforms)

    data_module = instantiate(
        train_transforms=train_transforms, val_transforms=val_transforms, **cfg.data)
    model = instantiate(cfg.model)
    model_save_checkpoint = instantiate(cfg.model_save_callback)
    trainer = pl.Trainer(
        logger=[wandb_logger], callbacks=[model_save_checkpoint], **cfg.trainer
    )
    trainer.fit(model=model, datamodule=data_module)
    trainer.test()


if __name__ == "__main__":
    cli_hydra()
