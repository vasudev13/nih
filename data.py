from typing import List
import pandas as pd
import numpy as np
from PIL import Image
import torch
from pytorch_lightning import LightningDataModule
from torch.data.utils import Dataset, DataLoader
import torchvision.transforms as transforms
import transformations
from hydra.utils import instantiate


class ChestXrayDataset(Dataset):
    """ Pytorch dataset for NIH ChestX-ray8 dataset.
        https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf
    """

    def __init__(self, image_paths: List[str], labels: List[np.ndarray], transformations: transforms = None):
        """
        Args:
            image_paths (List[str]): List of paths to radiograph images
            labels (List[np.ndarray]): List of where each element corresponds to observation of 14 pathologies.
            transformations (transforms, optional): torchvision transforms to apply on each image. Defaults to None.
        """
        super(ChestXrayDataset, self).__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transformations = transformations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index]).convert('RGB')
        x = self.transformations(x)
        y = torch.tensor(self.labels[index])
        return x, y


class ChestXrayDataModule(LightningDataModule):
    """ Pytorch-Lightning DataModule for NIH ChestX-ray8 dataset.
    """

    def __init__(self,
                 path: str = None,
                 train_transforms: transforms = None,
                 val_transforms: transforms = None,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 train_fraction: int = 1.0,
                 ):
        """ Args:
                path (str, optional): file path to split(train/test) files for NIH ChestX-ray8. Defaults to None.
                train_transforms (transforms, optional): torchvision transforms to be use during training. Defaults to None.
                val_transforms (transforms, optional): torchvision transforms to be use during validation/testing. Defaults to None.
                batch_size (int, optional): number of samples in mini-batch. Defaults to 64.
                num_workers (int, optional): number of subprocesses to use for data loading. Defaults to 4.
                train_fraction (int, optional): fraction of training samples to be use for training the given model. Defaults to 1.0.
        """
        super().__init__()
        self.path = path
        self.train_fraction = train_fraction
        self.train_transforms = instantiate(train_transforms)
        self.val_transforms = instantiate(val_transforms)
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage):
        if stage == 'fit':
            self.train_df = pd.read_csv(f'{self.path}/train.csv')
            self.train_df = self.train_df.sample(frac=self.train_fraction)
            self.val_df = pd.read_csv(f'{self.path}/test.csv')
            self.train_dataset = ChestXrayDataset(
                image_paths=self.train_df['Image Path'].values.tolist(),
                labels=self.train_df['One Hot Label'].values.tolist(),
                transformations=self.train_transforms
            )
            self.val_dataset = ChestXrayDataset(
                image_paths=self.val_df['Image Path'].values.tolist(),
                labels=self.val_df['One Hot Label'].values.tolist(),
                transformations=self.val_transforms
            )
        if stage == 'test':
            self.test_df = pd.read_csv(f'{self.path}/test.csv')
            self.test_dataset = ChestXrayDataset(
                image_paths=self.test_df['Image Path'].values.tolist(),
                labels=self.test_df['One Hot Label'].values.tolist(),
                transformations=self.val_transforms
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return test_loader
