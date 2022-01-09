import torch
import wandb
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule


class ChestXrayModel(LightningModule):

    def __init__(self, architecture: nn.Module = None, initialization: str = 'random', checkpoint: str = None,
                 num_classes: int = 14, loss_fn: str = 'BCELoss', optimizer_fn: str = 'adam', log_freq: int = 1000,
                 learning_rate: float = 0.001, freeze_encoder: bool = False,
                 **kwargs):
        super(ChestXrayModel, self).__init__()
        self.save_hyperparameters()
        self.architecture = architecture
        self.initialization = initialization
        self.freeze_encoder = freeze_encoder
        self.num_classes = num_classes
        if self.initialization in ("random", "imagenet"):
            pretrained = initialization == "imagenet"
            self.encoder = torch.hub.load(
                'pytorch/vision:v0.10.0', self.architecture, pretrained=pretrained)
            if 'resnet' in self.architecture:
                in_features = self.encoder.fc.in_features
                self.encoder.fc = nn.Identity()
                self.classification_head = nn.Linear(
                    in_features, self.num_classes)
            elif 'densent' in self.architecture:
                in_features = self.encoder.classifier.in_features
                self.encoder.classifier = nn.Identity()
                self.classification_head = nn.Linear(
                    in_features, self.num_classes)
            else:
                raise 'ViT Not Supported: Implementation of Vision Transformer Not Found.'
        else:
            self.encoder = torch.hub.load(
                'pytorch/vision:v0.10.0', self.architecture, pretrained=False)
            raise 'Custom Checkpoint Not Supported: Implementation of Custom Checkpoint Not Found.'
            # loaded_dict = torch.load(
            #     self.checkpoint, map_location=torch.device('cpu'))
            # TODO Need to fix loading from checkpoint
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.loss_fn = loss_fn
        self.loss = getattr(torch.nn, self.loss_fn)
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        metrics = torchmetrics.MetricCollection([
            torchmetrics.AUROC(average='macro', num_classes=self.num_classes),
        ]
        )
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()
        self.log_freq = log_freq
        wandb.watch((self.encoder, self.classification_head),
                    log_freq=self.log_freq)

    def forward(self, x):
        logits = self.model(x)
        ŷ = self.classification_head(logits)
        ŷ = torch.sigmoid(ŷ)
        return ŷ

    def training_step(self, batch, batch_idx):
        x, y = batch
        ŷ = self(x)
        loss = self.loss(ŷ, y)
        self.train_metrics(ŷ, y.int())
        return {
            'loss': loss,
            'predictions': ŷ,
            'labels': y
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ŷ = self(x)
        self.val_metrics(ŷ, y.int())
        loss = self.loss(ŷ, y)
        return {
            'loss': loss,
            'predictions': ŷ,
            'labels': y
        }

    def validation_epoch_end(self, outputs):
        loss = torch.tensor([x['loss'] for x in outputs])
        loss = loss.mean()
        self.log_dict(
            {'val_loss': loss, 'val_roc': self.val_metrics['AUROC']}, prog_bar=True, on_step=False, on_epoch=True)

    def training_epoch_end(self, outputs):
        loss = torch.tensor([x['loss'] for x in outputs])
        loss = loss.mean()
        self.log_dict(
            {'train_loss': loss, 'train_roc': self.train_metrics['AUROC']}, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_fn)(self.parameters(),
                                                            lr=self.learning_rate,
                                                            **self.kwargs
                                                            )
        return [optimizer]
