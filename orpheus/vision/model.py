import torch
import os
import torch.nn as nn
from einops import repeat
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import ConcordanceCorrCoef, PearsonCorrCoef, MeanSquaredError


class TileTransformer(pl.LightningModule):
    def __init__(
        self,
        lr=2e-5,
        warm_up=1000,
        lr_decay=0.9999,
        layers=2,
        input_dim=768,
        decay=2e-5,
        dropout=0.1,
        latent_dim=512,
        heads=8,
        preds_output_dir="preds/visual"
    ):
        super().__init__()
        assert latent_dim % heads == 0
        self.warm_up_step = warm_up
        self.lr_decay = lr_decay
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = lr
        self.weight_decay = decay
        self.preds_output_dir = preds_output_dir
        self.save_hyperparameters()

        self.latent_embedder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=heads,
                dim_feedforward=latent_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=layers,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        self.reg_head = nn.Sequential(nn.LayerNorm(self.latent_dim),
                               nn.Linear(self.latent_dim, 1))

        setattr(self, "train_concordance", ConcordanceCorrCoef())
        setattr(self, "train_pearson", PearsonCorrCoef())
        setattr(self, "train_mse", MeanSquaredError())
        setattr(self, "val_concordance", ConcordanceCorrCoef())
        setattr(self, "val_pearson", PearsonCorrCoef())
        setattr(self, "val_mse", MeanSquaredError())


    def add_cls_token(self, x):
        if x.shape[0] > 1:
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
        else:
            cls_tokens = self.cls_token
        x = torch.cat((cls_tokens, x), dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch['x'])['y_hat']
        loss = self.calculate_loss(y_hat, batch['y'].view(-1, 1))
        self.log_all(loss, y_hat, batch['y'], "train")
        return loss
    
    @staticmethod
    def calculate_loss(y_hat, y):
        return nn.functional.mse_loss(y_hat, y, reduction="mean")

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch['x'])['y_hat']
        loss = self.calculate_loss(y_hat, batch['y'].view(-1, 1))
        self.log_all(loss, y_hat, batch['y'], "val")

    def log_all(self, loss, y_hat, y, subset):
        logging_kwargs = {"on_step": True,
                          "on_epoch": True,
                          "sync_dist": True,
                          "batch_size": y.shape[0]}

        mse_metric = getattr(self, f"{subset}_mse")
        mse_metric(y_hat.reshape(-1), y.reshape(-1))
        self.log(f"{subset}_mse", mse_metric, **logging_kwargs)
        
        concordance_metric = getattr(self, f"{subset}_concordance")
        concordance_metric(y_hat.reshape(-1), y.reshape(-1))
        self.log(f"{subset}_concordance", concordance_metric, **logging_kwargs)

        pearson_metric = getattr(self, f"{subset}_pearson")
        pearson_metric(y_hat.reshape(-1), y.reshape(-1))
        self.log(f"{subset}_pearson", pearson_metric, **logging_kwargs)

        self.log(f"{subset}_loss", loss, **logging_kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.forward(batch['x'])
        y_hat, emb = output['y_hat'], output['emb']
        emb_output_filenames = batch["output_visual_embedding_path"]
        if not os.path.exists(os.path.dirname(emb_output_filenames[0])):
            os.makedirs(os.path.dirname(emb_output_filenames[0]))
        for i, emb_output_filename in enumerate(emb_output_filenames):
            torch.save(emb[i].cpu(), emb_output_filename)
        for i, y_hat_i in enumerate(y_hat):
            prediction_file_name = os.path.join(self.preds_output_dir, batch["split"][i], f"{batch['case_id'][i]}.pt")
            if not os.path.exists(os.path.dirname(prediction_file_name)):
                os.makedirs(os.path.dirname(prediction_file_name))
            torch.save(y_hat_i.cpu(), prediction_file_name)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        def calc_lr(epoch):
                step = self.trainer.global_step
                if step < self.warm_up_step:
                    lr_scale = float(step) / self.warm_up_step
                else:
                    lr_scale = self.lr_decay ** (step - self.warm_up_step)
                return lr_scale

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=calc_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "reduce_on_plateau": False,
            },
            "monitor": "val_loss",
        }

    def forward(self, x):
        x = self.latent_embedder(x)
        x = self.add_cls_token(x)
        x = self.transformer(x)
        z = x[:, 0]
        return {'emb': z, 'y_hat': self.reg_head(z)}
