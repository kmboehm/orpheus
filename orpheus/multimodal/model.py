import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
import torchmetrics
import torch
import os


class SimpleAttention(pl.LightningModule):
    def __init__(
        self,
        slide_input_dim=512,
        report_input_dim=768,
        lr=1e-5,
        weight_decay=0.0,
        lr_decay=0.9999,
        model_name="SimpleAttention",
        warm_up_steps=4000,
        latent_dim=512,
        dropout=0.1,
        post_fusion_dim=0,
        proj_dim=0,
        scheduler_type="constant",
        grad_blend=False,
        pretrained=True,
        preds_output_dir="preds/multimodal"
    ):
        super().__init__()
        self.slide_input_dim = slide_input_dim
        self.report_input_dim = report_input_dim

        self.scheduler_type = scheduler_type
        self.model_name = model_name
        self.warm_up_steps = warm_up_steps
        self.lr_decay = lr_decay
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.preds_output_dir = preds_output_dir
        self.save_hyperparameters()

        if model_name == "SimpleAttention":
            self.report_proj = nn.Sequential(nn.Linear(report_input_dim, latent_dim))
            self.slide_proj = nn.Sequential(nn.Linear(slide_input_dim, latent_dim))
            self.attn = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softmax(dim=1))
            self.warp = nn.Identity()
        self.reg_head = nn.Linear(latent_dim, 1)

        setattr(self, "val_concord", torchmetrics.ConcordanceCorrCoef())
        setattr(self, "train_concord", torchmetrics.ConcordanceCorrCoef())
        setattr(self, "val_pearson", torchmetrics.PearsonCorrCoef())
        setattr(self, "train_pearson", torchmetrics.PearsonCorrCoef())

    def forward(self, batch):
        x = torch.stack(
            [self.report_proj(batch["report_emb"]), self.slide_proj(batch["slide_emb"])], dim=1
        )
        a = self.attn(x)
        z = torch.sum(a * x, dim=1)
        z = self.warp(z)
        y_hat = self.reg_head(z)
        return y_hat

    def calculate_loss(self, y_hat, target):
        loss = nn.functional.mse_loss(y_hat, target.view(-1, 1), reduction="none").float()
        loss = loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.calculate_loss(y_hat, batch["y"])
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch["y"].shape[0])
        self._update_metrics("train", y_hat, batch["y"])
        self._log_metrics("train", batch["y"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.calculate_loss(y_hat, batch["y"])
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch["y"].shape[0])
        self._update_metrics("val", y_hat, batch["y"])
        self._log_metrics("val", batch["y"].shape[0])
        return loss.item()

    def _update_metrics(self, subset, preds, targets_):
        targets = targets_.view(-1, 1)
        pearson_metric = getattr(self, f"{subset}_pearson")
        concord_metric = getattr(self, f"{subset}_concord")
        pearson_metric(preds, targets)
        concord_metric(preds, targets)

    def _log_metrics(self, subset, batch_size):
        pearson_metric = getattr(self, f"{subset}_pearson")
        concord_metric = getattr(self, f"{subset}_concord")
        self.log(
            f"{subset}_pearson",
            pearson_metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size
        )
        self.log(
            f"{subset}_concord",
            concord_metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_type == "lambda":

            def calc_lr(epoch):
                step = self.trainer.global_step
                if step < self.warm_up_steps:
                    lr_scale = float(step) / self.warm_up_steps
                else:
                    lr_scale = self.lr_decay ** (step - self.warm_up_steps)
                return lr_scale

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=calc_lr)
        elif self.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000)
        elif self.scheduler_type == "constant":
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
        else:
            raise ValueError("scheduler type {} not supported".format(self.scheduler_type))

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


class TensorFusionNetwork(SimpleAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = None
        self.post_fusion_dim = kwargs["post_fusion_dim"]
        self.proj_dim = kwargs["proj_dim"]
        self.save_hyperparameters()
        print(self.proj_dim)
        if self.proj_dim > 0:
            self.report_proj = nn.Linear(self.hparams.report_input_dim, self.proj_dim)
            self.slide_proj = nn.Linear(self.hparams.slide_input_dim, self.proj_dim)
            self.numel = (self.proj_dim + 1) ** 2
        else:
            self.report_proj = nn.Identity()
            self.slide_proj = nn.Identity()
            self.numel = (self.hparams.report_input_dim + 1) * (self.hparams.slide_input_dim + 1)
        self.reg_head = nn.Sequential(
            nn.Linear(self.numel, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, 1),
        )
        # )
        self.dropout = nn.Dropout(self.hparams.dropout)
        # self.batch_norm = nn.BatchNorm1d(self.numel)

    def forward(self, batch, return_emb=False):
        r = self.report_proj(batch["report_emb"])
        s = self.slide_proj(batch["slide_emb"])
        r = self.prepend_one(r)
        s = self.prepend_one(s)
        z = self.fuse(r, s)
        if return_emb:
            z_full = z.detach().clone()
            z = self.dropout(z)
            y_hat = self.reg_head(z)
            for layer in self.reg_head[:-1]:
                z_full = layer(z_full)
            return y_hat, z_full
        else:
            z = self.dropout(z)
            y_hat = self.reg_head(z)
            return y_hat

    @staticmethod
    def prepend_one(t):
        b = t.shape[0]
        return torch.cat([torch.ones(b, 1).to(t.device), t], dim=1)

    @staticmethod
    def fuse(r, s):
        return torch.einsum("bp,bq->bpq", r, s).view(r.shape[0], -1)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        y_hat, emb = self.forward(batch, return_emb=True)
        emb_output_filenames = batch["multimodal_emb_path"]
        if not os.path.exists(os.path.dirname(emb_output_filenames[0])):
            os.makedirs(os.path.dirname(emb_output_filenames[0]))
        for i, emb_output_filename in enumerate(emb_output_filenames):
            torch.save(emb[i].cpu(), emb_output_filename)
        for i, y_hat_i in enumerate(y_hat):
            prediction_file_name = os.path.join(self.preds_output_dir, batch["split"][i], f"{batch['case_id'][i]}.txt")
            if not os.path.exists(os.path.dirname(prediction_file_name)):
                os.makedirs(os.path.dirname(prediction_file_name))
            torch.save(y_hat_i.cpu(), prediction_file_name)
