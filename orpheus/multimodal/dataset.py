from torch.utils.data import Dataset, DataLoader
import torch
from typing import List
import os
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from utils.utils import validate_dataframe


class EmbeddingDataset(Dataset):
    """
    Serves embeddings from .pt files
    emb_file_paths: List[str] = list of paths to pt files
    case_ids: List[str] = list of case ids
    scores: List[float] = list of scores
    """

    def __init__(
        self,
        df,
    ) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        return {
            "slide_emb": torch.load(row["output_visual_embedding_path"], map_location='cpu').float(),
            "report_emb": torch.load(row["output_linguistic_embedding_path"], map_location='cpu').float(),
            "y": torch.tensor(row["score"]).float(),
            "case_id": row["case_id"],
            "split": row["split"],
            "multimodal_emb_path": row["output_multimodal_embedding_path"],
        }
    

class MultimodalEmbeddingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataframe_path: str,
        num_workers: int = 0,
        batch_size: int = 1,
    ):
        super().__init__()
        self.dataframe_path = dataframe_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.save_hyperparameters()

        self.df = pd.read_csv(dataframe_path)
        self.df = self.df[self.df["output_multimodal_embedding_path"] != "NONE"]
        validate_dataframe(self.df)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            assert "train" in self.df["split"].unique(), "No train split in dataframe"
            assert "val" in self.df["split"].unique(), "No val split in dataframe"
            self.train_ds = EmbeddingDataset(
                self.df.loc[self.df["split"] == "train"]
            )
            self.val_ds = EmbeddingDataset(
                self.df.loc[self.df["split"] == "val"]
            )
        elif stage == "test":
            assert "test" in self.df["split"].unique(), "No test split in dataframe"
            self.test_ds = EmbeddingDataset(
                self.df.loc[self.df["split"] == "test"]
            )
        elif stage == "predict":
            self.predict_ds = EmbeddingDataset(self.df)
        else:
            raise NotImplementedError("Unknown stage {}".format(stage))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


if __name__ == "__main__":
    pass