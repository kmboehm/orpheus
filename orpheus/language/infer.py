from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader, Dataset
import os
from train import preprocess_function, load_dataset
import argparse
import sys
sys.path.append("orpheus")
from utils.utils import validate_dataframe
import pandas as pd


def save_outputs(outputs, output_emb_path, split, case_id):
    pred = outputs["logits"].flatten()

    pred_dir = f"preds/linguistic/{split}"
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    torch.save(pred, f"{pred_dir}/{case_id}.pt")

    hidden_states = outputs["hidden_states"]
    embedding = hidden_states[-1]  # 1, seq_len, 768
    cls_embedding = embedding[0][0]  # 768: get embedding specifically of CLS token

    if not os.path.exists(os.path.dirname(output_emb_path)):
        os.makedirs(os.path.dirname(output_emb_path))
    torch.save(cls_embedding, output_emb_path)


class TokensDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __getitem__(self, idx):
        batch = self.tokenized_dataset[idx]
        outputs = {
            "input_ids": torch.Tensor(batch["input_ids"]).long(),
            "attention_mask": torch.Tensor(batch["attention_mask"]),
            "token_type_ids": torch.Tensor(batch["token_type_ids"]).int(),
            "case_id": batch["case_id"],
            "split": batch["split"],
            "output_linguistic_embedding_path": batch["output_linguistic_embedding_path"],
        }
        return outputs

    def __len__(self):
        return len(self.tokenized_dataset)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--df_path", type=str, required=True)
    PARSER.add_argument("--ckpt_path", type=str, required=True)
    ARGS = PARSER.parse_args()

    DF = pd.read_csv(ARGS.df_path)
    validate_dataframe(DF)

    for SET_TYPE in ["train", "val", "test"]:
        DATASET = load_dataset(DF[DF.split == SET_TYPE])
        TOKENIZED_DATASET = DATASET.map(preprocess_function, batched=True)

        MODEL = AutoModelForSequenceClassification.from_pretrained(
            ARGS.ckpt_path, output_hidden_states=True, output_attentions=True
        )

        MODEL.cuda()

        PROGRESS_BAR = tqdm.tqdm(range(len(TOKENIZED_DATASET)))

        LOADER = DataLoader(
            TokensDataset(TOKENIZED_DATASET),
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        for BATCH in LOADER:
            with torch.no_grad():
                BATCH["input_ids"] = BATCH["input_ids"].cuda()
                BATCH["attention_mask"] = BATCH["attention_mask"].cuda()
                BATCH["token_type_ids"] = BATCH["token_type_ids"].cuda()
                MODEL.eval()
                OUTPUTS = MODEL(
                    input_ids=BATCH["input_ids"],
                    attention_mask=BATCH["attention_mask"],
                    token_type_ids=BATCH["token_type_ids"],
                )
                save_outputs(OUTPUTS, BATCH["output_linguistic_embedding_path"][0], BATCH["split"][0], BATCH["case_id"][0])
            PROGRESS_BAR.update(1)
