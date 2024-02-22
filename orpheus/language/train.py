from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import pandas as pd
from datasets import Dataset
from argparse import ArgumentParser
import sys
sys.path.append("orpheus")
from utils.utils import validate_dataframe


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return PEARSON.compute(predictions=predictions, references=labels)


def preprocess_function(examples):
    return TOKENIZER(examples["text"], truncation=True, max_length=512)


def load_dataset(df):
    # df = df[["text", "score", "case_id"]]
    df = df.rename(columns={"score": "label"})
    return Dataset.from_pandas(df)

TOKENIZER = AutoTokenizer.from_pretrained("tsantos/PathologyBERT")
PEARSON = evaluate.load("pearsonr")

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--df_path", type=str, required=True)
    ARGS = PARSER.parse_args()

    MODEL = AutoModelForSequenceClassification.from_pretrained(
        "tsantos/PathologyBERT", problem_type="regression", num_labels=1
    )

    MODEL.cuda()
    
    DF = pd.read_csv(ARGS.df_path)
    DF = DF[DF["output_linguistic_embedding_path"] != "NONE"]
    validate_dataframe(DF)

    TRAIN_DATASET = load_dataset(DF[DF.split == "train"])
    TOKENIZER_TRAIN_DATASET = TRAIN_DATASET.map(preprocess_function, batched=True)

    VAL_DATASET = load_dataset(DF[DF.split == "val"])
    TOKENIZED_VAL_DATASET = VAL_DATASET.map(preprocess_function, batched=True)

    DATA_COLLATOR = DataCollatorWithPadding(tokenizer=TOKENIZER)

    training_args = TrainingArguments(
        output_dir="outputs/text-models",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        dataloader_num_workers=0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="wandb",
        run_name="example",
    )

    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=TOKENIZER_TRAIN_DATASET,
        eval_dataset=TOKENIZED_VAL_DATASET,
        tokenizer=TOKENIZER,
        data_collator=DATA_COLLATOR,
        compute_metrics=compute_metrics,
    )

    trainer.train()
