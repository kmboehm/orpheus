# Orpheus

<img src="lyre.jpeg" alt="lyre" width="450"/>

Transformer-based multimodal integration of H&E-stained whole-slide images and their corresponding synoptic pathology reports to regress prognostic/predictive scores for cancer. See the preprint applying this to Oncotype scores [here](https://www.biorxiv.org/content/10.1101/2024.02.23.581806v1).

## Setup
### Extract tile-wise embeddings 
- Use [STAMP](https://github.com/KatherLab/STAMP)
- Convert to `.pt` files:
```python
import torch
from h5py import File
import os

stamp_dir = 'path/to/stamp/output'
os.mkdir('pt_files')
file_names = os.listdir(stamp_dir)  
for file_name in file_names:
    with File(os.path.join(stamp_dir, file_name), "r") as f:
        embeddings = f["feats"][:]
    embeddings = torch.from_numpy(embeddings).float()
    file_name_pt = file_name.replace('.h5', '.pt')
    torch.save(embeddings, os.path.join("pt_files", file_name_pt))
```

### Install Orpheus
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 einops lightning wandb torchmetrics pandas numpy h5py datasets transformers evaluate scipy scikit-learn seaborn matplotlib statsmodels accelerate tokenizers=0.13.02 -c pytorch -c nvidia -c huggingface -c conda-forge

pip install -U 'jsonargparse[signatures]>=4.26.1'
```
As needed, modify the PyTorch-related packages per the instructions at: https://pytorch.org/get-started/locally/

### Prepare data
- expects .csv file with columns:
    - **case_id**: any identifier for the image-text pair
    - **score**: float value within [0, 1]
    - **input_visual_embedding_path**: path to `.pt` file containing CTransPath embeddings derived using STAMP. `NONE` for rows without images.
    - **text**: text from synoptic pathology report. `NONE` for rows without text.
    - **split**: one of [train,val,test]
    - **output_visual_embedding_path**: desired location of `.pt` file containing whole-slide image embedding derived using Orpheus visual model. `NONE` for rows without images.
    - **output_linguistic_embedding_path**: desired location of `.pt` file containing aggregate text embedding derived using Orpheus visual model. `NONE` for rows without text.
    - **output_multimodal_embedding_path**: desired location of `.pt` file containing multimodal Orpheus-derived text-image embedding. `NONE` for rows with one missing modality.

If you like, you can create an example dataset in `orpheus/scratch` using `python orpheus/utils/utils.py`

### Log in to W&B
`wandb login`

## Train

### Train vision model
Adjust any parameters to suit your system in the config file.

`python orpheus/main.py fit --config orpheus/vision/config.yaml`

Logs are in `outputs/training_logs`, and checkpoints are in `outputs/vision-models`. Select the checkpoint with the lowest validation loss.

### Generate visual embeddings
```bash
wandb disabled
python orpheus/main.py predict --config orpheus/vision/config.yaml --ckpt_path outputs/vision-models/{best_model}.ckpt
wandb enabled
```

Predictions are stored as individual `.pt` files in `preds/visual/{split}` named by `case_id` in the dataframe you provide, and the whole-slide embeddings are stored where you designate in the `output_visual_embedding_path`.

### Train language model
`python orpheus/language/train.py --df_path scratch/example.csv`

Logs and checkpoints are in `outputs/text-models`. Select the checkpoint with the lowest validation loss.

### Generate linguistic embeddings
```bash
wandb disabled
python orpheus/language/infer.py --df_path scratch/example.csv --ckpt_path outputs/text-models/{best_model}
wandb enabled
```
Predictions are stored as individual `.pt` files in `preds/linguistic/{split}` named by `case_id` in the dataframe you provide, and the whole-slide embeddings are stored where you designate in the `output_linguistic_embedding_path`.

### Train multimodal model
`python orpheus/main.py fit --config orpheus/multimodal/config.yaml`

Logs and checkpoints are in `outputs/multimodal-models`. Select the checkpoint with the lowest validation loss, and compare against unimodal models in W&B.

<img src="wnb_chart.png" width="450"/>


### Generate multimodal embeddings
```bash
wandb disabled
python orpheus/main.py predict --config orpheus/multimodal/config.yaml --ckpt_path outputs/multimodal-models/{best_model}.ckpt
wandb enabled
```

Predictions are stored as individual `.pt` files in `preds/multimodal/{split}` named by `case_id` in the dataframe you provide, and the multimodal embeddings are stored where you designate in the `multimodal_embedding_path`.

### Align multimodal scores
`python orpheus/multimodal/align.py --df_path scratch/example.csv --img_pred_dir preds/visual --lan_pred_dir preds/linguistic --mult_pred_dir preds/multimodal --output_df_path all_predictions.csv`

Performs final alignment of multimodal scores and adds `pred_vis, pred_lan, pred_mul` to your initial dataframe, which it saves, e.g. at `all_predictions.csv` in this example

## Evaluate
`python eval.py --pred_df_path all_predictions.csv`

Generates `metrics.json` containing regression metrics for multimodal, visual, and linguistic models for training, validation, and test sets: r^2, mean average error with 95% C.I., concordance correlation coefficient with 95% C.I., Pearson correlation with 95% C.I. and p-value. Also generates one bar plot for each metric at `plots/{metric}`.