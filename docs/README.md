# Orpheus

![lyre](lyre.jpeg)

Transformer-based multimodal integration of H&E-stained whole-slide images and their corresponding synoptic pathology reports to regress prognostic/predictive scores for cancer. See the preprint applying this to Oncotype scores [here]().

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

#### Install Torch
- Follow instructions at: https://pytorch.org/get-started/locally/

#### Install other dependencies
```bash
conda install einops lightning wandb torchmetrics pandas numpy h5py datasets transformers evaluate scipy scikit-learn seaborn matplotlib statsmodels -c huggingface -c conda-forge

pip install -U 'jsonargparse[signatures]>=4.26.1'

pip install accelerate -U

pip install "dill<0.3.5"
```

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

## train

### train vision model
Adjust any parameters to suit your system in the config file.

`python orpheus/main.py fit --config orpheus/vision/config.yaml`

Logs are in `outputs/training_logs`, and checkpoints are in `outputs/vision-models`. Select the checkpoint with the lowest validation loss.

### embed with vision model
```bash
wandb disabled
python orpheus/main.py predict --config orpheus/vision/config.yaml --ckpt_path outputs/vision-models/{best_model}.ckpt
wandb enabled
```

Predictions are stored as individual `.pt` files in `preds/visual/{split}` named by `case_id` in the dataframe you provide, and the whole-slide embeddings are stored where you designate in the `output_visual_embedding_path`.

### train language model
`python orpheus/language/train.py --df_path scratch/example.csv`

Logs and checkpoints are in `outputs/text-models`. Select the checkpoint with the lowest validation loss.

### embed with language model
```bash
wandb disabled
python orpheus/language/infer.py --df_path scratch/example.csv --ckpt_path outputs/text-models/{best_model}
wandb enabled
```
Predictions are stored as individual `.pt` files in `preds/linguistic/{split}` named by `case_id` in the dataframe you provide, and the whole-slide embeddings are stored where you designate in the `output_linguistic_embedding_path`.

### train multimodal model
`python orpheus/main.py fit --config orpheus/multimodal/config.yaml`

Logs and checkpoints are in `outputs/multimodal-models`. Select the checkpoint with the lowest validation loss.

## embed with multimodal model
```bash
wandb disabled
python orpheus/main.py predict --config orpheus/multimodal/config.yaml --ckpt_path outputs/multimodal-models/{best_model}.ckpt
wandb enabled
```

Predictions are stored as individual `.pt` files in `preds/multimodal/{split}` named by `case_id` in the dataframe you provide, and the multimodal embeddings are stored where you designate in the `multimodal_embedding_path`.