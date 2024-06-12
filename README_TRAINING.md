# 4M Training

We provide instructions on how to pre-train 4M models and where to find the most relevant files. Please see the 4M paper for more background on our pre-training strategy. For instructions on how to train prepare the training datasets and train the tokenizers, please see [README_DATA.md](README_DATA.md) and [README_TOKENIZATION.md](README_TOKENIZATION.md).

## Structure

#### `fourm/models/`
- Code related to the 4M models
- Some important files:
    - `models/fm.py`: Main file defining the `FourM` module, containing all model architecture definitions and forward pass logic.
    - `models/encoder_embeddings.py` and `models/decoder_embeddings.py`: Contains per-modality modules that map the input tokens (or image patches) to embeddings, and the embeddings to logits. Also adds the positional and modality embeddings to tokens.
    - `fm_vit.py`: Contains the `FourMViT` module, which adapts the 4M models to behave as simple RGB-only ViT.
    - `generate.py`: Contains sampling logic & utilities for any-to-any generation with trained 4M models.

#### `fourm/data/`
- Handles data loading, preparation, augmentation, and input/target masking.
- Some important files:
    - `data/modality_info.py`: Defines modality metadata like name, type, vocabulary size, and encoder/decoder embedding modules.
    - `data/unified_datasets.py`: Loads aligned multimodal datasets, either locally or from cloud object stores (e.g. S3).
    - `data/modality_transforms.py`: Applies aligned data augmentations across modalities via UnifiedDataTransform. Also contains per-modality preprocessing and augmentation.
    - `data/masking.py`: Performs multimodal input/target masking based on provided token budgets and Dirichlet sampling parameters.


## General information

### Configs

Training runs are configured using YAML files that are organized as follows:
- Main training config: Contains most training information and hyperparameters (e.g. model architecture details, number of training steps, etc.), as well as logging and saving information. See [here](cfgs/default/4m/models/main/4m-b_mod7_500b.yaml) for an example.
- Data config: Provides details about the training data mix, including source datasets, input and target modalities, dataset paths, modality name mappings, etc. See [here](cfgs/default/4m/data/cc12m/main/mix_mod7_all2all_rgb2all_a0.5.yaml) for an example. The path to the data config needs to be specified in the main training config.
- Alphas configs: Defines the Dirichlet distribution hyperparameters used to sample proportions of tokens from each modality during training, and enables defining mixture of Dirichlet distributions. See [here](cfgs/default/4m/alphas_mixture/main/mix_mod7_all2all_rgb2all_a0.5.yaml) for an example. The path(s) to the alphas config(s) need to be specified in the data config.

Optionally, command-line arguments can be used to override some config information. To modify training settings, either either edit / add config files or provide additional command-line arguments.


### Training 4M Models

The 4M training script supports multi-node training with PyTorch Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP).

To train a 4M model using DDP (recommended for B-sized models) on a 8 GPU node, run:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_training_4m.py \
--config cfgs/default/4m/models/<config>.yaml
```

To train a 4M model using FSDP (more memory efficient for L and XL models) on a 8 GPU node, run: 


```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_training_4m_fsdp.py \
--config cfgs/default/4m/models/<config>.yaml
```


The training configurations for the 4M models in our papers are:

| Model | # Modalities | # Parameters | # GPUs | Config |
|-|-|-|-|-|  
| 4M-B | 7 | 198M | 64x A100 | [link](cfgs/default/4m/models/main/4m-b_mod7_500b.yaml) |
| 4M-L | 7 | 705M | 64x A100 | [link](cfgs/default/4m/models/main/4m-l_mod7_500b.yaml) |
| 4M-XL | 7 | 2.8B | 128x A100 | [link](cfgs/default/4m/models/main/4m-xl_mod7_500b.yaml) |
| 4M-B | 21 | 198M | 64x A100 | [link](cfgs/default/4m/models/main/4m-b_mod21_500b.yaml) |
| 4M-L | 21 | 705M | 64x A100 | [link](cfgs/default/4m/models/main/4m-l_mod21_500b.yaml) |
| 4M-XL | 21 | 2.8B | 128x A100 | [link](cfgs/default/4m/models/main/4m-xl_mod21_500b.yaml) |