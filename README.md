# 4M: Massively Multimodal Masked Modeling

*A framework for training any-to-any multimodal foundation models. <br>Scalable. Open-sourced. Across tens of modalities and tasks.*

EPFL - Apple

[`Website`](https://4m.epfl.ch) | [`BibTeX`](#citation)  | [`🤗 Demo`](https://huggingface.co/spaces/EPFL-VILAB/4M)

Official implementation and pre-trained models for :

[**4M: Massively Multimodal Masked Modeling**](https://arxiv.org/abs/2312.06647), NeurIPS 2023 (Spotlight) <br>
*[David Mizrahi](https://dmizrahi.com/)\*, [Roman Bachmann](https://roman-bachmann.github.io/)\*, [Oğuzhan Fatih Kar](https://ofkar.github.io/), [Teresa Yeo](https://aserety.github.io/), [Mingfei Gao](https://fly6464.github.io/), [Afshin Dehghan](https://www.afshindehghan.com/), [Amir Zamir](https://vilab.epfl.ch/zamir/)*

[**4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities**](https://arxiv.org/abs/2406.09406), NeurIPS 2024 <br>
*[Roman Bachmann](https://roman-bachmann.github.io/)\*, [Oğuzhan Fatih Kar](https://ofkar.github.io/)\*, [David Mizrahi](https://dmizrahi.com/)\*, [Ali Garjani](https://garjania.github.io/), [Mingfei Gao](https://fly6464.github.io/), [David Griffiths](https://www.dgriffiths.uk/), [Jiaming Hu](https://scholar.google.com/citations?user=vm3imKsAAAAJ&hl=en), [Afshin Dehghan](https://www.afshindehghan.com/), [Amir Zamir](https://vilab.epfl.ch/zamir/)*

<br>

![4M main figure](./assets/4M_main_fig_darkmode.png#gh-dark-mode-only)
![4M main figure](./assets/4M_main_fig_lightmode.png#gh-light-mode-only)

4M is a framework for training "any-to-any" foundation models, using tokenization and masking to scale to many diverse modalities. Models trained using 4M can perform a wide range of vision tasks, transfer well to unseen tasks and modalities, and are flexible and steerable multimodal generative models. We are releasing code and models for "4M: Massively Multimodal Masked Modeling" (here denoted 4M-7), as well as "4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities" (here denoted 4M-21).

## Table of contents
- [Usage](#usage)
    - [Installation](#installation)
    - [Getting started](#getting-started)
    - [Data](#data)
    - [Tokenization](#tokenization)
    - [4M Training](#4m-training)
    - [Generation](#generation)
- [Model Zoo](#model-zoo)
    - [4M models](#4m-models)
    - [4M text-to-image specialist models](#4m-text-to-image-specialist-models)
    - [4M super-resolution models](#4m-super-resolution-models)
    - [Tokenizers](#tokenizers)
- [License](#license)
- [Citation](#citation)

## Usage

### Installation

1. Clone this repository and navigate to the root directory:
```
git clone https://github.com/apple/ml-4m
cd ml-4m
```

2. Create a new conda environment, then install the package and its dependencies:
```
conda create -n fourm python=3.9 -y
conda activate fourm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Verify that CUDA is available in PyTorch by running the following in a Python shell:
```
# Run in Python shell
import torch
print(torch.cuda.is_available())  # Should return True
```
If CUDA is not available, consider re-installing PyTorch following the [official installation instructions](https://pytorch.org/get-started/locally/). Likewise, if you want to install xFormers (optional, for faster tokenizers), follow [their README](https://github.com/facebookresearch/xformers) to ensure that the CUDA version is correct.

4. (Optional) Expose the new conda environment as a kernel to Jupyter notebooks:
```bash
pip install ipykernel
python -m ipykernel install --user --name fourm --display-name "4M (fourm)"
```

### Getting started

We provide a demo wrapper to quickly get started with using 4M models for RGB-to-all or {caption, bounding boxes}-to-all generation tasks.
For example, to generate all modalities from a given RGB input, call:

```python
from fourm.demo_4M_sampler import Demo4MSampler, img_from_url
sampler = Demo4MSampler(fm='EPFL-VILAB/4M-21_XL').cuda()
img = img_from_url('https://storage.googleapis.com/four_m_site/images/demo_rgb.png') # 1x3x224x224 ImageNet-standardized PyTorch Tensor
preds = sampler({'rgb@224': img.cuda()}, seed=None) 
sampler.plot_modalities(preds, save_path=None)
```

You should expect to see an output like the following:

![4M demo sampler output](./assets/4M_demo_sample_darkmode.jpg#gh-dark-mode-only)
![4M demo sampler output](./assets/4M_demo_sample_lightmode.jpg#gh-light-mode-only)

For performing caption-to-all generation, you can replace the sampler input by: `preds = sampler({'caption': 'A lake house with a boat in front [S_1]'})`.
For a list of available 4M models, please see the model zoo below, and see [README_GENERATION.md](README_GENERATION.md) for more instructions on generation.

### Data  

See [README_DATA.md](README_DATA.md) for instructions on how to prepare aligned multimodal datasets.

### Tokenization  

See [README_TOKENIZATION.md](README_TOKENIZATION.md) for instructions on how to train modality-specific tokenizers.

### 4M Training

See [README_TRAINING.md](README_TRAINING.md) for instructions on how to train 4M models.

### Generation

See [README_GENERATION.md](README_GENERATION.md) for instructions on how to use 4M models for inference / generation. We also provide a [generation notebook](notebooks/generation_4M-21.ipynb) that contains examples for 4M inference, specifically performing conditional image generation and common vision tasks (i.e. RGB-to-All).


## Model Zoo

We provide 4M and tokenizer checkpoints as [safetensors](https://huggingface.co/docs/safetensors/en/index), and also offer easy loading via [Hugging Face Hub](https://huggingface.co/docs/hub/index).

### 4M models

| Model   | # Mod. | Datasets | # Params | Config | Weights         |
| ------- | ------ | -------- | -------- | ------ | --------------- |
| 4M-B | 7 | CC12M | 198M | [Config](cfgs/default/4m/models/main/4m-b_mod7_500b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7_B_CC12M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7_B_CC12M) |
| 4M-B | 7 | COYO700M | 198M | [Config](cfgs/default/4m/models/main/4m-b_mod7_500b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7_B_COYO700M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7_B_COYO700M) |
| 4M-B | 21 | CC12M+COYO700M+C4 | 198M | [Config](cfgs/default/4m/models/main/4m-b_mod21_500b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-21_B/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-21_B) |
| 4M-L | 7 | CC12M | 705M | [Config](cfgs/default/4m/models/main/4m-l_mod7_500b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7_L_CC12M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7_L_CC12M) |
| 4M-L | 7 | COYO700M | 705M | [Config](cfgs/default/4m/models/main/4m-l_mod7_500b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7_L_COYO700M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7_L_COYO700M) |
| 4M-L | 21 | CC12M+COYO700M+C4 | 705M | [Config](cfgs/default/4m/models/main/4m-l_mod21_500b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-21_L/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-21_L) |
| 4M-XL | 7 | CC12M | 2.8B | [Config](cfgs/default/4m/models/main/4m-xl_mod7_500b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7_XL_CC12M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7_XL_CC12M) |
| 4M-XL | 7 | COYO700M | 2.8B | [Config](cfgs/default/4m/models/main/4m-xl_mod7_500b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7_XL_COYO700M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7_XL_COYO700M) |
| 4M-XL | 21 | CC12M+COYO700M+C4 | 2.8B | [Config](cfgs/default/4m/models/main/4m-xl_mod21_500b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-21_XL/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-21_XL) |

To load models from Hugging Face Hub:
```python
from fourm.models.fm import FM

fm7b_cc12m  = FM.from_pretrained('EPFL-VILAB/4M-7_B_CC12M')
fm7b_coyo   = FM.from_pretrained('EPFL-VILAB/4M-7_B_COYO700M')
fm21b       = FM.from_pretrained('EPFL-VILAB/4M-21_B')

fm7l_cc12m  = FM.from_pretrained('EPFL-VILAB/4M-7_L_CC12M')
fm7l_coyo   = FM.from_pretrained('EPFL-VILAB/4M-7_L_COYO700M')
fm21l       = FM.from_pretrained('EPFL-VILAB/4M-21_L')

fm7xl_cc12m = FM.from_pretrained('EPFL-VILAB/4M-7_XL_CC12M')
fm7xl_coyo  = FM.from_pretrained('EPFL-VILAB/4M-7_XL_COYO700M')
fm21xl      = FM.from_pretrained('EPFL-VILAB/4M-21_XL')
```

To load the checkpoints manually, first download the safetensors files from the above links and call:
```python
from fourm.utils import load_safetensors
from fourm.models.fm import FM

ckpt, config = load_safetensors('/path/to/checkpoint.safetensors')
fm = FM(config=config)
fm.load_state_dict(ckpt)
```

### 4M text-to-image specialist models

These models were initialized with the standard 4M-7 CC12M models, but continued training with a modality mixture heavily biased towards text inputs. They are still able to perform all other tasks, but perform better at text-to-image generation compared to the non-finetuned models.

| Model   | # Mod. | Datasets | # Params | Config | Weights         |
| ------- | ------ | -------- | -------- | ------ | --------------- |
| 4M-T2I-B | 7 | CC12M | 198M | [Config](cfgs/default/4m/models/specialized/4m-b_mod7_500b--spec_text2im_100b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7-T2I_B_CC12M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7-T2I_B_CC12M) |
| 4M-T2I-L | 7 | CC12M | 705M | [Config](cfgs/default/4m/models/specialized/4m-l_mod7_500b--spec_text2im_100b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7-T2I_L_CC12M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7-T2I_L_CC12M) |
| 4M-T2I-XL | 7 | CC12M | 2.8B | [Config](cfgs/default/4m/models/specialized/4m-xl_mod7_500b--spec_text2im_100b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7-T2I_XL_CC12M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7-T2I_XL_CC12M) |

To load models from Hugging Face Hub:
```python
from fourm.models.fm import FM

fm7b_t2i_cc12m  = FM.from_pretrained('EPFL-VILAB/4M-7-T2I_B_CC12M')
fm7l_t2i_cc12m  = FM.from_pretrained('EPFL-VILAB/4M-7-T2I_L_CC12M')
fm7xl_t2i_cc12m  = FM.from_pretrained('EPFL-VILAB/4M-7-T2I_XL_CC12M')
```

Loading manually from checkpoints is performed in the same way as above for the base 4M models.

### 4M super-resolution models

| Model   | # Mod. | Datasets | # Params | Config | Weights         |
| ------- | ------ | -------- | -------- | ------ | --------------- |
| 4M-SR-L | 7 | CC12M | 198M | [Config](cfgs/default/4m/models/superres/4m-l_mod7_500b--sr_448_100b.yaml) | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M-7-SR_L_CC12M/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M-7-SR_L_CC12M) |

To load models from Hugging Face Hub:
```python
from fourm.models.fm import FM

fm7l_sr_cc12m  = FM.from_pretrained('EPFL-VILAB/4M-7-SR_L_CC12M')
```

Loading manually from checkpoints is performed in the same way as above for the base 4M models.

### Tokenizers

| Modality                   | Resolution | Number of tokens | Codebook size   | Diffusion decoder | Weights |
|----------------------------|------------|------------------|-----------------|-------------------|---------|
| RGB                        | 224-448    | 196-784          | 16k             | ✓                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_rgb_16k_224-448/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_rgb_16k_224-448) |
| Depth                      | 224-448    | 196-784          |  8k             | ✓                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_depth_8k_224-448/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_depth_8k_224-448) |
| Normals                    | 224-448    | 196-784          |  8k             | ✓                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_normal_8k_224-448/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_normal_8k_224-448) |
| Edges (Canny, SAM)         | 224-512    | 196-1024         |  8k             | ✓                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_edge_8k_224-512/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_edge_8k_224-512) |
| COCO semantic segmentation | 224-448    | 196-784          |  4k             | ✗                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_semseg_4k_224-448/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_semseg_4k_224-448) |
| CLIP-B/16                  | 224-448    | 196-784          |  8k             | ✗                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_CLIP-B16_8k_224-448/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_CLIP-B16_8k_224-448) |
| DINOv2-B/14                | 224-448    | 256-1024         |  8k             | ✗                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448) |
| DINOv2-B/14 (global)       | 224        | 16               |  8k             | ✗                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_DINOv2-B14-global_8k_16_224/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_DINOv2-B14-global_8k_16_224) |
| ImageBind-H/14             | 224-448    | 256-1024         |  8k             | ✗                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_ImageBind-H14_8k_224-448/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_ImageBind-H14_8k_224-448) |
| ImageBind-H/14 (global)    | 224        | 16               | 8k              | ✗                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_ImageBind-H14-global_8k_16_224/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_ImageBind-H14-global_8k_16_224) |
| SAM instances              | -          | 64               | 1k             | ✗                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_sam-instance_1k_64/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_sam-instance_1k_64) |
| 3D Human poses             | -          | 8                | 1k             | ✗                 | [Checkpoint](https://huggingface.co/EPFL-VILAB/4M_tokenizers_human-poses_1k_8/resolve/main/model.safetensors) / [HF Hub](https://huggingface.co/EPFL-VILAB/4M_tokenizers_human-poses_1k_8) |

To load models from Hugging Face Hub:
```python
from fourm.vq.vqvae import VQVAE, DiVAE

# 4M-7 modalities
tok_rgb = DiVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_rgb_16k_224-448')
tok_depth = DiVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_depth_8k_224-448')
tok_normal = DiVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_normal_8k_224-448')
tok_semseg = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_semseg_4k_224-448')
tok_clip = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_CLIP-B16_8k_224-448')

# 4M-21 modalities
tok_edge = DiVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_edge_8k_224-512')
tok_dinov2 = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448')
tok_dinov2_global = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_DINOv2-B14-global_8k_16_224')
tok_imagebind = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_ImageBind-H14_8k_224-448')
tok_imagebind_global = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_ImageBind-H14-global_8k_16_224')
sam_instance = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_sam-instance_1k_64')
human_poses = VQVAE.from_pretrained('EPFL-VILAB/4M_tokenizers_human-poses_1k_8')
```

To load the checkpoints manually, first download the safetensors files from the above links and call:
```python
from fourm.utils import load_safetensors
from fourm.vq.vqvae import VQVAE, DiVAE

ckpt, config = load_safetensors('/path/to/checkpoint.safetensors')
tok = VQVAE(config=config) # Or DiVAE for models with a diffusion decoder
tok.load_state_dict(ckpt)
```


## License

The code in this repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

The model weights in this repository are released under the Sample Code license as found in the [LICENSE_WEIGHTS](LICENSE_WEIGHTS) file.

## Citation

If you find this repository helpful, please consider citing our work:
```
@inproceedings{4m,
    title={{4M}: Massively Multimodal Masked Modeling},
    author={David Mizrahi and Roman Bachmann and O{\u{g}}uzhan Fatih Kar and Teresa Yeo and Mingfei Gao and Afshin Dehghan and Amir Zamir},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}

@article{4m21,
    title={{4M-21}: An Any-to-Any Vision Model for Tens of Tasks and Modalities},
    author={Roman Bachmann and O{\u{g}}uzhan Fatih Kar and David Mizrahi and Ali Garjani and Mingfei Gao and David Griffiths and Jiaming Hu and Afshin Dehghan and Amir Zamir},
    journal={arXiv 2024},
    year={2024},
}
```
