# Tokenization
Tokenization is at the core of the 4M framework, allowing us to cast a large and diverse set of modalities ranging from images, feature maps, structured data, and sequences into a unified representation space. Before training 4M models, we train/fit tokenizers on each modality and pre-compute the tokens on the entire dataset to make 4M training as efficient as possible.

This README provides instructions on training tokenizers, and where to find the most relevant files. Please see the [4M papers](https://4m.epfl.ch/) and their supplementary material for more background on our tokenizers, and [README_DATA.md](README_DATA.md) for more information on preparing the training data.

## Structure

#### `fourm/vq/`
- All code related to the tokenizers (vq = **v**ector **q**uantization)
- Some important files and directories:
    - `vq/models/`: Contains all the encoder and decoder architectures.
    - `vq/percept_losses/`: Contains different perceptual loss implementations for training VQ-VAEs.
    - `vq/quantizers/`: Contains different quantizer implementations.
    - `vq/scheduling/`: Contains diffusion schedules for training/inference and pipelines for inference.
    - `vq/vqvae.py`: Main file defining the standard VQ-VAE and diffusion-based VQ-VAE classes.
    - `vq/__init__.py`: Contains the `get_image_tokenizer` function used to autoload tokenizers with a given name.


#### Root directory
- `run_training_vqvae.py`: Main training script for training standard VQ-VAEs (with optional perceptual loss).
- `run_training_divae.py`: Main training script for training diffusion-based VQ-VAEs.
- `run_training_vqcontrolnet.py`: Main training script for training ControlNet detokenizers.


## General information

### Configs
The training scripts support both YAML config files and command-line arguments.

To modify training settings, either edit / add config files or provide additional command-line arguments.

:information_source: Config files arguments override default arguments, and command-line arguments override both default arguments and config arguments.

### Multi-resolution adaptation
Unlike ConvNets, ViT-based tokenizers won't work well at resolutions that are different from the training resolution. The training scripts support multi-resolution training, where each batch randomly samples a training resolution between a specified min and max. Because certain networks like UNets only support image sizes that are a multiple of a certain size, make sure to adapt `resolution_step` accordingly:

```yaml
input_size_min: 224
input_size_max: 512
resolution_step: 32
```

This feature is intended to be used as an adaptation step, after training the model at a fixed base resolution. The resulting tokenizers can then work at different resolutions, which allows us to train a super-resolution adaptation of 4M that maps tokens from 224x224 images to tokens from 448x448 images.

### Training a decoder with a frozen encoder
This is mostly relevant for training diffusion decoders on top of a frozen pre-trained VQ-VAE encoder, but our original 4M-7 RGB, depth, and surface normal tokenizers were trained with a diffusion decoder from scratch. Rather than performing the training end-to-end, we currently suggest first training a standard VQ-VAE to have more control over the learned representation, and then training a diffusion decoder afterwards. When training a diffusion decoder on top of a frozen VQ-VAE encoder, make sure to add the following lines to the config:

```yaml
full_ckpt: /path/to/checkpoint.pth
freeze_enc: True # Decoder can be trained from scratch or fine-tuned without the encoder
input_size_enc: 256 # Size of the encoder positional embeddings
```

### xFormers

:information_source: The ViT and UViT backbones, as well as Stable Diffusion (for VQ-ControlNet) will automatically switch to memory-efficient attention if you (optionally) install [xFormers](https://github.com/facebookresearch/xformers). Depending on the image size, this has the potential to significantly cut down on memory consumption and training time.

## VQ-VAE

We support training vector quantized autoencoders (VQ-VAEs) with a quantizer bottleneck and standard reconstruction objective with optional perceptual loss (we currently do not support GAN losses like VQ-GAN). Consider training a standard VQ-VAE when
1) you want control over semantic properties of the tokens by optimizing for optional perceptual losses,
1) training on a modality like semantic segmentation that does not benefit as much from a powerful generator as the decoder, or
1) quantizing neural network feature maps, global feature embeddings, or other modalities such as human poses.

VQ-VAE training configs can be found in [cfgs/default/tokenization/vqvae](cfgs/default/tokenization/vqvae). For each example we give a base resolution training config, and a multi-resolution adaptation config. We also provide an example config for training a VQ-VAE model on RGB using a CLIP perceptual loss, which is intended as a pre-training step to get a better encoder.

To train a VQ-VAE on a 8 GPU node, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_training_vqvae.py \
--config cfgs/default/tokenization/vqvae/<config>.yaml \
--data_path /path/to/dataset/train/ \
--eval_data_path /path/to/dataset/val/
```


## Diffusion-decoder VQ-VAE (DiVAE)

Diffusion-decoder VQ-VAE are vector quantized autoencoders with a standard quantizer bottleneck but using a diffusion model as the decoder. The diffusion model takes as usual a noised image as input, but is now additionally conditioned on the VQ-VAE encoder latents (after tokenization). Its objective is to either predict the clean image, the noise, or the velocity. At test time, the decoder generates an image conditioned on the encoded latents by performing the 1000 diffusion steps (or fewer when using other diffusion schedulers like DDIM).

Training diffusion-based VQ-VAE decoders is mostly intended for modalities like RGB which are information dense, and where a good generator is needed to "in-paint" realistic texture, since 16x16 tokens can only hold so much information. They can be trained end-to-end from scratch, or with a two-stage approach of first training a standard VQ-VAE with optional perceptual losses, followed by freezing the encoder and training a diffusion decoder on top.

DiVAE configs can be found in [cfgs/default/tokenization/divae](cfgs/default/tokenization/divae).

To train a Diffusion VQ-VAE on a 8 GPU node, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_training_divae.py \
--config cfgs/default/tokenization/divae/<config>.yaml \
--data_path /path/to/dataset/train/ \
--eval_data_path /path/to/dataset/val/
```


## Diffusion-decoder VQ-VAE with pre-trained diffusion model (ControlNet)

Diffusion decoders can be expensive to train, and we've had success with initializing them with pre-trained diffusion models, specifically Stable Diffusion. This can be seen as training a ControlNet on top of tokens from a frozen VQ-VAE encoder, and is significantly cheaper than training a diffusion decoder from scratch, but comes at the cost of a heavier decoder.

VQ-ControlNet configs can be found in [cfgs/default/tokenization/vqcontrolnet](cfgs/default/tokenization/vqcontrolnet).

To train a ControlNet detokenizer on a 8 GPU node, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_training_vqcontrolnet.py \
--config cfgs/default/tokenization/vqcontrolnet/<config>.yaml \
--data_path /path/to/dataset/train/ \
--eval_data_path /path/to/dataset/val/
```


## Text tokenizer

For sequence modalities that can be represented as text (e.g. captions or bounding boxes), we train a standard WordPiece tokenizer using the HuggingFace Tokenizers library. The text tokenizer is shared across these modalities, and can be found under [`fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json`](fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json). Please refer to [`train_wordpiece_tokenizer.py`](train_wordpiece_tokenizer.py) for more implementation details and instructions on how to train a similar tokenizer on your own datasets.


## Tokenizer checkpoints

Please see the download links and Hugging Face Hub examples in the [README.md](README.md). To load `.pth` checkpoints manually, please see the `get_image_tokenizer` function in `vq/__init__.py`. By default, the function expects the tokenizer checkpoints to be placed in the repository root under `./tokenizer_ckpts`. We recommend loading using Hugging Face Hub or the provided safetensors.


## Pre-computing tokens for efficient 4M training

When training 4M models, we want to reduce data loading and preparation time as much as possible, hence we pre-compute and store the tokenized versions of all modalities and avoid running the tokenizer encoders during 4M training. Once the tokenizer networks are trained, please follow the instructions in [README_DATA.md](README_DATA.md) on how to pre-compute the tokens.
