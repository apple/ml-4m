# Generation

We provide information and example scripts on how to perform multimodal chained generation using 4M models. Please see the 4M paper and its appendix for more background on the different ways 4M can be used at inference time and the relevant generation schemes. For inference on local dataset, please see the documentation in [README_DATA.md](README_DATA.md) regarding "Simple hierarchical format".


## Structure

#### Important files and directories:

#### `fm/`
- `fm/generate.py`: Contains most of the generation logic for different generation schedules.

#### `utils/`
- `utils/generation.py`: Contains helper functions for building generation schedules.
- `utils/plotting_utils.py`: Contains helper functions for decoding and plotting different modalities.


#### Scripts and notebooks:
- `run_generation.py`: Script that automatically performs chained X→Y→etc... generation on a given dataset.
- `notebooks/`: Jupyter notebooks that gives some examples for performing generation with 4M-7 and 4M-21 models.


#### Configs:

Configs for the `run_generation.py` script can be found in:

#### `cfgs/4m/generation/`
- `data/`: Contains data configs, e.g. for performing generation on CC12M, COCO, Parti prompts, etc.
- `models/`: Contains model configs that specify base (and optional super resolution) models, as well as their associated tokenizers. The script will load all models and tokenizers specified in this config, so if you are running out of VRAM, uncomment the unused ones.
- `settings_base/`: Contains configs that specify the (chained) generation schedules at the base resolution.
- `settings_sr/`: Contains configs that specify the (chained) generation schedules when performing super resolution, conditioning on the generated base resolution modalities.


## Generation parameters

To perform chained generation, i.e. generating modalities one-by-one and conditioning each subsequent modalities on the previously generated ones, we provide several utility functions. `utils/generation.py` contains a function to build a _generation schedule_ to specify which modality to generate with which settings, as well as a Sampler that wraps a trained 4M model and performs the generation using various generation schemes when provided an input and generation schedule.

Inputs to the model are given as dictionaries of modalities, where each modality in turn is a dictionary that specifies a set of tokens that can contain data to input to the model or placeholder values to fill in during generation, an input mask that specifies which parts of the given tokens are used as input to the model, and a target mask that specifies which parts of the tokens are to be predicted. See the provided generation notebook `notebooks/generation.ipynb` for concrete examples.

When building a generation schedule, we expose several hyperparameters to enable fine-grained control over the generation procedure. Except for `cfg_grow_conditioning`, `top_p`, and `top_k`, all other parameters can either be specified for **all** target modalities at the same time, or specified per target modality by separating the settings by hyphens in the configs and as lists of ints/strings/floats when directly using the `build_chained_generation_schedules` function. Let's say we generate the chain caption→CLIP→RGB, i.e. `cond_domains: caption` and `target_domains: tok_clip-tok_rgb`. If we want both CLIP and RGB to be generated with the same temperature, we could just specify `temps: 3.0`, but if we want more control per generated modality, we could specify `temps: 3.0-0.5`.

#### Condition & target domains
- `cond_domains`: Domains that are used as conditioning.
- `target_domains`: Domains that are predicted. The order given is the order in which the domains are predicted one-by-one.

#### Generation settings
- `tokens_per_target`: Number of tokens to decode per target modality. 196 for images at the base resolution, 256 for captions and bounding boxes.
- `autoregression_schemes`: Generation scheme for each target modality. `maskgit` or `roar` for image-like modalities, and `autoregressive` for sequence modalities.
- `decoding_steps`: Integers that specify in how many steps each target modality should be decoded. For example, if predicting an image of 196 tokens in 49 steps with a linear decoding schedule, each step decodes 4 tokens, or when specifying 1 step, the entire image is predicted in a single forward pass. Depending on the input and target modalities, generation requires fewer or more steps. For example, RGB→normal can be predicted well in a single step, but tasks like caption→RGB require multiple generation steps.
- `token_decoding_schedules`: Type of decoding schedule that specifies how many tokens are being decoded at which decoding step (if applicable). `cosine` starts and ends with a small number of decoded tokens, but decodes many in the middle of the schedule. `linear` decodes the same number of tokens each time step.

#### Temperature settings
- `temps`: Sampling temperatures for each target modality.
- `temp_schedules`: Temperature sampling schedules for each target modality. `constant` keeps the temperature constant for the duration of decoding, `linear` linearly decays the temperature from the indicated temperature down to zero, and `onex:{min_t}:{power}` decays the temperature proportional to x^-power from the starting temperature until the minimum temperature min_t.

#### Classifier-free guidance settings
- `cfg_scales`: Classifier-free guidance scales for each target modality. A value of 1.0 means no guidance is performed. Values > 1.0 perform positive guidance, values between 0.0 and 1.0 perform weak guidance, 0.0 is equal to an unconditional case, and lower values perform negative guidance.
- `cfg_schedules`: Only the `constant` schedule is implemented at the moment.
- `cfg_grow_conditioning`: True or False. If True, each completed modality is added to the classifier-free guidance conditioning.

#### Top-k & top-p sampling settings
- `top_p`: When top_p > 0.0, keep only the top tokens with cumulative probability >= top_p (a.k.a. nucleus filtering).
- `top_k`: When top_k > 0, keep only the top k tokens with highest probability (a.k.a. top-k filtering).


## Generation tips

Performing generation with 4M can be complex due to the large range of possibilities that come with all the ways chained generation can be performed, the different available generation schedules and their hyperparameters, etc. Here are some tips and best practices we found so far, but feel free to experiment!

- When generating information dense modalities like RGB from more abstract modalities like text, it can be beneficial to break down the generation process into one or more intermediate steps using chained generation. For example, we've had success in first generating CLIP tokens as an intermediate modality, and then conditioning the RGB generation on both the caption and the generated CLIP tokens.
- When generating image-like modalities, we provide two different schedules, MaskGIT and Random Order Auto Regression (ROAR). We found that MaskGIT-generated images tend to be "simpler" but easier to control, while the ROAR-generated images are more diverse.
- Depending on the chained generation schedule, make sure your temperature when sampling the first few tokens is sufficiently high and decays over the generation schedule. Make sure to play around with the supported temperature schedules. Most of the rough image content is decided during the early stages of sampling tokens / modalities.
- When performing super resolution, high classifier-free guidance and low temperature values can lead to it producing somewhat blurry results.
- Play around with the top-p and top-k parameters. For image generation, we usually set top-p to around 0.8.
- Classifier-free guidance can have a large impact on the generation fidelity, but is most important for input/output pairs that did not have clean aligned training data like images and captions. We found that increasing the guidance scale slightly when doing RGB→X inference (e.g. surface normal or segmentation prediction) can, however, also improve how well the generated modality matches the given input.
- Multi-modal guidance can be an effective tool to balance the influence of different input modalities during the generation process.
- The generation samplers and decoding functions support batching for faster inference.
- The default generation script can only generate a limited number of SAM instances due to limits on the number of tokens the model can handle. To get a denser estimation of SAM instances use the `generate_sam_dense` method (as shown in `notebooks/generation_4M-21.ipynb`). The method performs multiple independent SAM instance predictions and aggregates them into one dense estimation.
 - Avoid using the output of `generate_sam_dense` as the condition for generation. The output can contain large number of tokens and using it as the conditioning input can create memory issues.

## Generation script usage

We provide an example script for performing X→Y→etc... generation from a provided dataset. The script works by specifying several config files regarding the base resolution model, optional super resolution model, used tokenizers, dataset, as well as base and super resolution generation parameters.

For example, assuming you have 8 GPUs, to perform text→CLIP→RGB generation using 4M-XL on the Parti prompts, followed by a super resolution step, run the following:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_generation.py -c cfgs/default/generation/models/4m-xl_mod7+sr_4m-l_mod7.yaml -dc cfgs/default/generation/data/parti_3x.yaml -gc cfgs/default/generation/settings_base/T2CR_roar49-25_cfg3_t6-0.5.yaml -src cfgs/default/generation/settings_sr/x2CR_mg8_cfg3_t1const.yaml
```
This generates and saves three variants for each prompt in the dataset. Before running this, make sure you either downloaded the 4M and tokenizer checkpoints and pointed the config entries to the right paths, or load the models via Hugging Face Hub.


## Generation notebooks

Please see the provided Jupyter notebooks in `notebooks/` for more examples on how to use 4M-7 and 4M-21 models for inference/generation. We recommend running it on an A100 GPU, with `xformers` installed.
