# Data

This README provides guidelines on how to structure and prepare aligned multimodal training datasets.

## Dataset format and structure

### Modified WebDataset format (for larger datasets and/or cloud storage)

We recommend organizing training data using a modified version of the [WebDataset format](https://github.com/webdataset/webdataset). With this format, the dataset is split into tarfiles, with each tarfile containing 1'000 to 10'000 samples from one modality (e.g., RGB, caption, depth, etc.). This data can be stored either on a local disk or in common cloud object stores like S3. Storing the data as tarfiles reduces the number of object read requests when streaming directly from cloud buckets. The data is organized as such:

```
root/modality_a/shard-00000.tar
root/modality_a/shard-00001.tar
root/modality_a/shard-00002.tar

root/modality_b/shard-00000.tar
root/modality_b/shard-00001.tar
root/modality_b/shard-00002.tar
```

Here, `modality_a` and `modality_b` are placeholders for the names of the modalities (e.g., `rgb`, `caption`, or more specific modality names for dataset versioning).

Each tarfile expands into individual files with arbitrary names but the same extension, such as:
```
xxx.ext
xxy.ext
xxz.ext
```

The file extension varies depending on the modality (e.g., `.jpg` for RGB images, `.txt` or `.json` for captions, `.npy` for pre-computed tokens, etc.).

To load aligned samples from all modalities, make sure that **filenames are identical for all modalities, except for the modality name and file extensions**. Shards should also be **ordered numerically** (as shown above) to support brace-expand notation. New modalities can be easily added by creating a new folder with tarfiles in the same directory. Existing modalities can also be modified by updating their specific tarfiles or creating new ones.

### Simple hierarchical format (for smaller local datasets)

For smaller datasets that can be stored locally, we also support a simpler hierarchical structure. This is convenient for datasets like validation sets or transfer datasets. The data structure is as follows:

```
root/modality_a/folder_x/xxx.ext
root/modality_a/folder_y/xxy.ext
root/modality_a/folder_z/xxz.ext

root/modality_b/folder_x/xxx.ext
root/modality_b/folder_y/xxy.ext
root/modality_b/folder_z/xxz.ext
```

The folder and file names can be arbitrary as long as they are aligned across modalities.

## Datasets

We use the following datasets to train and/or evaluate 4M models.
For pre-training:
- [**Conceptual Captions 12M (CC12M)**](https://github.com/google-research-datasets/conceptual-12m)

For transfers and evaluations:
- [**ImageNet**](https://www.image-net.org/)
- [**COCO**](https://cocodataset.org)
- [**ADE20K**](http://sceneparsing.csail.mit.edu/)
- [**Hypersim**](https://github.com/apple/ml-hypersim)
- [**NYUv2**](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [**Taskonomy**](https://github.com/StanfordVL/taskonomy/tree/master/data)


Please refer to their respective pages for instructions on how to download them and license information.

## Pseudo labeling

Starting from text-image pairs, we use pseudo labeling to create an aligned multimodal dataset across all training modalities. For this purpose, we use the off-the-shelf networks listed in the table below. Please refer to their respective pages for inference instructions and license information.

| Modality              | Model                                | Homepage                                                                                             |
|-----------------------|--------------------------------------|------------------------------------------------------------------------------------------------------|
| Depth                 | Omnidata DPT-B-Hybrid (v2)           | [link](https://docs.omnidata.vision/pretrained.html#Pretrained-Models)                               |
| Surface normals       | Omnidata DPT-B-Hybrid (v2)           | [link](https://docs.omnidata.vision/pretrained.html#Pretrained-Models)                               |
| Semantic segmentation | Mask2Former Swin-B                   | [link](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md#panoptic-segmentation) |
| Bounding boxes        | ViTDet ViT-H with Cascade Mask-RCNN  | [link](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet#cascade-mask-r-cnn)  |
| CLIP features         | CLIP ViT-B/16                        | [link](https://github.com/OpenAI/CLIP#clip)                                                          |
| DINOv2 features       | DINOv2 ViT-B/14                      | [link](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models)              |
| ImageBind features    | ImageBind ViT-H/14                   | [link](https://github.com/facebookresearch/ImageBind?tab=readme-ov-file#imagebind-model)             |
| SAM instances         | SAM ViT-H                            | [link](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)    |
| 3D human poses & shape| HMR2.0                               | [link](https://github.com/shubham-goel/4D-Humans)                                                    |
| Color palette         | PyPalette                            | [link](https://github.com/adamgrieger/pypalette)                                                     |

## Pre-tokenization

During training, all modalities are maps to sets or sequences of discrete tokens using modality-specific tokenizers. Please refer to [README_TOKENIZATION.md](README_TOKENIZATION.md) for more information. To avoid dataloading and tokenization from becoming a training bottleneck, we instead pre-compute the tokens of all image-like modalities once before training (i.e. pre-tokenization), and then directly load the tokens.

To pre-tokenize any modality, run the provided `save_vq_tokens.py` file with the appropriate arguments.

:information_source: For non-square images or if `--n_crops` is > 1, pre-tokenization requires cropping the original image. Therefore, to ensure that the tokens from all modalities are aligned, we automatically create a `crop_settings` directory with the crop information for all samples the first time that a dataset is tokenized. This information is then used when tokenizing the same dataset with a different modality.
