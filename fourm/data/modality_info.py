# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial

import fourm.utils.data_constants as data_constants
from fourm.data.modality_transforms import (CaptionTransform, DepthTransform,
                                      DetectionTransform, MaskTransform,
                                      NormalTransform, RGBTransform,
                                      SemsegTransform, TokTransform,
                                      CaptionEmbTransform, MetadataTransform,
                                      HumanPoseTransform, ColorPaletteTransform,
                                      SAMInstanceTokTransform, SAMInstanceTransform)
from fourm.models.decoder_embeddings import (ImageTokenDecoderEmbedding,
                                   SequenceDecoderEmbedding)
from fourm.models.encoder_embeddings import (ImageEncoderEmbedding,
                                   ImageTokenEncoderEmbedding,
                                   SequenceEncoderEmbedding,
                                   SequenceEmbEncoderEmbedding)
from fourm.utils import generate_uint15_hash

MODALITY_INFO = {
    # 4M-7 modalities
    'rgb@224': {
        'input_size': 224,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=3),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('rgb@224'),
        'path': 'rgb',
    },
    'rgb': { # used for tokenizer training
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('rgb'),
        'path': 'rgb',
    },
    'caption': {
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'min_tokens': 0,
        'max_tokens': 256,
        'type': 'seq',
        'id': generate_uint15_hash('caption'),
    },
    'det': { 
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'min_tokens': 0,
        'max_tokens': 256,
        'type': 'seq',
        'id': generate_uint15_hash('det'),
    },
    'tok_rgb@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 16384,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=16384),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=16384),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_rgb@224'),
        'pretokenized': True,
    },
    'tok_depth@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_depth@224'),
        'pretokenized': True,
    },
    'depth': { # used for tokenizer training
        'type': 'img',
        'num_channels': 1,
        'id': generate_uint15_hash('depth'),
    },
    'tok_normal@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_normal@224'),
        'pretokenized': True,
    },
    'normal': { # used for tokenizer training
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('normal'),
    },
    'tok_semseg@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 4096,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=4096),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=4096),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_semseg@224'),
        'pretokenized': True,
    },
    'semseg_coco': { # used for tokenizer training
        'type': 'img', 
        'num_channels': 64,
        'num_labels': data_constants.COCO_SEMSEG_NUM_CLASSES,
        'id': generate_uint15_hash('semseg_coco'),
    },
    'tok_clip@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_clip@224'),
        'pretokenized': True,
    },
    'CLIP-B16': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 512,
        'id': generate_uint15_hash('CLIP-B16'),
    },

    # 4M-21 modalities
    't5_caption': {
        'encoder_embedding': partial(SequenceEmbEncoderEmbedding, max_length=77, padding_idx=0),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': 77,
        'type': 'seq_emb',
        'id': generate_uint15_hash('t5_caption'),
    },
    'metadata': { 
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=40, padding_idx=0, sincos_pos_emb=True),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=40, padding_idx=0, sincos_pos_emb=True),
        'min_tokens': 0,
        'max_tokens': 40, # At most 2x19=38 for 19 metadata types, +1 for EOS, +1 for sentinel
        'type': 'seq',
        'id': generate_uint15_hash('metadata'),
        'shared_vocab': ['caption'],
        'path': 'metadata',
    },
    'human_poses': { 
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=263, padding_idx=0, sincos_pos_emb=True),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=263, padding_idx=0, sincos_pos_emb=True),
        'min_tokens': 0,
        'max_tokens': 275, #7*39+1 EOS+1 S_1#263, #261 in one of the models, or 263 to have EOS #261+1+1 #238,
        'type': 'seq',
        'num_channels': 207, # for tokenization training, only the pose part is needed
        'id': generate_uint15_hash('human_poses'),
        'shared_vocab': ['caption'],
    },
    'color_palette': { 
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=23, padding_idx=0, sincos_pos_emb=True),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=23, padding_idx=0, sincos_pos_emb=True),
        'min_tokens': 0,
        'max_tokens': 23, #7x3=21 for 7 colors, +1 for EOS, +1 for sentinel
        'type': 'seq',
        'id': generate_uint15_hash('color_palette'),
        'shared_vocab': ['caption'],
        'path': 'color_palette',
    },
    'sam_mask': {
        'encoder_embedding': None,
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': 64,
        'type': 'img',
        'num_channels': 1,
        'id': generate_uint15_hash('sam_mask'),
    },
    'sam_instance': {
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=290, padding_idx=0, sincos_pos_emb=True),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=290, padding_idx=0, sincos_pos_emb=True),
        'min_tokens': 0,
        'max_tokens': 290,
        'type': 'seq',
        'id': generate_uint15_hash('sam_instance'),
        'shared_vocab': ['caption'],
        'pretokenized': True,
    },
    'tok_canny_edge@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_canny_edge@224'),
        'pretokenized': True,
    },
    'canny_edge': { # used for tokenizer training
        'type': 'img',
        'num_channels': 1,
        'id': generate_uint15_hash('canny_edge'),
    },
    'tok_sam_edge@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_sam_edge@224'),
        'pretokenized': True,
    },
    'tok_dinov2@224': {
        'input_size': 224,
        'patch_size': 14,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 256
        'type': 'img',
        'id': generate_uint15_hash('tok_dinov2@224'),
        'pretokenized': True,
    },
    'DINOv2-B14': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 768,
        'id': generate_uint15_hash('DINOv2-B14'),
    },
    'tok_imagebind@224': {
        'input_size': 224,
        'patch_size': 14,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 256
        'type': 'img',
        'id': generate_uint15_hash('tok_imagebind@224'),
        'pretokenized': True,
    },
    'ImageBind-H14': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 1280,
        'id': generate_uint15_hash('ImageBind-H14'),
    },
    'tok_dinov2_global': {
        'vocab_size': 8192,
        'patch_size': 56,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192, sincos_pos_emb=False),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192, sincos_pos_emb=False),
        'min_tokens': 0,
        'max_tokens': 16,
        'type': 'img',
        'id': generate_uint15_hash('tok_dinov2_global'),
        'pretokenized': True,
    },
    'DINOv2-B14-global': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 768,
        'id': generate_uint15_hash('DINOv2-B14-global'),
    },
    'tok_imagebind_global': {
        'vocab_size': 8192,
        'patch_size': 56,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192, sincos_pos_emb=False),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192, sincos_pos_emb=False),
        'min_tokens': 0,
        'max_tokens': 16,
        'type': 'img',
        'id': generate_uint15_hash('tok_imagebind_global'),
        'pretokenized': True,
    },
    'ImageBind-H14-global': { # used for tokenizer training
        'type': 'feature_map',
        'num_channels': 1280,
        'id': generate_uint15_hash('ImageBind-H14-global'),
    },

    ### 224->448 super resolution modalities
    'rgb@448': {
        'input_size': 448,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=3),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('rgb@448'),
        'path': 'rgb',
    },
    'tok_rgb@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 16384,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=16384),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=16384),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_rgb@448'),
        'pretokenized': True,
    },
    'tok_depth@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_depth@448'),
        'pretokenized': True,
    },
    'tok_normal@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_normal@448'),
        'pretokenized': True,
    },
    'tok_semseg@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 4096,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=4096),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=4096),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_semseg@448'),
        'pretokenized': True,
    },
    'tok_clip@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_clip@448'),
        'pretokenized': True,
    },
}

# Note: @res suffix is ignored for modality transforms
MODALITY_TRANSFORMS = {
    # 4M-7 modalities
    'rgb': RGBTransform(imagenet_default_mean_and_std=True),
    'caption': CaptionTransform(aligned_captions=True),
    'det': DetectionTransform(det_threshold=0.6, det_max_instances=None, bbox_order='dist_to_orig', coord_bins=1000, min_visibility=0.0),
    'tok_rgb': TokTransform(),
    'tok_depth': TokTransform(),
    'tok_normal': TokTransform(),
    'tok_semseg': TokTransform(),
    'tok_clip': TokTransform(),
    # 4M-21 modalities
    't5_caption': CaptionEmbTransform(),
    'metadata': MetadataTransform(special_vmin=0, special_vmax=999, shuffle=True, random_trunc=False, return_chunks=True),
    'human_poses': HumanPoseTransform(coord_bins=1000),
    'color_palette': ColorPaletteTransform(coord_bins=1000),
    'sam_instance': SAMInstanceTokTransform(image_size=224, points_per_side=7, point_order='random'),
    'tok_canny_edge': TokTransform(),
    'tok_sam_edge': TokTransform(),
    'tok_dinov2': TokTransform(),
    'tok_imagebind': TokTransform(),
    'tok_dinov2_global': TokTransform(),
    'tok_imagebind_global': TokTransform(),
    # Other
    'mask_valid': MaskTransform(mask_pool_size=1),
}

MODALITY_TRANSFORMS_DIVAE = {
    'rgb': RGBTransform(imagenet_default_mean_and_std=False),
    'depth': DepthTransform(standardize_depth=True),
    'normal': NormalTransform(standardize_surface_normals=False),
    'mask_valid': MaskTransform(mask_pool_size=1),
    'semseg_coco': SemsegTransform(shift_idx_by_one=True),
    'canny_edge': RGBTransform(imagenet_default_mean_and_std=False),
    'human_poses': HumanPoseTransform(coord_bins=1000, only_pose=True),
    'sam_mask': SAMInstanceTransform(mask_size=64, max_instance_n=1),
}

MODALITY_TRANSFORMS_VQCONTROLNET = {
    'rgb': RGBTransform(imagenet_default_mean_and_std=False),
    'mask_valid': MaskTransform(mask_pool_size=1),
    'caption': CaptionTransform(aligned_captions=True),
}
