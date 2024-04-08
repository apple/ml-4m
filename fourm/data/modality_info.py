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
                                      SemsegTransform, TokTransform)
from fourm.models.decoder_embeddings import (ImageTokenDecoderEmbedding,
                                   SequenceDecoderEmbedding)
from fourm.models.encoder_embeddings import (ImageEncoderEmbedding,
                                   ImageTokenEncoderEmbedding,
                                   SequenceEncoderEmbedding)
from fourm.utils import generate_uint15_hash

MODALITY_INFO = {
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
    'rgb': RGBTransform(imagenet_default_mean_and_std=True),
    'mask_valid': MaskTransform(mask_pool_size=1),
    'caption': CaptionTransform(aligned_captions=True),
    'det': DetectionTransform(det_threshold=0.6, det_max_instances=None, bbox_order='dist_to_orig', coord_bins=1000, min_visibility=0.0),
    'tok_rgb': TokTransform(),
    'tok_depth': TokTransform(),
    'tok_normal': TokTransform(),
    'tok_semseg': TokTransform(),
    'tok_clip': TokTransform(),

}

MODALITY_TRANSFORMS_DIVAE = {
    'rgb': RGBTransform(imagenet_default_mean_and_std=False),
    'depth': DepthTransform(standardize_depth=True),
    'normal': NormalTransform(standardize_surface_normals=False),
    'mask_valid': MaskTransform(mask_pool_size=1),
    'semseg_coco': SemsegTransform(shift_idx_by_one=True),
}

MODALITY_TRANSFORMS_VQCONTROLNET = {
    'rgb': RGBTransform(imagenet_default_mean_and_std=False),
    'mask_valid': MaskTransform(mask_pool_size=1),
    'caption': CaptionTransform(aligned_captions=True),
}
