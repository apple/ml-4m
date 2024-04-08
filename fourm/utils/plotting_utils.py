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
import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
import textwrap
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from itertools import groupby

# For visualizing CLIP feature maps
from sklearn.decomposition import PCA

# Detectron2 for semantic segmentation visualizations
try:
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from detectron2.data import MetadataCatalog
    coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    USE_DETECTRON = True
except Exception as e:
    print(e)
    print("Detectron2 can be used for semseg visualizations. Please install detectron2 to use this feature, or plotting will fall back to matplotlib.")
    USE_DETECTRON = False

from fourm.data.modality_transforms import get_transform_key, get_transform_resolution
from fourm.utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, COCO_SEMSEG_NUM_CLASSES
from fourm.utils import denormalize, get_sentinel_to_id_mapping, merge_span_masking
from fourm.utils.generation import unbatch

def tensor_to_images(tensor):
    """
    Converts a (B C H W) tensor to numpy arrays.
    If B = 1, the tensor is unbatched and converted to a single image.
    If C = 1, the channel dimension is removed.

    Args:
        tensor (torch.Tensor): Tensor to convert to images.
    """
    B, C, H, W = tensor.shape
    if B == 1:
        img = rearrange(unbatch(tensor), "c h w -> h w c")
    else:
        img = rearrange(tensor, "b c h w -> b h w c")
    if C == 1:
        img = img[..., 0]
    return img.detach().cpu().numpy()

def pca_visualize(features, n_components=3):
    """
    Visualizes a feature map using PCA.

    Args:
        features (torch.Tensor): CxHxW feature map to visualize.
        n_components (int): Number of PCA components to use.
    """
    C, H, W = features.shape
    features_flat = rearrange(features.float(), 'c h w -> (h w) c').detach().cpu().numpy()
    pca = PCA(n_components=n_components)
    img_pca = rearrange(pca.fit_transform(features_flat), '(h w) c -> h w c', h=H, w=W)
    img_pca = (img_pca - img_pca.min()) / (img_pca.max() - img_pca.min())
    return img_pca

def np_squeeze(array, axis=0):
    """
    Squeeses a numpy array along a given axis if that axis is one-dimensional.
    Otherwise, it returns the same array.
    
    Args:
        array (numpy.ndarray): Array to squeeze.
        axis (int): Axis to squeeze.
    """
    if array.shape[axis] == 1:
        return np.squeeze(array, axis=axis)
    else:
        return array

def decode_input_rgb(mod_dict, key='rgb'):
    """
    Decodes (denormalizes) an RGB image from a model dictionary.

    Args:
        mod_dict (dict): Model output dictionary.
        key (str): Key of the RGB modality to decode.
    """
    img = denormalize(mod_dict[key]['tensor'])
    return tensor_to_images(img)
    
def decode_tok_rgb(mod_dict, tokenizers, key='tok_rgb', image_size=224, patch_size=16, t=25, verbose=False):
    """
    Decodes a sequence of RGB tokens from a model dictionary into an RGB image.

    Args:
        mod_dict (dict): Model output dictionary.
        tokenizers (dict): Dictionary of tokenizers.
        key (str): Key of the tokenized RGB modality to decode.
        image_size (int): Size of the image.
        patch_size (int): Size of the patches.
        t (int): Number of timesteps to decode using the tokenizer diffusion model (if applicable).
        verbose (bool): Whether to print the decoding progress.
    """
    img_tok = rearrange(mod_dict[key]['tensor'], "b (nh nw) -> b nh nw", nh=image_size//patch_size, nw=image_size//patch_size)
    rec = tokenizers[get_transform_key(key)].decode_tokens(img_tok, timesteps=t, image_size=image_size, verbose=verbose)
    rec = denormalize(rec, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).clamp(0, 1)
    return tensor_to_images(rec)

def decode_tok_rgb_controlnet(mod_dict, tokenizers, key='tok_rgb', image_size=224, patch_size=16, 
                              t=25, guidance_scale=2.5, cond_scale=0.8, verbose=False):
    """
    Decodes a sequence of RGB tokens from a model dictionary into an RGB image using a ControlNet.

    Args:
        mod_dict (dict): Model output dictionary.
        tokenizers (dict): Dictionary of tokenizers. Needs to contain the key 'controlnet'.
        key (str): Key of the tokenized RGB modality to decode.
        image_size (int): Size of the image.
        patch_size (int): Size of the patches.
        t (int): Number of timesteps to decode using the ControlNet.
        guidance_scale (float): Classifier-free guidance scale.
        cond_scale (float): ControlNet conditioning scale.
        verbose (bool): Whether to print the decoding progress.
    """
    img_tok = rearrange(mod_dict[key]['tensor'], "b (nh nw) -> b nh nw", nh=image_size//patch_size, nw=image_size//patch_size)
    rec = tokenizers['controlnet'].decode_tokens(
        img_tok, timesteps=t, guidance_scale=guidance_scale, cond_scale=cond_scale, verbose=verbose
    )
    rec = tokenizers['controlnet'].vae_decode(rec)
    rec = denormalize(rec, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).clamp(0, 1)
    return tensor_to_images(rec)

def decode_tok_normal(mod_dict, tokenizers, key='tok_normal', image_size=224, patch_size=16, t=25, verbose=False):
    """
    Decodes a sequence of surface normal tokens from a model dictionary into an RGB image.

    Args:
        mod_dict (dict): Model output dictionary.
        tokenizers (dict): Dictionary of tokenizers.
        key (str): Key of the tokenized normal modality to decode.
        image_size (int): Size of the image.
        patch_size (int): Size of the patches.
        t (int): Number of timesteps to decode using the tokenizer diffusion model (if applicable).
        verbose (bool): Whether to print the decoding progress.
    """
    img_tok = rearrange(mod_dict[key]['tensor'], "b (nh nw) -> b nh nw", nh=image_size//patch_size, nw=image_size//patch_size)
    rec = tokenizers[get_transform_key(key)].decode_tokens(img_tok, timesteps=t, image_size=image_size, verbose=verbose)
    rec = denormalize(rec, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).clamp(0, 1)
    return tensor_to_images(rec)


def decode_tok_depth(mod_dict, tokenizers, key='tok_depth', image_size=224, patch_size=16, t=25, verbose=False, cmap='turbo'):
    """
    Decodes a sequence of depth tokens from a model dictionary into an RGB image.

    Args:
        mod_dict (dict): Model output dictionary.
        tokenizers (dict): Dictionary of tokenizers.
        key (str): Key of the tokenized depth modality to decode.
        image_size (int): Size of the image.
        patch_size (int): Size of the patches.
        t (int): Number of timesteps to decode using the tokenizer diffusion model (if applicable).
        verbose (bool): Whether to print the decoding progress.
        cmap (str): Colormap to use for the depth image.
    """
    img_tok = rearrange(mod_dict[key]['tensor'], "b (nh nw) -> b nh nw", nh=image_size//patch_size, nw=image_size//patch_size)
    rec = tokenizers[get_transform_key(key)].decode_tokens(img_tok, timesteps=t, image_size=image_size, verbose=verbose)
    rec = rec.detach().cpu().numpy()[:,0]

    if cmap is None:
        return rec
    
    colormap = plt.get_cmap('turbo')
    imgs = []
    for img in rec:
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
        rgb_image = colormap(img_norm)[..., :3]
        imgs.append(rgb_image)

    rgb_image = np_squeeze(np.stack(imgs), axis=0)
    
    return rgb_image
    
def decode_tok_semseg(rgb_img, mod_dict, tokenizers, key='tok_semseg', image_size=224, patch_size=16, use_detectron=True, return_logits=False):
    """
    Decodes a sequence of semantic segmentation tokens from a model dictionary into an RGB image.

    Args:
        rgb_img (torch.Tensor): RGB image to overlay the semantic segmentation on.
        mod_dict (dict): Model output dictionary.
        tokenizers (dict): Dictionary of tokenizers.
        key (str): Key of the tokenized semantic segmentation modality to decode.
        image_size (int): Size of the image.
        patch_size (int): Size of the patches.
        use_detectron (bool): Uses detectron2's visualization for the semseg output.
    """
    tokens = mod_dict[key]['tensor']
    tokens = tokens.unsqueeze(0) if tokens.ndim == 1 else tokens
    img_tok = rearrange(tokens, "b (nh nw) -> b nh nw", nh=image_size//patch_size, nw=image_size//patch_size)
    rec = tokenizers[get_transform_key(key)].decode_tokens(img_tok.cuda()).detach().cpu()
    if return_logits:
        return rec
    semsegs = rec.argmax(1)
    B, H, W = semsegs.shape

    if not use_detectron:
        return semsegs if B > 1 else semsegs[0]
    else:
        rgb_imgs = [rgb_img] * B
        imgs = []
        for rgb, semseg in zip(rgb_imgs, semsegs):
            if USE_DETECTRON:
                v = Visualizer(255*rgb, coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
                img = v.draw_sem_seg((semseg-1).cpu()).get_image() / 255.0
            else:
                colormap = plt.get_cmap('viridis')
                img = colormap(semseg.cpu())[..., :3]
            imgs.append(img)
        imgs = np_squeeze(np.stack(imgs), axis=0)
        return imgs

def decode_tok_clip(mod_dict, tokenizers, key='tok_clip', image_size=224, patch_size=16):
    """
    Decodes a sequence of CLIP tokens from a model dictionary into an PCA representation.

    Args:
        mod_dict (dict): Model output dictionary.
        key (str): Key of the tokenized CLIP modality to decode.
        tokenizers (dict): Dictionary of tokenizers.
        image_size (int): Size of the image.
        patch_size (int): Size of the patches.
    """
    n_patches = image_size // patch_size
    img_tok = rearrange(mod_dict[key]['tensor'], "b (nh nw) -> b nh nw", nh=n_patches, nw=n_patches)
    rec = tokenizers[get_transform_key(key)].decode_tokens(img_tok)
    pca_viz = [pca_visualize(feat) for feat in rec]
    pca_viz = np_squeeze(np.stack(pca_viz), axis=0)
    return pca_viz


# metadata_transform = MetadataTransform(shuffle=False, random_trunc=False, return_chunks=False)

def _split_metadata_string(input_string):    
    result = []
    current_subseq = []
    
    for part in input_string.split():
        # If we encounter a "ymin" and there's already a subsequence being built, 
        # we add it to the result and start a new one
        if 'ymin' in part and current_subseq:
            result.append(current_subseq)
            current_subseq = []
        
        current_subseq.append(part)

    # Append any remaining subsequence to the result
    if current_subseq:
        result.append(current_subseq)
        
    return result

def decode_metadata(mod_dict, text_tokenizer, key='metadata'):
    """
    Decodes a sequence of metadata tokens into a dictionary of metadata.

    Args:
        mod_dict (dict): Model output dictionary.
        key (str): Key of the metadata modality to decode.
        text_tokenizer (tokenizers.Tokenizer): Text tokenizer.
    """
    decoded = decode_text(mod_dict, key, text_tokenizer)[2]
    all_decoded = decoded if isinstance(decoded, list) else [decoded]
    all_decoded = [d.replace(' [EOS]', '').replace(' [PAD]', '') for d in all_decoded]

    all_metadata = []

    for decoded in all_decoded:

        parts = _split_metadata_string(decoded)
        
        invalid_parts = []
        metadata_dict = {}
        
        for part in parts:
            
            # Check if part has been parsed correctly
            if len(part) != 2:
                invalid_parts.append(str(part))    
                continue
            metadata_id, metadata_value = part
            if (not metadata_id.startswith('ymin=') or 
                not metadata_value.startswith('xmin=') or 
                metadata_id not in metadata_transform.id_metadata_map):
                invalid_parts.append(str(part))
                
            # Parse metadata type and value
            metadata_type = metadata_transform.id_metadata_map[metadata_id]
            
            metadata_value = int(metadata_value.split('=')[1])
            
            if metadata_type in metadata_transform.image_dim_modalities:
                metadata_value *= metadata_transform.image_dim_bin_size
            elif metadata_type in metadata_transform.metadata_min_max_bins:
                vmin, vmax, bins = metadata_transform.metadata_min_max_bins[metadata_type]
                metadata_value = (vmax - vmin) * (metadata_value / bins) + vmin
            
            metadata_dict[metadata_type] = metadata_value
            
        metadata_dict = {k: metadata_dict[k] for k in metadata_transform.metadata_id_map if k in metadata_dict}
        all_metadata.append(metadata_dict)

    all_metadata = all_metadata[0] if len(all_metadata) == 1 else all_metadata
    
    return all_metadata

def decode_text(mod_dict, key, text_tokenizer):
    """
    Decodes a text sequence from a model dictionary.

    Args:
        mod_dict (dict): Model output dictionary.
        key (str): Key of the text modality to decode.
        text_tokenizer (tokenizers.Tokenizer): Text tokenizer.
    """
    input_texts, target_texts, merged_texts = [], [], []

    sentinel_ids = set(get_sentinel_to_id_mapping(text_tokenizer).values())
    B = mod_dict[key]['tensor'].shape[0]

    for i in range(B):

        input_seq = mod_dict[key]['tensor'][i]
        input_seq = input_seq[mod_dict[key]['input_mask'][i] == 0]
        input_seq = input_seq.tolist()
        
        target_seq = mod_dict[key]['tensor'][i]
        target_seq = target_seq[mod_dict[key]['target_mask'][i] == 0]
        target_seq = target_seq.tolist()
        
        merged_seq = merge_span_masking(input_seq, target_seq, sentinel_ids=sentinel_ids)
    
        input_text = text_tokenizer.decode(input_seq, skip_special_tokens=False)
        target_text = text_tokenizer.decode(target_seq, skip_special_tokens=False)
        merged_text = text_tokenizer.decode(merged_seq, skip_special_tokens=False)

        input_texts.append(input_text)
        target_texts.append(target_text)
        merged_texts.append(merged_text)

    if B == 1:
        input_texts, target_texts, merged_texts = input_texts[0], target_texts[0], merged_texts[0]
    
    return input_texts, target_texts, merged_texts

def decode_dict(mod_dict, tokenizers, text_tokenizer, image_size=224, patch_size=16, 
                decoding_steps=25, activate_controlnet=False, controlnet_guidance_scale=2.5, controlnet_cond_scale=0.8,
                to_rgb=True, seed=None):
    """
    Decodes the model output dictionary into a dictionary of images and text.

    Args:
        mod_dict (dict): Model output dictionary.
        tokenizers (dict): Dictionary of tokenizers.
        text_tokenizer (tokenizers.Tokenizer): Text tokenizer.
        image_size (int): Image size.
        patch_size (int): Patch size.
        decoding_steps (int): Number of diffusion decoding steps (if applicable).
        activate_controlnet (bool): Whether to activate the RGB ControlNet and override the RGB detokenizer.
        controlnet_guidance_scale (float): Classifier-free guidance scale for the ControlNet.
        controlnet_cond_scale (float): ControlNet conditioning scale.
    """
    dec_dict = {}
    
    for key in mod_dict:
        k, res = get_transform_key(key), get_transform_resolution(key, image_size, to_tuple=False)

        if k == 'rgb':
            decoded = decode_input_rgb(mod_dict, key=key)
        elif k == 'tok_rgb':
            if not activate_controlnet or 'controlnet' not in tokenizers:
                decoded = decode_tok_rgb(
                    mod_dict, tokenizers, key=key, 
                    image_size=res, patch_size=patch_size, 
                    t=decoding_steps, verbose=False
                )
            else:
                decoded = decode_tok_rgb_controlnet(
                    mod_dict, tokenizers, key=key, 
                    image_size=res, patch_size=patch_size, 
                    t=decoding_steps, guidance_scale=controlnet_guidance_scale, 
                    cond_scale=controlnet_cond_scale, verbose=False
                )
        elif k == 'tok_normal':
            decoded = decode_tok_normal(
                mod_dict, tokenizers, key=key, 
                image_size=res, patch_size=patch_size, 
                t=decoding_steps, verbose=False
            )
        elif k == 'tok_depth':
            decoded = decode_tok_depth(
                mod_dict, tokenizers, key=key, 
                image_size=res, patch_size=patch_size, 
                t=decoding_steps, verbose=False, cmap='turbo' if to_rgb else None
            )
        elif k == 'tok_semseg':
            decoded = decode_tok_semseg(
                np.ones((res, res, 3)), mod_dict, tokenizers, key=key, 
                image_size=res, patch_size=patch_size, return_logits=not to_rgb
            )
        elif k == 'tok_clip':
            decoded = decode_tok_clip(
                mod_dict, tokenizers, key=key, 
                image_size=res, patch_size=patch_size
            )
        elif k in ['caption', 'det']:
            decoded = decode_text(mod_dict, key, text_tokenizer)[2]
            decoded = decoded if isinstance(decoded, list) else [decoded]
            decoded = [d.replace(' [EOS]', '') for d in decoded]
        elif k in ['metadata']:
            decoded = decode_metadata(
                mod_dict, text_tokenizer, key=key
            )
        elif k in ['t5_caption']: 
            if 'ascii_tensor' in mod_dict[key]:
                decoded = []
                for ascii_tensor in mod_dict[key]['ascii_tensor']:
                    ascii_values = ascii_tensor.flatten().tolist()
                    decoded_text = ''.join(chr(val) for val in ascii_values if val != 0)
                    decoded.append(f"T5-XXL embedding of: {decoded_text}")
                decoded = decoded[0] if len(decoded) == 1 else decoded
            else:
                decoded = "T5-XXL embedding"        
        dec_dict[key] = decoded
    return dec_dict




# Plotting utils

MOD_PRINT_NAMES = {
    'rgb': 'RGB',
    'tok_rgb': 'RGB (tok)',
    'tok_normal': 'Normal (tok)',
    'tok_depth': 'Depth (tok)',
    'tok_semseg': 'Semseg (tok)',
    'tok_clip': 'CLIP (tok)',
    'tok_canny': 'Canny (tok)',
    'tok_sam': 'SAM (tok)',

    'rgb@224': 'RGB@224',
    'tok_rgb@224': 'RGB@224 (tok)',
    'tok_normal@224': 'Normal@224 (tok)',
    'tok_depth@224': 'Depth@224 (tok)',
    'tok_semseg@224': 'Semseg@224 (tok)',
    'tok_clip@224': 'CLIP@224 (tok)',

    'rgb@448': 'RGB@448',
    'tok_rgb@448': 'RGB@448 (tok)',
    'tok_normal@448': 'Normal@448 (tok)',
    'tok_depth@448': 'Depth@448 (tok)',
    'tok_semseg@448': 'Semseg@448 (tok)',
    'tok_clip@448': 'CLIP@448 (tok)',

    'caption': 'Caption',
    'det': 'Detection',
    't5_caption': 'T5 XXL',
    'metadata': 'Metadata',
}

def remove_ticks_and_labels(ax):
    """
    Remove the axis ticks and labels

    Args:
        ax (matplotlib.axes.Axes): Axis to remove ticks and labels from
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
def remove_spines(ax):
    """
    Removes the spines from the given axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to remove spines from
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
def convert_string_to_bboxes(bboxes_str, bins=1000):
    """
    Converts a string of bboxes to a list of bboxes.

    Args:
        bboxes_str (str): String of bboxes
        bins (int): Number of bins (default: 1000)
    """
    bboxes_str = bboxes_str.split(" ")
    bboxes = []
    for token in bboxes_str:
        if "=" in token:
            coord = token.split("=")[1]
            coord = float(coord) / (bins - 1)

            if token.startswith("xmin="):
                bboxes.append([coord,])
            else:
                bboxes[-1].append(coord)
        elif len(bboxes[-1]) == 4:
            bboxes[-1].append(token)
        else:
            bboxes[-1][4] = " ".join([bboxes[-1][4], token])

    bboxes = [bbox for bbox in bboxes if len(bbox) == 5]

    return bboxes

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bboxes(img, bboxes_str, color=BOX_COLOR, thickness=2):
    """
    Visualizes bounding boxes on the image.

    Args:
        img (np.array): Image to draw bounding boxes on.
        bboxes_str (str): String containing bounding boxes in the format:
            xmin=1 ymin=2 xmax=3 ymax=4 class_name ...
        color (tuple): Color of the bounding box.
        thickness (int): Thickness of the bounding box.
    """
    if img is None:
        img = 255 * np.ones((256,256,3), dtype=np.int32)
    img = img.copy()
    
    bboxes_str = bboxes_str.replace('[PAD]', '')

    if len(bboxes_str.replace('[EOS]', '')) == 0:
        return img
    
    try:
        bboxes = convert_string_to_bboxes(bboxes_str.replace(' [EOS]', ''))
    except:
        return img
    
    for bbox in bboxes:
        x_min, y_min, x_max, y_max, class_name = bbox
        img_h, img_w = img.shape[0], img.shape[1]
        x_min, x_max, y_min, y_max = int(x_min * img_w), int(x_max * img_w), int(y_min * img_h), int(y_max * img_h)
    
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        
        ((text_width, text_height), _) = cv2.getTextSize(class_name.rstrip(), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=f"{class_name}",
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35, 
            color=TEXT_COLOR, 
            lineType=cv2.LINE_AA,
        )
    return img


def plot_text_in_square(ax, text, padding=0.5, fontsize=14, wrap_width=50):
    """
    Plots text in a square.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on
        text (str): Text to plot
        padding (float): Padding around the text
        fontsize (int): Font size of the text
        wrap_width (int): Width of the text to wrap
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if isinstance(text, list):
        text = text[0]

    text = text.replace('[PAD]', '')
    
    # Wrap the text if necessary
    wrapped_text = textwrap.fill(text, int(wrap_width))

    # Add the padding
    bbox_props = dict(boxstyle="square,pad=" + str(padding), facecolor="white", edgecolor="black")

    # Add the text to the plot
    ax.text(0.5, 0.5, wrapped_text, ha='center', va='center', fontsize=fontsize, bbox=bbox_props)

    remove_ticks_and_labels(ax)
    remove_spines(ax)
    
def plot_modality(dec_dict, key, ax, figscale=4.0):
    """
    Plots a single modality. Function name has a typo because of legacy reasons.

    Args:
        dec_dict (dict): Dictionary of decoded modalities
        key (str): Key of the modality to plot
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on
        figscale (float): Scaling factor for the figure (used to scale the caption box)
    """
    modality = dec_dict[key]
    k = get_transform_key(key)
    
    if 'tok' in k or k == 'rgb':
        ax.imshow(modality.clip(0,1))
    elif k == 'caption':
        plot_text_in_square(ax, modality, wrap_width=max(1,int(7*figscale))) # 7*figscale turns out to make caption box fit nicely
    elif k == 't5_caption':
        plot_text_in_square(ax, modality, wrap_width=max(1,int(7*figscale))) # 7*figscale turns out to make caption box fit nicely
    elif k == 'metadata':
        modality = ',\n'.join([f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}' for k, v in modality.items()])
        plot_text_in_square(ax, modality, wrap_width=max(1,int(7*figscale)), fontsize=11)
    elif k == 'det':
        bbox_img = visualize_bboxes(np.ones((224,224,3)), modality, thickness=2)
        ax.imshow(bbox_img.clip(0,1))
        
def plot_conds_and_targets(cond_domains, target_domains, dec_dicts, save_path=None, fs_titles=15, figscale=4.0, dpi=100):
    """
    Plots the conditioning and target modalities for a batch of samples.

    Args:
        cond_domains (list of str): List of conditioning domains
        target_domains (list of str): List of target domains
        dec_dicts (list of dicts): List of dictionaries containing the decoded conditioning and target modalities
        save_path (str): Path to save the figure. If None, the figure is not saved but plotted instead.
        fs_titles (int): Font size of the titles
        figscale (float): Scaling factor for the figure size (minimum 4.0 for good results)
        dpi (float): Dots per inch for the saved figure    
    """

    n_cond = len(cond_domains)
    n_target = len(target_domains)
    n_samples = len(dec_dicts)
    ncols = n_samples + 1 if n_cond > 0 else n_samples
    nrows = max(n_cond, n_target)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*figscale, nrows*figscale), facecolor='white')

    if nrows == 1 and ncols == 1:
        ax = np.array([[ax]])
    elif nrows == 1:
        ax = np.expand_dims(ax, axis=0)
    elif ncols == 1:
        ax = np.expand_dims(ax, axis=1)

    for cond_idx, cond_domain in enumerate(cond_domains):
        axi = ax[cond_idx, 0]
        plot_modality(dec_dicts[0], key=cond_domain, ax=axi)
        axi.set_title(f'Conditioning: {MOD_PRINT_NAMES[cond_domain]}', fontsize=fs_titles)

    # Remove spines that are not needed
    if n_cond > 0:
        for i in range(n_cond, nrows, 1):
            remove_spines(ax[i, 0])

    offset = 0 if n_cond == 0 else 1

    for sample_idx, dec_dict in enumerate(dec_dicts):
        for target_idx, target_domain in enumerate(target_domains):
            axi = ax[target_idx, sample_idx+offset]
            plot_modality(dec_dict, key=target_domain, ax=axi)
            axi.set_title(f'{sample_idx+1}.{target_idx+1}: {MOD_PRINT_NAMES[target_domain]}', fontsize=fs_titles)
                
        # Remove spines that are not needed
        for i in range(n_target, nrows, 1):
            remove_spines(ax[i, sample_idx+offset])

    for ax in fig.axes:
        remove_ticks_and_labels(ax)
    
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi) #, pil_kwargs={'quality': 30})
        plt.close()
    else:
        plt.show()


def save_conds_and_targets(cond_domains, target_domains, dec_dicts, save_dir, sample_idx, suffix=None, vis_det=False):
    """
    Saves the conditioning and target modalities for a batch of samples.

    Args:
        cond_domains (list of str): List of conditioning domains
        target_domains (list of str): List of target domains
        dec_dicts (list of dicts): List of dictionaries containing the decoded conditioning and target modalities
        save_dir (str): Path to save the modalities
        sample_idx (int): Unique index of the dataset sample
        suffix (str): Suffix to append to the saved file names
        vis_det (bool): Whether to visualize detection
    """
    for variant_idx, dec_dict in enumerate(dec_dicts):

        for domain in cond_domains + target_domains:
            if variant_idx != 0 and domain in cond_domains:
                continue
            
            variant_suffix = f'_{variant_idx}' if domain in target_domains else ''
            if suffix is not None:
                variant_suffix += f'_{suffix}'

            domain_save_dir = os.path.join(save_dir, 'conds' if domain in cond_domains else 'targets', domain)
            os.makedirs(domain_save_dir, exist_ok=True)

            if 'tok' in domain or domain == 'rgb':
                img = Image.fromarray((255 * dec_dict[domain]).astype(np.uint8))
                if domain == 'tok_clip':
                    img = img.resize((224,224), resample=Image.NEAREST)
                save_path = os.path.join(domain_save_dir, f'{sample_idx:06d}{variant_suffix}.png')
                img.save(save_path)

            elif domain in ['caption', 'det']:
                if vis_det:
                    save_path = os.path.join(domain_save_dir, f'{sample_idx:06d}{variant_suffix}.png')
                    bbox_img = visualize_bboxes(np.ones((512,512,3)), dec_dict[domain], thickness=2)
                    bbox_img = Image.fromarray((255 * bbox_img.clip(0,1)).astype(np.uint8))
                    bbox_img.save(save_path)
                else:
                    # Save caption as text file
                    save_path = os.path.join(domain_save_dir, f'{sample_idx:06d}{variant_suffix}.txt')
                    with open(save_path, 'w') as f:
                        f.write(dec_dict[domain])


def plot_images_with_captions(images, captions, save_path=None, dpi=100, wrap_length=40, figscale=4.0):
    """
    Plots images with their corresponding captions.

    Parameters:
    - images (torch.Tensor): A tensor of shape Bx3xHxW with images.
    - captions (list): A list of B captions.
    """
    assert len(images) == len(captions), "Number of images must match number of captions!"
    
    B = len(images)
    sqrt_B = int(B**0.5)
    
    # Determine the number of rows and columns for subplots
    nrows = sqrt_B
    ncols = (B + nrows - 1) // nrows
    
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figscale*ncols, figscale*nrows))

    axarr = np.array([axarr]) if nrows == 1 and ncols == 1 else axarr.ravel()
    
    for i, ax in enumerate(axarr):
        if i < B:
            # Convert tensor image to numpy
            image_np = images[i].permute(1, 2, 0).cpu().float().numpy()
            ax.imshow(image_np)

            # Place caption below the image
            caption_wrapped = textwrap.fill(captions[i], width=wrap_length)
            ax.text(0.5, -0.1, caption_wrapped, ha='center', va='top', transform=ax.transAxes, wrap=True)

            ax.axis("off")
        else:
            ax.axis("off")  # Hide any additional subplots

    plt.subplots_adjust(hspace=0.6)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close()
    else:
        plt.show()