import os
import logging
import torch
from webdataset import WebLoader
from video2dataset.dataloader import get_video_dataset
from torch.utils.data.dataloader import default_collate

SHARDS = "/store/swissai/a08/data/raw/howto100m/v2d_40k/00000{00000..00030}.tar" #use this
# SHARDS = "/store/swissai/a08/data/raw/howto100m/v2d_40k/0000000002.tar" #use this

if __name__ == "__main__":
    decoder_kwargs = {
        "n_frames": 8,  # get 8 frames from each video
        "fps": 10,  # downsample to 10 FPS
        "num_threads": 12,  # use 12 threads to decode the video
    }
    resize_size = crop_size = 256
    batch_size = 10

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resize_size,
        crop_size=crop_size,
        enforce_additional_keys=[],
    )

    num_workers = 0  # 6 dataloader workers for gpu

    dl = WebLoader(dset, batch_size=None, num_workers=num_workers)

    for i, sample in enumerate(dl):
        print("here in file")
        print("in loop")
        try:
            
            video_batch = sample["mp4"]
            print(video_batch.shape)  # torch.Size([32, 8, 256, 256, 3])
            # TODO: need to add option for text/metadata preprocessing (tokenization etc.)
            text_batch = sample["json"]
            #print(text_batch[0])
        except Exception as e:
            print(f"index: {i} and {e}")
            
            
    print("done")