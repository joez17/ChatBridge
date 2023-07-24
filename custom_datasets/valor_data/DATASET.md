# First-stage dataset preparation

## Download the filtered Conceptual Captions, SBU, LAION datasets

We use the filtered Conceptual Captions, SBU, LAION datasets as the image-text dataset.
You can refer to [minigpt4](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/dataset/README_1_STAGE.md) to prepare these datasets.

## Download the webvid2.5m datasets

We use the webvid2.5m datasets as the video-text dataset.

You can refer to [video2dataset](https://github.com/iejMac/video2dataset/blob/main/dataset_examples/WebVid.md) to prepare these datasets.

## Download the wavcaps datasets

We use the wavcaps datasets as the audio-text dataset.

You can refer to [WavCaps](https://github.com/XinhaoMei/WavCaps#dataset) to prepare these datasets.

## Re-organize all datasets

We recommand you to re-organize all one-stage pretraining datasets in two ways.

### Custom datasets
The first way is to use custom_datasets [custom_datasets/valor_data/data.py](custom_datasets/valor_data) in our codes. 

You need to re-organize the pretraining datasets as follows:
 ```
    ├── datasets
    │   ├── dataset_name
    │   │   ├── images(optional)
    │   │   │    ├── image0.mp4
    │   │   │    └── image1.mp4
    │   │   ├── videos(optional)
    │   │   │    ├── video0.mp4
    │   │   │    └── video1.mp4
    │   │   ├── frames(optional)
    │   │   │    ├── video0
    │   │   │    │   ├──img_0001.jpg
    │   │   │    │   └──img_0002.jpg
    │   │   │    └── video1
    │   │   │    │   ├──img_0001.jpg
    │   │   │    │   └──img_0002.jpg
    │   │   ├── audios(optional)
    │   │   │    ├── video0.wav
    │   │   │    └── video1.wav
    │   │   └── pretrain_txt_mapper.json    
```
And we create a indepentdent config file [train_configs/audio%cc16m%webvid2m%laion_v4a2.json](train_configs/audio%cc16m%webvid2m%laion_v4a2.json) to manage the pretraining datasets and its sample rate/batch size in training procedure. You can set different data type, sample rate, batch size and num workers for each dataloader by changing task, steps, batch_size and n_workers.

For more details, please refer to [valor](https://github.com/TXH-mercury/VALOR).




### default datasets
You can also use default datasets provided by [LAVIS](https://github.com/salesforce/LAVIS) and [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4),
which re-organize the datasets in following ways:

```
.
├── ${MINIGPT4_DATASET}
│   ├── cc_sbu
│       ├── convert_cc_sbu.py
│       ├── download_cc_sbu.sh
│       ├── ccs_synthetic_filtered_large.json
│       ├── ccs_synthetic_filtered_large.tsv
│       └── cc_sbu_dataset
│           ├── 00000.tar
│           ├── 00000.parquet
│           ...
│   ├── laion
│       ├── convert_laion.py
│       ├── download_laion.sh
│       ├── laion_synthetic_filtered_large.json
│       ├── laion_synthetic_filtered_large.tsv
│       └── laion_dataset
│           ├── 00000.tar
│           ├── 00000.parquet
│           ...
...   
```

# Second-stage dataset preparation

## Download MULTIS

We provide re-organized text annotation files in MULTIS datasets in this [googledrive_link](https://drive.google.com/file/d/1C7k8flfITJ1GxMwFSvEmBFGyevDZl1ke/view?usp=drive_link) or [baiduyun_link](https://pan.baidu.com/s/1GsUi4yLsBBEjkGu-4nwngw) key:2gt3.

Additionally, you should download the raw images/videos/audios by yourself.
These raw data is from MSCOCO, MSRVTT and Audioset.


## Re-organize all instruction datasets
two-stage datasets are re-organized as following formats:

 ```
    ├── datasets
    │   ├── MULTIS
    │   │   ├── images(optional)
    │   │   │    ├── image0.mp4
    │   │   │    └── image1.mp4
    │   │   ├── videos(optional)
    │   │   │    ├── video0.mp4
    │   │   │    └── video1.mp4
    │   │   ├── frames(optional)
    │   │   │    ├── video0
    │   │   │    │   ├──img_0001.jpg
    │   │   │    │   └──img_0002.jpg
    │   │   │    └── video1
    │   │   │    │   ├──img_0001.jpg
    │   │   │    │   └──img_0002.jpg
    │   │   ├── audios(optional)
    │   │   │    ├── video0.wav
    │   │   │    └── video1.wav
    │   │   │── MULTIS_annotation      
    │   │   │    ├── annotation0.json
    │   │   │    └── annotation1.json 
```

You can also change dataloader settings in [instructiontuning_configs/ivaav_inschat.json](instructiontuning_configs/ivaav_inschat.json).
And you can add or modify task-specific prompts in [instructiontuning_configs/task_prompt.json](instructiontuning_configs/task_prompt.json)