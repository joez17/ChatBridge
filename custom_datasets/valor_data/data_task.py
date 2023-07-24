"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Dataset interfaces
"""

from cProfile import label
import json
from toolz.sandbox import unzip
import torch
from torch.utils.data import Dataset

from torchvision.transforms.transforms import *
from torchvision import transforms
import random
from os.path import join 
from decord import VideoReader
# import av
import os

from PIL import Image
from utils.logger import LOGGER
import ipdb
import matplotlib.pyplot as plt
import string 
from time import time
from typing import List, Tuple, Optional, Dict
from torch import Tensor
import torch.nn.functional as F
punctuation = string.punctuation
import numpy as np
from torchvision.transforms import functional as transform_F
from dataloader import KVReader
import io
import decord
import torchaudio
import pickle
from utils.hdfs_io import hload_pkl
from custom_datasets.valor_data.loader import pre_caption

class TASKDataset(Dataset):
    def __init__(self, desc, video_mapper, audio_mapper, training, task, prompts, zh=False):
        
        self.video_mapper = video_mapper
        if self.video_mapper is not None:
            self.video_mapper.training = training 

        self.audio_mapper = audio_mapper
        if self.audio_mapper is not None:
            self.audio_mapper.training = training 

        self.annos = json.load(open(desc))

        self.idx = list(range(len(self.annos)))
        if self.video_mapper is not None:
            self.dataset_name = self.video_mapper.datatype.split('_')[-1]
        else:
            self.dataset_name = 'none'
        self.training = training
        self.task = task
        if task in prompts.keys():
            self.prompts = prompts[task]
        else:
            self.prompts = None
        self.zh = zh
        
    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
        anno = self.annos[i]
        if 'video_id' in anno.keys():
            id_ = anno['video_id']
        else:
            id_ = anno['image_id']
 
        raw_captions = [None]
        question = [None]
        answer = [None]
        text = [None]

        video_pixels = None 
        audio_spectrograms = None 
        if 'desc' in anno or 'caption' in anno:
            raw_captions = anno['desc'] if 'desc' in anno else anno['caption'] 
            if isinstance(raw_captions, list):
                raw_captions = random.choice(raw_captions)
            else:
                raw_captions = raw_captions
        if 'question' in anno:
            if isinstance(anno['question'],list):       
                sampled_idx = random.choice(list(range(len(anno['question']))))         
                question = anno['question'][sampled_idx]
                answer = anno['answer'][sampled_idx]
            else:
                question = anno['question']
                answer = anno['answer']  
        if 'text' in anno:
            if isinstance(anno['text'],list):       
                text = ', '.join(anno['text'])
            else:
                text = anno['text']
        if 'conversations' in anno:
            conversations = anno['conversations']

        id_txt = [id_]

        if self.video_mapper is not None:
            video_pixels = self.video_mapper[id_]
            if video_pixels is None: ###wrong img/video and needs to resample 
                resample_idx = random.choice(self.idx)
                LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
                return self.__getitem__(resample_idx)

        if self.audio_mapper is not None:   
            audio_spectrograms = self.audio_mapper[id_]
            if audio_spectrograms is None: ### wrong audio and needs to resample
                resample_idx = random.choice(self.idx)
                LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong audio, use {resample_idx} instead.')
                return self.__getitem__(resample_idx)


        if 'ti-' in self.task:
            if self.zh:
                prefix = '给定图片：<query>。'
            else:
                prefix = '###Human: Given following image: <query>.'
        elif 'tv-' in self.task:
            if self.zh:
                prefix = '给定视频：<query>。'
            else:
                prefix = '###Human: Given following video: <query>.'
        elif 'ta-' in self.task:
            if self.zh:
                prefix = '给定音频：<query>。'
            else:
                prefix = '###Human: Given following audio: <query>.'
        elif 'tva-' in self.task:
            if self.zh:
                prefix = '给定视频：<query>。'
            else:
                prefix = '###Human: Given following video: <query> and its background audio: <query>.'


        conversation = ""
        if 'cap' in self.task:
            text_input = random.choice(self.prompts)
            text_output = raw_captions
        elif 'qa' in self.task:
            text_input = random.choice(self.prompts).replace('<QUESTION>', question)
            text_output = answer
        elif 'ocr' in self.task:
            text_input = random.choice(self.prompts)
            text_output = text
        elif 'det' in self.task or 'rea' in self.task:
            text_input = clean(conversations[0]['value'])
            text_output = conversations[1]['value']
        elif 'conv' in self.task:
            conversation = prefix
            for idx in range(min(len(conversations), 8)):
                conv = conversations[idx]
                conv_text = conv['value']
                if conv['from']=="human":
                    conversation += f'###Human: {clean(conv_text)}'
                elif conv['from']=='gpt':
                    conversation += f'###Assistant:{conv_text}'
                    if self.zh:
                        conversation += '</s>'
            conversation += "###"
        if len(conversation)==0:
            conversation = prefix + f'###Human: {text_input}###Assistant:{text_output}'
            if self.zh:
                conversation += '</s>\n###'
            else:
                conversation += '\n###'
        if self.zh:
            conversation = conversation.replace('###Human: ', '\n###问题：').replace('###Assistant:','\n###答案：').replace('</s>\n','\n</s>')
        return id_, video_pixels, audio_spectrograms, conversation

def clean(text):
    text = text.replace("<image>\n", "").replace("\n<image>", "")
    text = text.replace("<video>\n", "").replace("\n<video>", "")
    text = text.replace("<audio>\n", "").replace("\n<audio>", "")
    text = text.replace("<图片>\n", "").replace("\n<图片>", "")
    text = text.replace("<视频>\n", "").replace("\n<视频>", "")
    text = text.replace("<音频>\n", "").replace("\n<音频>", "")
    return text


def task_collate(inputs):
    (ids, video_pixels, audio_spectrograms, conversation) = map(list, unzip(inputs))
    video_pixels = torch.stack(video_pixels, dim=0) if video_pixels[0] is not None else None
    audio_spectrograms = torch.stack(audio_spectrograms, dim=0) if audio_spectrograms[0] is not None else None
    batch =   {'ids': ids,
             'conversation': conversation,
             'image': video_pixels,
             'audio':audio_spectrograms}   
    return batch


    
