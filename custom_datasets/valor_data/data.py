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
class VideoMapper(object):
    def __init__(self, video_dir, opts, data_type = 'video', sample_num = 4, video_transforms='none',data_format='kvreader'):
        self.video_dir = video_dir
        self.datatype = data_type
        self.frame_syncaug = True
        self.training = True
        self.sample_num = sample_num 
        self.data_format = data_format
        
        self.resolution = opts.video_resolution

        if opts.video_encoder_type.startswith('clip') or opts.video_encoder_type.startswith('evaclip'):
            self.mean = [0.48145466, 0.4578275, 0.40821073] 
            self.std  = [0.26862954, 0.26130258, 0.27577711]
        else:       
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]
        
        LOGGER.info(f'{data_type} mean : {self.mean}')
        LOGGER.info(f'{data_type} std : {self.std}')      
        
        self.video_transforms = video_transforms
        if video_transforms == 'none':
            self.train_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
                
            self.test_transforms = transforms.Compose([Resize((self.resolution,self.resolution)),
                                                        Normalize(self.mean,self.std)])
        elif video_transforms == 'crop_flip':
            self.train_transforms = transforms.Compose([RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                                        RandomHorizontalFlip(),
                                                        Normalize(self.mean,self.std)])

            self.test_transforms = transforms.Compose([Resize(self.resolution),
                                    CenterCrop(self.resolution),
                                    Normalize(self.mean,self.std)])
                                    
        else:
            raise NotImplementedError

        LOGGER.info(f'{data_type} video_transforms : {video_transforms} ')    
            
    def __getitem__(self, id_):
      
        if  self.datatype.startswith('video'):

            video_pixels = []        
            sample_num = self.sample_num
            try:

                if self.data_format == 'kvreader':
                    videos  = self.kv_reader.read_many([id_,])[0]                   
                    file_obj = io.BytesIO(videos)   
    

                    container = decord.VideoReader(file_obj)     
                    frames_ids = list(range(len(container)))
            
                    frames_splited = split(frames_ids, sample_num)
                    if self.training:
                        sample_idx = [random.choice(i) for i in frames_splited]
                    else:
                        sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited] 

                    frames = container.get_batch(sample_idx).numpy()
    
                    for i in frames: 
                        frame = Image.fromarray(i)
                        frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
                        video_pixels.append(frame.unsqueeze(0))

                elif self.data_format == 'frame':
                    frame_path = os.path.join(self.video_dir, str(id_))
                    if not os.path.exists(frame_path):
                        frame_path = os.path.join(self.video_dir, str(id_)[:2].lower(), str(id_))
                    frames = os.listdir(frame_path)
                    frames.sort()   ### ['img_0001.jpg','img_0002.jpg',...]
                    sample_num = self.sample_num
                    frames_splited = split(frames,sample_num)    
                    if self.training:
                        sample_idx = [random.choice(i) for i in frames_splited]
                    else:
                        sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited]
                    for i in range(sample_num):
                        frame = Image.open(os.path.join(frame_path,sample_idx[i]))
                        frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
                        video_pixels.append(frame.unsqueeze(0))
                elif self.data_format == 'raw':
                    video_path = os.path.join(self.video_dir, str(id_))
                    if not os.path.exists(video_path):
                        video_path = os.path.join(self.video_dir, str(id_).split('/')[-1])
                    container = decord.VideoReader(uri=video_path)    
                    frames_ids = list(range(len(container)))
            
                    frames_splited = split(frames_ids, sample_num)
                    if self.training:
                        sample_idx = [random.choice(i) for i in frames_splited]
                    else:
                        sample_idx = [i[(len(i)+1)//2-1] for i in frames_splited] 

                    frames = container.get_batch(sample_idx).numpy()
    
                    for i in frames: 
                        frame = Image.fromarray(i)
                        frame = transforms.ToTensor()(frame)   ## frame: 3XhXw
                        video_pixels.append(frame.unsqueeze(0))


                video_pixels = torch.cat(video_pixels,dim=0)   ### nX3xHxW
                if self.training:
                    video_pixels = self.train_transforms(video_pixels)    
                else:
                    video_pixels = self.test_transforms(video_pixels)     
                return video_pixels

            except Exception as e:
                print(e)
                print(id_)
                return None



        elif self.datatype.startswith('image'):
            
            try:
                if self.data_format=='kvreader':
                    img_path = self.kv_reader.read_many([id_])[0]
                    img_path = io.BytesIO(img_path)
                elif self.data_format=='frame':
                    img_path = os.path.join(self.video_dir, id_)
                    if not os.path.exists(img_path):
                        img_path += '.jpg'
                    if not os.path.exists(img_path):
                        img_path =  img_path.replace('.jpg','.JPEG')

                img = Image.open(img_path)
                img = img.convert('RGB')  #### convert 1-channel gray image and 4-channel CMYK image to RGB image
                img = transforms.ToTensor()(img)
                if self.training:    
                    img = self.train_transforms(img)
                else:
                    img = self.test_transforms(img)

                img = img.unsqueeze(0)
                return img

            except Exception as e:
                print(e)
                return None

        else:
            raise NotImplementedError()

def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]





class AudioMapper(object):
    def __init__(self, audio_dir, opts, sample_num, check_exists=True ):
        self.audio_dir = audio_dir
        self.melbins = opts.audio_melbins
        self.target_length = opts.audio_target_length
        self.check_exists = check_exists
        self.training = True
        self.frame_shift = 10
        self.sample_num = sample_num
        self.audio_type = opts.audio_type
        self.audio_encoder_type = opts.audio_encoder_type
        if self.audio_encoder_type in ['ast','ssast']:
            self.mean = -4.2677393
            self.std = 4.5689974
        elif self.audio_encoder_type == 'beat':
            self.mean =  15.41663
            self.std = 6.55582 
        else:
            raise NotImplementedError
       


    def __getitem__(self, id_):

        wav_file = os.path.join(self.audio_dir, id_)
        if self.check_exists:
            if not os.path.exists(wav_file):
                wav_file = os.path.join(self.audio_dir, id_+'.wav')
            if not os.path.exists(wav_file):
                wav_file = wav_file.replace('wav','mp3')
            if not os.path.exists(wav_file):
                wav_file = wav_file.replace('mp3','mkv')
            if not os.path.exists(wav_file):
                wav_file = os.path.join(self.audio_dir, id_[:2].lower(), id_+'.wav')
            if not os.path.exists(wav_file):
                wav_file = wav_file.replace('wav','mkv')
            if not os.path.exists(wav_file):
                # with open('./output/filter_audios.txt','a') as f:
                #     f.writelines(f'{id_}\n')
                print('not have audio', id_)
                return torch.zeros(self.sample_num, self.target_length, self.melbins)
        

        try:
            
            if self.audio_encoder_type in ['ast','ssast']:
                
                waveform, sr = torchaudio.load(wav_file)

                waveform = waveform - waveform.mean()
                fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                        window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.frame_shift)

                                        #### fbank shape :(src_length,64)
         
            elif self.audio_encoder_type=='beat':

                waveform, sr = torchaudio.load(wav_file)
                if sr != 16000:
                    trans = torchaudio.transforms.Resample(sr, 16000)
                    waveform = trans(waveform)
            
                waveform = waveform * 2 ** 15
                fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=self.melbins, sample_frequency=16000, frame_length=25, frame_shift=10)

            else:
                raise NotImplementedError


            # ### normalization
            fbank = (fbank - self.mean) / (self.std * 2)


            src_length = fbank.shape[0]
            # #### sample 

            output_slices = []

            pad_len = max(self.target_length * self.sample_num -src_length, self.target_length - src_length%self.target_length)

            fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)


            total_slice_num = fbank.shape[0] // self.target_length
            total_slice_num = list(range(total_slice_num))
            total_slice_num = split(total_slice_num, self.sample_num)
            
            if self.training:
                sample_idx = [random.choice(i) for i in total_slice_num]
            else:
                sample_idx = [i[(len(i)+1)//2-1] for i in total_slice_num]

            
            for i in sample_idx:
                output_slices.append(fbank[i*self.target_length : (i+1)*self.target_length])
            
            fbank = torch.stack(output_slices,dim=0)   ### n, 1024, 128

            
            return fbank
           

        except Exception as e:
            print(e)
            return    
                








class VALORDataset(Dataset):
    def __init__(self, desc, video_mapper, audio_mapper, training):
        
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
        
        
    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
        anno = self.annos[i]
        if isinstance(anno, str):
            question_root = '/mnt/bn/sihanchen/valor114/cc3m/CC3M_QA_3M_right_224_with_right_balanced_scaled_binary'
            id_ = anno
            annos = json.load(open(os.path.join(question_root, id_+'.json')))
            anno = random.choice(annos)
        else:
            id_ = anno['video_id']
 
        raw_captions = [None]
        question_id = [None]
        question = [None]
        answer = [None]
        text = [None]

        video_pixels = None 
        audio_spectrograms = None 

        if 'desc' in anno or 'caption' in anno:
            raw_captions = anno['desc'] if 'desc' in anno else anno['caption'] 
            if isinstance(raw_captions, list):
                if self.training and not self.all_text:
                    sentence = [random.choice(raw_captions)]

            else:
                raw_captions=[pre_caption(raw_captions)]
            num_samples = len(raw_captions)

        elif 'question' in anno:
            if isinstance(anno['question'],list):       
                if self.training and not self.all_text:
                    sampled_idx = random.choice(list(range(len(anno['question']))))         
                    answer_weights = []
                    answer_nums = 1
                    question = [anno['question'][sampled_idx]]
                    answer = [anno['answer'][sampled_idx]]
                else:
                    question = anno['question']
                    answer = anno['answer']
                    if 'question_id' in anno:
                        question_id = anno['question_id']
            else:
                question = [anno['question']]
                answer = [anno['answer']]     
            num_samples = len(question)
        if 'text' in anno:
            if isinstance(anno['text'],list):       
                # if self.training and not self.all_text:
                #     sampled_idx = random.choice(list(range(len(anno['text']))))         
                #     text = [anno['text'][sampled_idx]]
                # else:
                #     text = anno['text']
                # random.shuffle(anno['text'])
                text = [','.join(anno['text'])]
            else:
                text = [anno['text']]
            num_samples = len(text)

        id_txt = [id_] * num_samples

        # print(raw_captions)

        
        if self.video_mapper is not None:
            video_pixels = self.video_mapper[id_]
            if video_pixels is None: ###wrong img/video and needs to resample 
                # with open('./output/filter_videos.txt','a') as f:
                #     f.writelines(f'{id_}\n')
                resample_idx = random.choice(self.idx)
                LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} instead.')
                return self.__getitem__(resample_idx)

        if self.audio_mapper is not None:   
            audio_spectrograms = self.audio_mapper[id_]
            
            if audio_spectrograms is None: ### wrong audio and needs to resample

                # json.dump([],open(f'./output/filter_audios/{id_}.json','w'))
                resample_idx = random.choice(self.idx)
                LOGGER.info(f'current idx {id_} from {self.dataset_name} returns wrong audio, use {resample_idx} instead.')
                return self.__getitem__(resample_idx)


        return id_, raw_captions, video_pixels, audio_spectrograms, id_txt, num_samples, question, answer, question_id, text




def valor_collate(inputs):
    

    (ids, raw_captions, video_pixels, audio_spectrograms, ids_txt, num_samples, questions, answers, question_ids, texts) = map(list, unzip(inputs))

    
    ids_txt = [ j  for i in ids_txt for j in i]
    raw_captions = [ j  for i in raw_captions for j in i]
    questions = [ j  for i in questions for j in i]
    answers = [ j  for i in answers for j in i]
    question_ids = [ j  for i in question_ids for j in i]
    video_pixels = torch.stack(video_pixels, dim=0) if video_pixels[0] is not None else None
    audio_spectrograms = torch.stack(audio_spectrograms, dim=0) if audio_spectrograms[0] is not None else None
    texts = [ j  for i in texts for j in i]
    
    
    # batch =   {'ids': ids,
    #          'raw_captions': raw_captions,
    #          'video_pixels': video_pixels,
    #          'audio_spectrograms':audio_spectrograms,
    #          'ids_txt': ids_txt,
    #          'sample_num': num_samples,
    #          'raw_questions': questions,
    #          'raw_answers': answers,
    #          'question_ids': question_ids}
    batch =   {'ids': ids,
             'text_input': raw_captions,
             'image': video_pixels,
             'audio':audio_spectrograms,
             'ids_txt': None,
             'sample_num': num_samples, 
             'question': questions,
             'answer':answers, 
             'text':texts}   
    return batch

def valor_collate_stage2(inputs):
    

    (ids, raw_captions, video_pixels, audio_spectrograms, ids_txt, num_samples, questions, answers, question_ids) = map(list, unzip(inputs))

    
    ids_txt = [ j  for i in ids_txt for j in i]
    text_input = []
    text_output = []
    for i in raw_captions:
        for j in i:
            caps = j.split(' ')
            if len(caps)>=2:
                rand = random.randint(1, len(caps)//2)
                text_input += [' '.join(caps[:rand])]
                text_output += [' '.join(caps[rand:])]
            else:
                text_input += ['']
                text_output += [j]

    raw_captions = [ j  for i in raw_captions for j in i]
    questions = [ j  for i in questions for j in i]
    answers = [ j  for i in answers for j in i]
    question_ids = [ j  for i in question_ids for j in i]
    video_pixels = torch.stack(video_pixels, dim=0) if video_pixels[0] is not None else None
    audio_spectrograms = torch.stack(audio_spectrograms, dim=0) if audio_spectrograms[0] is not None else None

    
    
    # batch =   {'ids': ids,
    #          'raw_captions': raw_captions,
    #          'video_pixels': video_pixels,
    #          'audio_spectrograms':audio_spectrograms,
    #          'ids_txt': ids_txt,
    #          'sample_num': num_samples,
    #          'raw_questions': questions,
    #          'raw_answers': answers,
    #          'question_ids': question_ids}
    batch =   {'ids': ids,
             'text_input': text_input,
             'text_output': text_output,
             'image': video_pixels,
             'audio':audio_spectrograms,
             'ids_txt': None,
             'sample_num': num_samples}   
    return batch


    
