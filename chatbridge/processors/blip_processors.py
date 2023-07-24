"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from chatbridge.common.registry import registry
from chatbridge.processors.base_processor import BaseProcessor
from chatbridge.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchaudio

class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("blip_question")
class BlipQuestionProcessor(BaseProcessor):
    def __init__(self, max_words=50):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_words = cfg.get("max_words", 50)

        return cls(max_words=max_words)

    def pre_question(self, question):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question


@registry.register_processor("blip_image_train")
class BlipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("blip_image_eval")
class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)

import torch
@registry.register_processor("blip_audio_eval")
class BlipAudioEvalProcessor(BaseProcessor):
    def __init__(self, sample_num=1):
        super().__init__()

        self.melbins = 64
        self.target_length = 1024

        self.training = True
        self.frame_shift = 10
        self.sample_num = sample_num

        self.mean =  15.41663
        self.std = 6.55582 


    def __call__(self, fbank):
        if fbank is None:
            return torch.zeros(self.sample_num, self.target_length, self.melbins)
        if isinstance(fbank, str):
            fbank = load_wav(fbank)
        if fbank is None:
            return torch.zeros(self.sample_num, self.target_length, self.melbins)
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
        
        # if self.training:
        #     sample_idx = [random.choice(i) for i in total_slice_num]
        # else:
        #     sample_idx = [i[(len(i)+1)//2-1] for i in total_slice_num]
        sample_idx = [i[(len(i)+1)//2-1] for i in total_slice_num]
        
        for i in sample_idx:
            output_slices.append(fbank[i*self.target_length : (i+1)*self.target_length])
        
        fbank = torch.stack(output_slices,dim=0)   ### n, 1024, 128

        
        return fbank


    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        sample_num = cfg.get("sample_num", 1)

        # mean = cfg.get("mean", None)
        # std = cfg.get("std", None)

        return cls(sample_num=sample_num)

def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]

import os

def load_wav(wav_file):
    if not os.path.exists(wav_file):
        wav_file = wav_file.replace('.mp4', '')
        id_ = wav_file.split('/')[-1]
        audio_dir = wav_file.replace(id_, '')
        wav_file = os.path.join(audio_dir, str(id_)[:2].lower(), str(id_))
        if not os.path.exists(wav_file):
            print('no audio:', wav_file)
            return None
    waveform, sr = torchaudio.load(wav_file)
    if sr != 16000:
        trans = torchaudio.transforms.Resample(sr, 16000)
        waveform = trans(waveform)

    waveform = waveform * 2 ** 15
    fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=64, sample_frequency=16000, frame_length=25, frame_shift=10)
    return fbank