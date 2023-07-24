"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from chatbridge.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import torchaudio
import torch
import random
class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        # visual_key = "image" if "image" in ann else "video"
        if "image" in ann:
            visual_key = "image"
        elif "video" in ann:
            visual_key = "video"
        elif "audio" in ann:
            visual_key = "audio"
        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )


class RetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instance_id": ann["instance_id"],
        }


class RetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):

        image_path = os.path.join(self.vis_root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {"image": image, "index": index}


class VideoRetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["video"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        vpath = os.path.join(self.vis_root, ann["video"])

        video = self.vis_processor(vpath)
        caption = self.text_processor(ann["caption"])

        # return image, caption, self.img_ids[ann['image_id']]
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["video"]],
        }


class VideoRetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["video"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        vpath = os.path.join(self.vis_root, ann["video"])
        video = self.vis_processor(vpath)

        return {"video": video, "index": index}

# class AudioRetrievalDataset(BaseDataset, __DisplMixin):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         """
#         vis_root (string): Root directory of images (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         """
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths)

#         self.img_ids = {}
#         n = 0
#         for ann in self.annotation:
#             img_id = ann["image_id"]
#             if img_id not in self.img_ids.keys():
#                 self.img_ids[img_id] = n
#                 n += 1

#     def __getitem__(self, index):

#         ann = self.annotation[index]

#         image_path = os.path.join(self.vis_root, ann["image"])
#         image = Image.open(image_path).convert("RGB")

#         image = self.vis_processor(image)
#         caption = self.text_processor(ann["caption"])

#         return {
#             "image": image,
#             "text_input": caption,
#             "image_id": self.img_ids[ann["image_id"]],
#             "instance_id": ann["instance_id"],
#         }


class AudioRetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.mean =  15.41663
        self.std = 6.55582 
        self.melbins = 64
        self.target_length = 1024
        self.training = False
        self.sample_num = 1
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["audio"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):
        ann = self.annotation[index]
        waveform, sr = torchaudio.load(os.path.join(self.vis_root, ann["audio"]))
        if sr != 16000:
            trans = torchaudio.transforms.Resample(sr, 16000)
            waveform = trans(waveform)
    
        waveform = waveform * 2 ** 15
        fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=self.melbins, sample_frequency=16000, frame_length=25, frame_shift=10)
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
        return {"audio": fbank, "index": index}

def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]