"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from chatbridge.datasets.datasets.base_dataset import BaseDataset

from chatbridge.datasets.datasets.caption_datasets import CaptionDataset


class VideoCaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class VideoCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["video"] if 'video' in ann.keys() else ann['video_id']+'.mp4'
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "image_id": ann["image_id"] if "image_id" in ann.keys() else ann['video_id'],
            "instance_id": ann["instance_id"],
        }



class ValorCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, aud_processor, vis_root, aud_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.aud_processor = aud_processor
        self.aud_root = aud_root
    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"] if 'video' in ann.keys() else ann['video_id']+'.mp4'
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)
        aname = ann["audio"] if 'audio' in ann.keys() else vname.replace('mp4', 'wav')
        apath = os.path.join(self.aud_root, aname)
        auds = self.aud_processor(apath)
        return {
            "video": video,
            "audio": auds, 
            "image_id": ann["image_id"] if "image_id" in ann.keys() else ann['video_id']
        }

import torch
from PIL import Image
class VISTCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video_id"]
        video_path = os.path.join(self.vis_root, vname)
        images = []
        for i in range(5):
            image_path = os.path.join(video_path, f'frame_000{i}.jpg')
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)
            images.append(image.unsqueeze(0))
        images = torch.cat(images, dim=0)
        return {
            "video": images,
            "video_id": ann["video_id"],
            "caption": ann["caption"],
        }