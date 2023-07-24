"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from chatbridge.datasets.datasets.base_dataset import BaseDataset

from chatbridge.datasets.datasets.caption_datasets import CaptionDataset


class AudioCaptionEvalDataset(BaseDataset):
    def __init__(self, aud_processor, text_processor, aud_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(None, text_processor, None, ann_paths)
        self.aud_processor = aud_processor
        self.aud_root = aud_root

    def __getitem__(self, index):
        ann = self.annotation[index]

        aname = ann["audio"]
        audio_path = os.path.join(self.aud_root, aname)

        audio = self.aud_processor(audio_path)
        # caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "audio": audio,
            "image_id": ann["audio"],
        }
