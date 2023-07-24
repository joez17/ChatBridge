"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from collections import OrderedDict

from chatbridge.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)


class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class AVQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, aud_processor, vis_root, aud_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.aud_processor = aud_processor
        self.aud_root = aud_root

    # def _build_class_labels(self, ans_path):
    #     ans2label = json.load(open(ans_path))

    #     self.class_labels = ans2label

    # def _get_answer_label(self, answer):
    #     if answer in self.class_labels:
    #         return self.class_labels[answer]
    #     else:
    #         return len(self.class_labels)

    def __getitem__(self, index):
        # assert (
        #     self.class_labels
        # ), f"class_labels of {__class__.__name__} is not built yet."
        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)
        aname = ann["audio"] if 'audio' in ann.keys() else vname.replace('mp4', 'wav')
        apath = os.path.join(self.aud_root, aname)
        frms = self.vis_processor(vpath)
        auds = self.aud_processor(apath)
        question = self.text_processor(ann["question"])

        return {
            "video": frms,
            "audio": auds, 
            "text_input": question,
            "answers": ann["answer"]
        }


class MOSEIDataset(AVQADataset):
    def __init__(self, vis_processor, text_processor, aud_processor, vis_root, aud_root, ann_paths):
        super().__init__(vis_processor, text_processor, aud_processor, vis_root, aud_root, ann_paths)
    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["video_id"]
        vpath = os.path.join(self.vis_root, vname+'.mp4')
        apath = os.path.join(self.aud_root, vname+'.wav')
        frms = self.vis_processor(vpath)
        auds = self.aud_processor(apath)
        texts = ann['subtitle']
        # question = self.text_processor(ann["question"])
        if ann['sent']>0:
            answer='positive'
        elif ann['sent']<0:
            answer='negative'
        else:
            answer='neutral'
        return {
            "video": frms,
            "audio": auds, 
            "text_input": texts,
            "answers": answer
        }