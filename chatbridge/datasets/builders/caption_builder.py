"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from chatbridge.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from chatbridge.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from chatbridge.common.registry import registry
from chatbridge.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
    VISTCaptionEvalDataset,
    ValorCaptionEvalDataset
)
from chatbridge.datasets.datasets.audio_caption_datasets import AudioCaptionEvalDataset
import chatbridge.common.utils as utils
@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }


@registry.register_builder("vist_caption")
class VISTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VISTCaptionEvalDataset
    eval_dataset_cls = VISTCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vist/defaults_cap.yaml",
    }
import os
import warnings
@registry.register_builder("audio_caption")
class AudioCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = AudioCaptionEvalDataset
    eval_dataset_cls = AudioCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/clotho/defaults_cap.yaml",
    }
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        aud_info = build_info.get('audios')

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            aud_processor = (
                self.aud_processors["train"]
                if is_train
                else self.aud_processors["eval"]
            )
            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths


            # aud_path = vis_path.replace('all_videos', 'audio_22050hz')
            aud_path = aud_info.storage
            if not os.path.isabs(aud_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                aud_path = utils.get_cache_path(aud_path)

            if not os.path.exists(aud_path):
                warnings.warn("storage path {} does not exist.".format(aud_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                text_processor=text_processor,
                aud_processor=aud_processor,
                ann_paths=ann_paths,
                aud_root=aud_path
            )

        return datasets
    
@registry.register_builder("av_caption")
class AVCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ValorCaptionEvalDataset
    eval_dataset_cls = ValorCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/valor/defaults_cap.yaml",
    }
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)
        aud_info = build_info.get('audios')

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            aud_processor = (
                self.aud_processors["train"]
                if is_train
                else self.aud_processors["eval"]
            )
            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage
            # aud_path = vis_path.replace('all_videos', 'audio_22050hz')
            aud_path = aud_info.storage
            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                aud_processor=aud_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                aud_root=aud_path
            )

        return datasets