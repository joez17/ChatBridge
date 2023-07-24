"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import chatbridge.tasks as tasks
from chatbridge.common.config import Config
from chatbridge.common.dist_utils import get_rank, init_distributed_mode
from chatbridge.common.logger import setup_logger
from chatbridge.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from chatbridge.common.registry import registry
from chatbridge.common.utils import now

# imports modules for registration
from chatbridge.datasets.builders import *
from chatbridge.models import *
from chatbridge.processors import *
from chatbridge.runners import *
from chatbridge.tasks import *
import wandb
import torch.distributed as dist
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()
    if is_main_process():
        wandb_key = getattr(cfg.run_cfg, 'wandb_key', None)
        if wandb_key:
            os.environ['WANDB_API_KEY'] = wandb_key
            wandb.init(project=cfg.run_cfg.output_dir.split('/')[-2],
                name=cfg.run_cfg.output_dir.split('/')[-1],
                entity='zjzhao',
                config=cfg,
                resume=cfg.run_cfg.output_dir.split('/')[-1])
    task = tasks.setup_task(cfg)
    # datasets = task.build_datasets(cfg)
    dataloader = task.build_dataloader(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=None, dataloader=dataloader
    )
    runner.train()


if __name__ == "__main__":
    main()
