"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from chatbridge.common.registry import registry
from chatbridge.tasks.base_task import BaseTask
from chatbridge.common.logger import MetricLogger, SmoothedValue
import logging
from chatbridge.datasets.data_utils import prepare_sample
import torch
from chatbridge.tasks.valor_train_utils import create_train_dataloaders
@registry.register_task("tri_pretrain")
class TriPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass
    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            ds_name, samples = next(data_loader)
            task, ds_name = ds_name.split('--')
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                    "ds_name": ds_name,
                    "task": task
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
            # optimizer.zero_grad()
            # aq = model.module.audio_query_tokens.clone().detach()
            # iq = model.module.image_query_tokens.clone().detach()
            # aw = model.module.aud_Qformer.bert.encoder.layer[0].output_query.dense.weight.clone().detach()
            # iw = model.module.Qformer.bert.encoder.layer[0].output_query.dense.weight.clone().detach()
            # loss.backward()
            # print('pre audio_query_diff:', (model.module.audio_query_tokens-aq).abs().sum())
            # print('pre image_query_diff:', (model.module.image_query_tokens-iq).abs().sum())
            # print('pre audio_qformer_diff:', (model.module.aud_Qformer.bert.encoder.layer[0].output_query.dense.weight-aw).abs().sum())
            # print('pre image_qformer_diff:', (model.module.Qformer.bert.encoder.layer[0].output_query.dense.weight-iw).abs().sum())
            # optimizer.step()
            # # optimizer.zero_grad()
            # print('===============grad======================')
            # print('audio_query_grad:', model.module.audio_query_tokens.grad.abs().sum() if model.module.audio_query_tokens.grad is not None else None )
            # print('image_query_grad:', model.module.image_query_tokens.grad.abs().sum() if model.module.image_query_tokens.grad is not None else None )
            # print('audio_qformer_all_grad:', sum(p.grad.abs().sum() if p.grad is not  None else 0 for p in model.module.aud_Qformer.parameters()))
            # print('image_qformer_all_grad:', sum(p.grad.abs().sum() if p.grad is not  None else 0 for p in model.module.Qformer.parameters()))

            # print('--------------diff----------------------')
            # print('audio_query_diff:', (model.module.audio_query_tokens-aq).abs().sum())
            # print('image_query_diff:', (model.module.image_query_tokens-iq).abs().sum())
            # print('audio_qformer_diff:', (model.module.aud_Qformer.bert.encoder.layer[0].output_query.dense.weight-aw).abs().sum())
            # print('image_qformer_diff:', (model.module.Qformer.bert.encoder.layer[0].output_query.dense.weight-iw).abs().sum())
            # import ipdb; ipdb.set_trace()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update_dict(f"loss_{task}%{ds_name}", loss)

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def build_dataloader(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dataloader
        """
        # valor_data_opts_path = 'lavis/tasks/valor_data.json'
        valor_data_opts_path = cfg.run_cfg.valor_data_opts_path
        stage = getattr(cfg.run_cfg, 'stage', 1)
        loader = create_train_dataloaders(valor_data_opts=valor_data_opts_path, stage=stage)
        self.loss_names = getattr(cfg.run_cfg, 'losses', ['itm', 'itc', 'lm'])
        # for ds in loader:
        #     import pdb
        #     pdb.set_trace()
        #     print(ds)

        return loader