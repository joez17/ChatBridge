import os
from numpy import short
import torch 
import json
import argparse
from custom_datasets.valor_data import  VideoMapper, AudioMapper
from custom_datasets.valor_data.data import VALORDataset, valor_collate, valor_collate_stage2
from custom_datasets.valor_data.data_task import TASKDataset, task_collate
import torch.distributed as dist
from utils.distributed_sh import DistributedSampler_wopadding
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from time import time
# from data.data_webvid_web import WebvidFrameDataset
from custom_datasets.valor_data import  MetaLoader, PrefetchLoader , AccumMetaLoader

# from dataloader import KVReader
from easydict import EasyDict as edict
from torch.utils.data import ConcatDataset
# def kv_worker_init_fn(_):
#     worker_info = torch.utils.data.get_worker_info()
#     dataset = worker_info.dataset
#     # Avoid "cannot pickle KVReader object" error
#     if  isinstance(dataset, ConcatDataset):
#         for i in range(len(dataset.datasets)):
#             # if not dataset.datasets[i].data_type.startswith('image_coco'): ### use kv_reader
#             if not dataset.datasets[i].data_type.startswith('image'):
#                 dataset.datasets[i].video_mapper.kv_reader = KVReader(dataset.datasets[i].video_mapper.video_dir, 4)
          
#     else:
#         # if not dataset.data_type.startswith('image_coco'): ### use kv_reader
#         if not dataset.datasets[i].data_type.startswith('image'):
#             dataset.video_mapper.kv_reader = KVReader(dataset.video_mapper.video_dir, 4)



def get_rank(): return int(os.getenv('RANK', 0))
                

def create_train_dataloaders(valor_data_opts, stage=1):
    opts = edict(json.load(open(valor_data_opts)))

    
    data_cfg = opts.data_cfg.train
    dataloaders = []
    dataloaders_dict={}
    train_steps = []
    loader_names = []
    video_sample_num_ls = [1]
    audio_sample_num_ls = [1]
    for d_cfg in data_cfg:
        concate_name = ''
        dataset_ls = []



        # worker_init_fn = kv_worker_init_fn  if getattr(d_cfg,'data_format', 'kvreader')=='kvreader' else None
        worker_init_fn = None
        use_sampler = True


        for dset in d_cfg['datasets']:
            name = dset['name']
            concate_name = concate_name + name if concate_name == '' else concate_name + '_' + name
            assert dset['datatype'] in ['video','image','audio']
            if dset['datatype'] == 'audio':
                worker_init_fn = None
            data_type = dset['datatype'] + '_' + name
            
            task = d_cfg['task'].split('_') 
    
            video_mapper = None
            audio_mapper = None
            if 'video' in dset:
                video_path = dset['video']
                video_sample_num = d_cfg['video_sample_num'] if data_type.startswith('video') else 1
                video_sample_num_ls.append(video_sample_num)
                video_transforms =  dset.get('video_transforms','none')
                data_format =  getattr(d_cfg,'data_format', 'kvreader')
                video_mapper = VideoMapper(video_path, opts, data_type, video_sample_num, video_transforms, data_format)
            
            if 'audio' in dset:
                audio_path = dset['audio']
                audio_sample_num = d_cfg['audio_sample_num']
                audio_sample_num_ls.append(audio_sample_num)
                audio_mapper = AudioMapper(audio_path, opts, audio_sample_num)
                                

            dataset = VALORDataset(dset['txt'], video_mapper, audio_mapper, training=True)
            collate_fn = valor_collate_stage2 if stage==2 else valor_collate

            dataset.data_type = data_type
            print("Create Dataset {} Success".format(name))
            dataset_ls.append(dataset)
        dataset = ConcatDataset(dataset_ls)
        
        print("Create Dataset {} Success".format(concate_name))
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        n_workers = d_cfg['n_workers'] 

        if 'steps' in d_cfg:
            train_steps.append(d_cfg['steps'])
        else:
            epoch = d_cfg['epoch']
            train_steps.append(int((len(dataset) // batch_size) * epoch))

        loader = build_dataloader(dataset, collate_fn, True, batch_size, n_workers, worker_init_fn, use_sampler)

        dataloaders.append(loader)
        loader_names.append(f'{task}--{concate_name}')
    
    total_train_steps = sum(train_steps)
    for i in range(len(dataloaders)):
        ratio = train_steps[i]
        dataloaders_dict[loader_names[i]] = (dataloaders[i], ratio)

    n_gpu = dist.get_world_size()
    for name, (loader, ratio) in dataloaders_dict.items():
        # epoch = (ratio * loader.batch_size * n_gpu ) // len(loader.dataset)
        print(f" loader {name} , ratio {ratio} , bs_pergpu {loader.batch_size}, n_workers {loader.num_workers}" )

    if opts.dataset_mix_type == 'random':
        meta_loader = MetaLoader(dataloaders_dict,
                                accum_steps=opts.gradient_accumulation_steps,
                                distributed=n_gpu > 1)
        opts.num_train_steps = total_train_steps
    elif opts.dataset_mix_type in ['accum','round-robin']:
        assert opts.gradient_accumulation_steps == 1
        meta_loader = AccumMetaLoader(dataloaders_dict,
                                distributed=n_gpu > 1)
        
        
        
    meta_loader = PrefetchLoader(meta_loader)
    meta_loader.ndata = len(dataloaders_dict)
    meta_loader.ds_name = loader_names
    # opts.valid_steps = opts.num_train_steps // opts.valid_freq -1
    opts.video_sample_num = max(video_sample_num_ls)
    opts.audio_sample_num = max(audio_sample_num_ls)
    assert opts.video_sample_num > 0
    assert opts.audio_sample_num > 0
    
    return meta_loader


def build_dataloader(dataset, collate_fn, is_train, batch_size, n_workers=None, worker_init_fn=None, use_sampler=True):
    batch_size = batch_size // dist.get_world_size()
    if use_sampler:
        if is_train:
            sampler = DistributedSampler(dataset)
        else:
            sampler = DistributedSampler_wopadding(dataset)
        loader = DataLoader(dataset, sampler = sampler, batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train,worker_init_fn=worker_init_fn)
    else:

        loader = DataLoader(dataset,  batch_size = batch_size,
                            num_workers=n_workers, pin_memory=True,
                            collate_fn=collate_fn, drop_last=is_train,worker_init_fn=worker_init_fn)    

    return loader

def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
    else:
        raise Exception("Invalid Bool Value")

def create_task_dataloaders(valor_data_opts, prompt_path):
    opts = edict(json.load(open(valor_data_opts)))
    prompts = json.load(open(prompt_path))
    zh = getattr(opts, 'zh', False)
    data_cfg = opts.data_cfg.train
    dataloaders = []
    dataloaders_dict={}
    train_steps = []
    loader_names = []
    video_sample_num_ls = [1]
    audio_sample_num_ls = [1]
    for d_cfg in data_cfg:
        concate_name = ''
        dataset_ls = []
        worker_init_fn = None
        use_sampler = True


        for dset in d_cfg['datasets']:
            name = dset['name']
            concate_name = concate_name + name if concate_name == '' else concate_name + '_' + name
            assert dset['datatype'] in ['video','image','audio']
            if dset['datatype'] == 'audio':
                worker_init_fn = None
            data_type = dset['datatype'] + '_' + name
            
            task = d_cfg['task'].split('_') 
    
            video_mapper = None
            audio_mapper = None
            if 'video' in dset:
                video_path = dset['video']
                video_sample_num = d_cfg['video_sample_num'] if data_type.startswith('video') else 1
                video_sample_num_ls.append(video_sample_num)
                video_transforms =  dset.get('video_transforms','none')
                data_format =  getattr(d_cfg,'data_format', 'kvreader')
                video_mapper = VideoMapper(video_path, opts, data_type, video_sample_num, video_transforms, data_format)

            if 'audio' in dset:
                audio_path = dset['audio']
                audio_sample_num = d_cfg['audio_sample_num']
                audio_sample_num_ls.append(audio_sample_num)
                audio_mapper = AudioMapper(audio_path, opts, audio_sample_num)
                                

            dataset = TASKDataset(dset['txt'], video_mapper, audio_mapper, training=True, task=d_cfg['task'], prompts=prompts, zh=zh)
            collate_fn = task_collate

            dataset.data_type = data_type
            print("Create Dataset {} Success".format(name))
            dataset_ls.append(dataset)
        dataset = ConcatDataset(dataset_ls)
        
        print("Create Dataset {} Success".format(concate_name))
        task = d_cfg['task']
        batch_size = d_cfg['batch_size']
        n_workers = d_cfg['n_workers'] 

        if 'steps' in d_cfg:
            train_steps.append(d_cfg['steps'])
        else:
            epoch = d_cfg['epoch']
            train_steps.append(int((len(dataset) // batch_size) * epoch))

        loader = build_dataloader(dataset, collate_fn, True, batch_size, n_workers, worker_init_fn, use_sampler)

        dataloaders.append(loader)
        loader_names.append(f'{task}--{concate_name}')
    
    total_train_steps = sum(train_steps)
    for i in range(len(dataloaders)):
        ratio = train_steps[i]
        dataloaders_dict[loader_names[i]] = (dataloaders[i], ratio)

    n_gpu = dist.get_world_size()
    for name, (loader, ratio) in dataloaders_dict.items():
        # epoch = (ratio * loader.batch_size * n_gpu ) // len(loader.dataset)
        print(f" loader {name} , ratio {ratio} , bs_pergpu {loader.batch_size}, n_workers {loader.num_workers}" )

    if opts.dataset_mix_type == 'random':
        meta_loader = MetaLoader(dataloaders_dict,
                                accum_steps=opts.gradient_accumulation_steps,
                                distributed=n_gpu > 1)
        opts.num_train_steps = total_train_steps
    elif opts.dataset_mix_type in ['accum','round-robin']:
        assert opts.gradient_accumulation_steps == 1
        meta_loader = AccumMetaLoader(dataloaders_dict,
                                distributed=n_gpu > 1)
        
        
        
    meta_loader = PrefetchLoader(meta_loader)
    meta_loader.ndata = len(dataloaders_dict)
    meta_loader.ds_name = loader_names
    # opts.valid_steps = opts.num_train_steps // opts.valid_freq -1
    opts.video_sample_num = max(video_sample_num_ls)
    opts.audio_sample_num = max(audio_sample_num_ls)
    assert opts.video_sample_num > 0
    assert opts.audio_sample_num > 0
    
    return meta_loader
