from chatbridge.models.beats.BEATs import  BEATsConfig, BEATs
import torch
def create_beat(): 
    checkpoint = torch.load('/mnt/bn/sihanchen/tasks/VLP3/pretrained_weights/BEATs_iter3_plus_AS2M.pt')
    cfg = BEATsConfig(checkpoint['cfg'])
    audio_encoder = BEATs(cfg)
    audio_encoder.load_state_dict(checkpoint['model'])
    audio_dim = 768
    return audio_encoder, audio_dim