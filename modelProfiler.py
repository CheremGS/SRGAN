import torch
from utils import yaml_read
from SRGAN_model import Generator, Discriminator
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    cfg = yaml_read(yaml_path='./trainers/config.yaml')
    Generator(n_blocks=cfg['n_blocks'],
              scaling_factor=cfg['upscale_factor']).to(cfg['device'])