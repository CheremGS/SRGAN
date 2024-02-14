import cv2
import torch

from SRGAN_model import *
from interpolators import Interpolator
from utils import yaml_read, cv_imshow, double_imshow_cv, transforms_init
from datasetCustom import SRDataset


train_config = yaml_read(yaml_path='./trainers/train_config.yaml')
super_resolution = 256


gen_model = Generator()
generator_path = r'./runs/run1/SRGAN_16blocks_2x.pth'
gen_info = torch.load(generator_path)
gen_model.load_state_dict(gen_info['model_weights'])
gen_model.eval()

interpolator_algorithm = 'bicubic'
interpol = Interpolator(int_algo=interpolator_algorithm,
                        super_resolution=super_resolution)

obj = SRDataset(root_dir=train_config['data_path_train'],
                dir_in=train_config['train_in_dir'],
                dir_out=train_config['train_out_dir'],
                transforms=transforms_init(cfg=train_config))

for op in iter(obj):
    inp, out = op
    with torch.no_grad():
        generator_res = gen_model(inp[None, ...])

    inter_res = interpol(inp.permute(1, 2, 0).numpy())
    double_imshow_cv(inter_res, generator_res[0].permute(1, 2, 0).numpy())
    # inp, out = inp.permute(1, 2, 0).numpy(), out.permute(1, 2, 0).numpy()
    # double_imshow_cv(inp, out)

