import cv2
import torch

from SRGAN_model import *
from interpolators import Interpolator
from utils import yaml_read, cv_imshow, double_imshow_cv, transforms_init, quadra_imshow_cv
from datasetCustom import SRDataset


train_config = yaml_read(yaml_path='./trainers/train_config.yaml')
super_resolution = 256


gen_model = Generator()
generator_path = r'runs/the_best_one/SRGAN_16blocks_2x.pth'
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

    input_pic = inp.permute(1, 2, 0).numpy()
    target_pic = out.permute(1, 2, 0).numpy()
    model1_interpolator_pic = interpol(inp.permute(1, 2, 0).numpy())
    model2_gan_pic = generator_res[0].permute(1, 2, 0).numpy()

    pic_caption = {'input_pic': input_pic,
                   'target_pic': target_pic,
                   'interpolate_pic': model1_interpolator_pic,
                   'gan_pic': model2_gan_pic}

    quadra_imshow_cv(pic_caption)

