import torch
from skimage.metrics import structural_similarity as ssim
from SRGAN_model import *
from interpolators import Interpolator
from utils import yaml_read, transforms_init, quadra_imshow_cv
from datasetCustom import SRDataset
from imgLoss import PerceptionLoss
import pandas as pd
from tqdm import tqdm

from time import perf_counter
from contextlib import contextmanager
import warnings


@contextmanager
def catchtime() -> float:
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()


def psnr(img1: np.array, img2: np.array, max_i: float=1.0):
    diffs = np.mean((img1 - img2)**2)
    return 10*np.log10(max_i**2/diffs) if diffs > 0 else 0


def custom_generator_load(generator_path: str = r'runs/run/SRGAN_16blocks_2x.pth',
                          device: str = 'cpu'):
    gen_model = Generator()
    gen_info = torch.load(generator_path, map_location=device)
    gen_model.load_state_dict(gen_info['model_weights'])
    gen_model.eval()
    for p in gen_model.parameters():
        p.requires_grad = False
    return gen_model


def custom_interpolation_load(model_type: str, super_resolution: int):
    l = model_type.split()
    inter_algo = 'bicubic' if len(l) < 2 else l[0]
    model = Interpolator(int_algo=inter_algo,
                         super_resolution=super_resolution)
    return model


def models_out_comparison(test_phase: str, model1_type: str, model2_type: str, cfg: dict):
    test_phase_list = ['count_stats', 'show_res']
    interpolation_types = ['interpol', 'interpolation', 'bicubic interpolation',
                           'biliniar interpolation', 'nearest interpolation']
    gen_types = ['gan', 'generator', 'nn']
    str_info = lambda x: f'Specify wrong model{x} type \n Availables types={gen_types+interpolation_types}'

    assert model1_type in gen_types+interpolation_types, str_info(1)
    assert model2_type in gen_types+interpolation_types, str_info(2)
    assert test_phase in test_phase_list, f"Specify wrong test phase. Available phases={test_phase_list}"

    if model1_type in interpolation_types:
        model1 = custom_interpolation_load(model_type=model1_type,
                                           super_resolution=cfg['image_super_resolution'])
    else:
        model1 = custom_generator_load(generator_path=r'../runs/the_pretty_one/SRGAN_16blocks_2x.pth',
                                       device='cpu')

    if model2_type in interpolation_types:
        model2 = custom_interpolation_load(model_type=model2_type,
                                           super_resolution=cfg['image_super_resolution'])
    else:
        model2 = custom_generator_load(generator_path=r'../runs/the_pretty_one/SRGAN_16blocks_2x.pth',
                                       device='cpu')

    obj = SRDataset(root_dir=cfg['data_path_test'],
                    dir_in=cfg['test_in_dir'],
                    dir_out=cfg['test_out_dir'],
                    transforms=transforms_init(cfg=cfg))

    if test_phase == "count_stats":
        test_res = []
        loss = PerceptionLoss(num_feature_layer=cfg['num_vgg_layer_mse'], device='cpu')
        for op in tqdm(iter(obj), desc='Test process', total=len(obj)):
            inp, out = op
            numpy_output = out.permute(1, 2, 0).numpy()
            numpy_input = inp.permute(1, 2, 0).numpy()

            model1_out, _ = model_inference(model=model1,
                                             inp=inp[None, ...] if model1_type in gen_types else numpy_input,
                                             stats_inference_mode=True)
            model2_out, _ = model_inference(model=model2,
                                             inp=inp[None, ...] if model2_type in gen_types else numpy_input,
                                             stats_inference_mode=True)

            l1 = loss(out[None, ...], model1_out if model1_type in gen_types else torch.from_numpy(model1_out).permute(2, 0, 1)[None, ...])
            l2 = loss(out[None, ...], model2_out if model2_type in gen_types else torch.from_numpy(model2_out).permute(2, 0, 1)[None, ...])

            if model1_type in gen_types:
                model1_out = model1_out[0].permute(1, 2, 0).numpy()
            if model2_type in gen_types:
                model2_out = model2_out[0].permute(1, 2, 0).numpy()

            test_res.append([psnr(numpy_output, model1_out),
                             psnr(numpy_output, model2_out),
                             ssim(numpy_output, model1_out, multichannel=True).flat.base.item(),
                             ssim(numpy_output, model2_out, multichannel=True).flat.base.item(),
                             l1,
                             l2])

            img_paths = obj.imgs_out
            columns = [
                       f'{model1_type}_psnr', f'{model2_type}_psnr',
                       f'{model1_type}_ssim', f'{model2_type}_ssim',
                       f'{model1_type}_loss', f'{model2_type}_loss'
                       ]
        test_res = np.array(test_res)
        res_df = pd.DataFrame(test_res, columns=columns, index=img_paths)
        res_df.to_csv('./test.csv')

    elif test_phase == "show_res":
        ind_col_width = 16
        col_width = 14
        float_degree_round = 8

        for op in iter(obj):
            inp, out = op
            numpy_output = out.permute(1, 2, 0).numpy()
            numpy_input = inp.permute(1, 2, 0).numpy()

            model1_out, t1 = model_inference(model=model1,
                                            inp=inp[None, ...] if model1_type in gen_types else numpy_input,
                                            stats_inference_mode=False)
            model2_out, t2 = model_inference(model=model2,
                                            inp=inp[None, ...] if model2_type in gen_types else numpy_input,
                                            stats_inference_mode=False)

            if model1_type in gen_types:
                model1_out = model1_out[0].permute(1, 2, 0).numpy()
            if model2_type in gen_types:
                model2_out = model2_out[0].permute(1, 2, 0).numpy()

            print(f"|{'Models':{ind_col_width}}|{model1_type:{col_width}}|{model2_type:{col_width}}|")
            print('='*(ind_col_width+col_width+col_width+4))
            print(f"|{'Inference time':{ind_col_width}}|{round(t1(), float_degree_round):{col_width}}|"
                  f"{round(t2(), float_degree_round):{col_width}}|")
            print(f"|{'PSNR':{ind_col_width}}|{round(psnr(numpy_output, model1_out), float_degree_round):{col_width}}|"
                  f"{round(psnr(numpy_output, model2_out), float_degree_round):{col_width}}|")
            print(f"|{'SSIM':{ind_col_width}}|"
                  f"{round(ssim(numpy_output, model1_out, multichannel=True).flat.base.item(), float_degree_round):{col_width}}|"
                  f"{round(ssim(numpy_output, model2_out, multichannel=True).flat.base.item(), float_degree_round):{col_width}}|")
            print('='*(ind_col_width+col_width+col_width+4)+'\n')

            pic_caption = {'input_pic': inp.permute(1, 2, 0).numpy(),
                           'target_pic': numpy_output,
                           model1_type: model1_out,
                           model2_type: model2_out}

            quadra_imshow_cv(pic_caption)


def model_inference(model: object, inp: torch.Tensor, stats_inference_mode: bool):
    with torch.no_grad():
        if not stats_inference_mode:
            with catchtime() as t:
                model_out = model(inp)
        else:
            t = None
            model_out = model(inp)
    return model_out, t


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    train_config = yaml_read(yaml_path='../trainers/config.yaml')

    test_phase = "show_res" # "show_res" or "count_stats"
    models_out_comparison(test_phase=test_phase, model1_type='gan', model2_type='interpolation', cfg=train_config)
