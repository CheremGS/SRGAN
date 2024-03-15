from skimage.metrics import structural_similarity as ssim
from SRGAN_model import *
from interpolators import Interpolator
from utils import yaml_read, transforms_init, quadra_imshow_cv
from datasetCustom import SRDataset
import time


def psnr(img1: np.array, img2: np.array, max_i: float=1.0):
    diffs = np.mean((img1 - img2)**2)
    return 10*np.log10(max_i**2/diffs) if diffs > 0 else 0


if __name__ == "__main__":
    train_config = yaml_read(yaml_path='trainers/config.yaml')

    gen_model = Generator()
    generator_path = r'runs/the_pretty_one/SRGAN_16blocks_2x.pth'
    gen_info = torch.load(generator_path)
    gen_model.load_state_dict(gen_info['model_weights'])
    gen_model.eval()

    interpolator_algorithm = 'bicubic'
    interpol = Interpolator(int_algo=interpolator_algorithm,
                            super_resolution=train_config['image_super_resolution'])

    obj = SRDataset(root_dir=train_config['data_path_test'],
                    dir_in=train_config['test_in_dir'],
                    dir_out=train_config['test_out_dir'],
                    transforms=transforms_init(cfg=train_config))

    for op in iter(obj):
        inp, out = op
        with torch.no_grad():
            st = time.time()
            generator_res = gen_model(inp[None, ...])
            et = time.time() - st

        input_pic = inp.permute(1, 2, 0).numpy()
        target_pic = out.permute(1, 2, 0).numpy()
        model1_interpolator_pic = interpol(inp.permute(1, 2, 0).numpy())
        model2_gan_pic = generator_res[0].permute(1, 2, 0).numpy()

        print(f'Inference time={et}')
        print(f"Model 1: PSNR={psnr(target_pic, model1_interpolator_pic)}, "
              f"SSIM={ssim(target_pic, model1_interpolator_pic, multichannel=True)}")
        print(f"Model 2: PSNR={psnr(target_pic, model2_gan_pic)}, "
              f"SSIM={ssim(target_pic, model2_gan_pic, multichannel=True)}\n")

        pic_caption = {'input_pic': input_pic,
                       'target_pic': target_pic,
                       'interpolate_pic': model1_interpolator_pic,
                       'gan_pic': model2_gan_pic}

        quadra_imshow_cv(pic_caption)

