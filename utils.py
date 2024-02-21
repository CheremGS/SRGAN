import yaml
import os
import cv2
import gc
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


def custom_save_model(save_state: dict, model_name: str) -> None:
    """Save torch model and del cache"""
    print('Model train is over')
    if save_state is not None:
        print(f'Model was saved in {model_name}')
        try:
            torch.save(save_state, model_name)
        except Exception as e:
            print(f'Model wasnt save. Error occurred: \n{e}')
    else:
        print('Model fitting was interrupted too early. Model wasnt save.')

    empty_cache()


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def save_plot_hist(hist: list, plot_name: str) -> None:
    label = os.path.basename(plot_name)[:-4]
    plt.plot(np.arange(len(hist)), np.array(hist), label=label)
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)


def check_folder_name(save_dir_path: str) -> str:
    # checkout: folder exists and its not empty, then we change folder name
    path_len = len(save_dir_path)
    i = 1
    while os.path.isdir(save_dir_path) and (os.listdir(save_dir_path)):
        save_dir_path = save_dir_path[:path_len] + str(i)
        i += 1
    os.makedirs(save_dir_path, exist_ok=True)
    return save_dir_path


def global_seed(determ: bool = False) -> None:
    SEED = torch.initial_seed()%2**32
    torch.manual_seed(SEED)
    if determ:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def yaml_read(yaml_path: str = './config.yaml') -> dict:
    assert os.path.isfile(yaml_path), "Wrong specified yaml path"
    with open(yaml_path, 'r') as yaml_stream:
        fi = yaml.safe_load(yaml_stream)
    return fi


def cv_imshow(img_path: str, transforms=None) -> None:
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    if transforms:
        pass
    cv2.imshow(f'image - {os.path.basename(img_path)}', img)
    cv2.waitKey(0)


def get_imgs_names(img_dir: str) -> (list, int):
    assert os.path.isdir(img_dir), 'Wrong imgs dir path'
    img_names = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
    imgs_number = len(img_names)
    return img_names, imgs_number


def img_concat_cv(img1: np.array, img2: np.array, y_pad: int, x_pad: int, axis: int = 1) -> np.array:
    if (x_pad == 0) and (y_pad == 0):
        return np.concatenate((img1, img2), axis=axis)
    else:
        return np.concatenate((img1, cv2.copyMakeBorder(img2, top=y_pad, bottom=y_pad,
                                                            right=x_pad, left=x_pad,
                                                            borderType=cv2.BORDER_CONSTANT,
                                                            value=0.)), axis=axis)


def double_imshow_cv(img1: np.array, img2: np.array) -> None:
    x_adds, y_adds = img_padding(img1, img2)
    if img1.shape[0] < img2.shape[0]:
        show_pic = img_concat_cv(img1=img2, img2=img1, y_pad=y_adds, x_pad=x_adds)
    else:
        show_pic = img_concat_cv(img1=img1, img2=img2, y_pad=y_adds, x_pad=x_adds)

    cv2.imshow(f'image 1 and image 2', show_pic)
    cv2.waitKey(0)


def img_padding(img1:np.array, img2:np.array):
    y_adds = (img1.shape[0] - img2.shape[0]) // 2
    x_adds = (img1.shape[1] - img2.shape[1]) // 2
    if img1.shape[0] < img2.shape[0]:
        y_adds *= -1
        x_adds *= -1
    return x_adds, y_adds


def quadra_imshow_cv(pics: dict) -> None:
    assert pics['target_pic'].shape == pics['interpolate_pic'].shape == pics['gan_pic'].shape, \
            'Models outputs and target image have different sizes'

    img_input = pics['input_pic']
    img_target = pics['target_pic']
    img_model1 = pics['interpolate_pic']
    img_model2 = pics['gan_pic']

    x_adds, y_adds = img_padding(img_input, img_target)
    in_out_img_pic = img_concat_cv(img1=img_target, img2=img_input, y_pad=y_adds, x_pad=x_adds)
    models_pic = img_concat_cv(img1=img_model1, img2=img_model2, y_pad=0, x_pad=0)
    quadra_pic = img_concat_cv(in_out_img_pic, models_pic, axis=0, y_pad=0, x_pad=0)

    caption_shift = 20

    quadra_pic = cv2.putText(quadra_pic, 'input', (quadra_pic.shape[1] // 2 + caption_shift,
                                                   caption_shift),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    quadra_pic = cv2.putText(quadra_pic, 'target', (caption_shift,
                                                    caption_shift),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    quadra_pic = cv2.putText(quadra_pic, 'interpolation', (caption_shift,
                                                           quadra_pic.shape[1] // 2 + caption_shift),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    quadra_pic = cv2.putText(quadra_pic, 'gan', (quadra_pic.shape[1] // 2 + caption_shift,
                                                 quadra_pic.shape[1] // 2 + caption_shift),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow(f'inference comparison', quadra_pic)
    cv2.waitKey(0)


def transforms_init(cfg: dict) -> list:
    l = cfg['image_base_resolution']
    transf = [A.Compose([A.Resize(width=l, height=l, interpolation=cv2.INTER_NEAREST),
                         A.Normalize(max_pixel_value=255.,
                                     mean=(0, 0, 0),
                                     std=(1, 1, 1)),
                         ToTensorV2()]),
              None]
    return transf







