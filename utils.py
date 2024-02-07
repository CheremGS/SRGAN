import yaml
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


def img_concat_cv(img1: np.array, img2: np.array, y_pad: int, x_pad:int) -> np.array:
    return np.concatenate((img1, cv2.copyMakeBorder(img2, top=y_pad, bottom=y_pad,
                                                        right=x_pad, left=x_pad,
                                                        borderType=cv2.BORDER_CONSTANT,
                                                        value=0.)), axis=1)


def double_imshow_cv(img1: np.array, img2: np.array) -> None:
    y_adds = (img1.shape[0] - img2.shape[0]) // 2
    x_adds = (img1.shape[1] - img2.shape[1]) // 2
    if img1.shape[0] < img2.shape[0]:
        y_adds *= -1
        x_adds *= -1
        show_pic = img_concat_cv(img1=img2, img2=img1, y_pad=y_adds, x_pad=x_adds)
    else:
        show_pic = img_concat_cv(img1=img1, img2=img2, y_pad=y_adds, x_pad=x_adds)

    cv2.imshow(f'Out model and orig images', show_pic)
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







