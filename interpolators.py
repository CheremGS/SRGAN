from utils import yaml_read, get_imgs_names
import cv2
import numpy as np


class Interpolator:
    INTER_ALGOS = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4
    }

    def __init__(self, int_algo: str = "bicubic",
                 super_resolution: int or (int, int) = 256):
        self.algo = int_algo
        self.super_res = super_resolution
        self.check_attributes()

    def check_attributes(self):
        avail_methods = list(self.INTER_ALGOS.keys())
        if self.algo not in avail_methods:
            RuntimeError(f'Specify wrong interpolator algorithm. Available algos: {avail_methods}')
        if isinstance(self.super_res, int):
            self.super_res = (self.super_res, self.super_res)

    def __call__(self, x: np.array):
        return cv2.resize(x, dsize=self.super_res, interpolation=self.INTER_ALGOS[self.algo])


if __name__ == "__main__":
    yaml_path = 'trainers/train_config.yaml'

    conf_data = yaml_read(yaml_path)
    img_dir = conf_data['data_path']
    img_names, imgs_number = get_imgs_names(img_dir)
    base_res = conf_data['image_base_resolution']
    super_res = conf_data['image_super_resolution']

    if isinstance(base_res, int):
        base_res = (base_res, base_res)
    if isinstance(super_res, int):
        super_res = (super_res, super_res)

    interpolation_algorithm = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4
    }

    interpolate_method = "bicubic"
    y_adds = (super_res[0]-base_res[0])//2
    x_adds = (super_res[1]-base_res[1])//2
    for img_path in img_names:
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        out_img = cv2.resize(img, dsize=super_res, interpolation=interpolation_algorithm[interpolate_method])
        show_pic = np.concatenate((out_img, cv2.copyMakeBorder(img, top=y_adds, bottom=y_adds,
                                                                    right=x_adds, left=x_adds,
                                                                    borderType=cv2.BORDER_CONSTANT)),
                                  axis=1)
        cv2.imshow(f'Out model and orig images', show_pic)
        cv2.waitKey(0)