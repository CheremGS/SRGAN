# 64 to 128 image reconstruction
import os
import numpy as np
import cv2
from tqdm import tqdm
from utils import yaml_read, cv_imshow, get_imgs_names


if __name__ == "__main__":
    yaml_path = './config.yaml'
    show_img = True
    size_stat = False
    transforms = None

    conf_data = yaml_read(yaml_path)
    img_dir = conf_data['data_path']
    img_names, imgs_number = get_imgs_names(img_dir)

    img_data = np.zeros(shape=(imgs_number, 2))

    if size_stat:
        for i_file in tqdm(range(imgs_number), desc='Collect image size data'):
            img = cv2.imread(img_names[i_file], cv2.COLOR_BGR2RGB)
            img_data[i_file, 0] = img.shape[0]
            img_data[i_file, 1] = img.shape[1]

        print(f'Height all image: mean={img_data[:, 0].mean()}, std={img_data[:, 0].std()}')
        print(f'Width all image: mean={img_data[:, 1].mean()}, std={img_data[:, 1].std()}')

    if show_img:
        img_inds = np.random.choice(np.arange(imgs_number), 10)
        for i in img_inds:
            cv_imshow(img_names[i])
