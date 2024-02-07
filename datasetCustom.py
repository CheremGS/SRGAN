import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from torch.utils.data import Dataset
from utils import yaml_read, double_imshow_cv, transforms_init


class SRDataset(Dataset):
    def __init__(self, root_dir, dir_in: str, dir_out: str, transforms: list = [None, None]):
        self.dir_path_in = os.path.join(root_dir, dir_in)
        self.dir_path_out = os.path.join(root_dir, dir_out)
        self.transforms = transforms

        self.imgs_in = [os.path.join(self.dir_path_in, x) for x in os.listdir(self.dir_path_in)]
        self.imgs_out = [os.path.join(self.dir_path_out, x) for x in os.listdir(self.dir_path_out)]

    def __len__(self):
        return len(os.listdir(self.dir_path_in))

    def __getitem__(self, index):
        in_img = self.imgs_in[index]
        out_img = self.imgs_out[index]

        inp = cv2.imread(in_img, cv2.IMREAD_COLOR).astype(dtype=np.float32)
        out = cv2.imread(out_img, cv2.IMREAD_COLOR).astype(dtype=np.float32)

        for i in range(len(self.transforms)):
            if self.transforms[i] is None:
                self.transforms[i] = A.Compose([
                                                A.Normalize(max_pixel_value=255.,
                                                            mean=(0, 0, 0),
                                                            std=(1, 1, 1)),
                                                ToTensorV2()])

        inp = self.transforms[0](image=inp)['image']
        out = self.transforms[1](image=out)['image']

        return inp, out


if __name__ == "__main__":
    cfg = yaml_read()
    obj = SRDataset(root_dir = cfg['data_path_train'],
                    dir_in = cfg['train_in_dir'],
                    dir_out = cfg['train_out_dir'],
                    transforms=transforms_init(cfg=cfg))

    for op in iter(obj):
        i, o = op
        i, o = i.permute(1, 2, 0).numpy(), o.permute(1, 2, 0).numpy()
        double_imshow_cv(i, o)



