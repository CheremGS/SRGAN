import cv2

from torch import optim, cuda
from torch.utils.data import DataLoader
from datasetCustom import SRDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = None

    def init_save_path(self):
        pass

    def train_loop(self):
        pass

    def train_step(self, data, model, optimizer, lr_scheduler, criterion, i_epoch, scaler) -> float or (float, float):
        pass

    def run(self):
        self.train_loop()

    def transforms_init(self) -> list:
        l = self.cfg['image_base_resolution']
        transf = [A.Compose([A.Resize(width=l, height=l, interpolation=cv2.INTER_NEAREST),
                             A.Normalize(max_pixel_value=255.,
                                         mean=(0, 0, 0),
                                         std=(1, 1, 1)),
                             ToTensorV2()]),
                  A.Compose([
                             A.Normalize(max_pixel_value=255.,
                                         mean=(0, 0, 0),
                                         std=(1, 1, 1)),
                              ToTensorV2()]
                            )
                  ]
        return transf

    def init_dataloaders(self):
        dataset = SRDataset(root_dir=self.cfg['data_path_train'],
                            dir_in=self.cfg['train_in_dir'],
                            dir_out=self.cfg['train_out_dir'],
                            transforms=self.transforms_init())

        dataloader = DataLoader(dataset,
                                shuffle=True,
                                batch_size=self.cfg['batch_size'],
                                pin_memory=True,
                                num_workers=self.cfg['workers'])
        return dataloader

    def init_optimizer(self, model):
        algorithm = self.cfg['opt_algo'].lower()
        if algorithm == 'adamw':
            optimizer = optim.AdamW(params=[x for x in model.parameters() if x.requires_grad],
                                    lr=self.cfg['lr'],
                                    weight_decay=self.cfg['weight_decay'],
                                    amsgrad=True)
        elif algorithm == 'sgd':
            optimizer = optim.SGD(params=[x for x in model.parameters() if x.requires_grad],
                                  lr=self.cfg['lr'],
                                  momentum=self.cfg['momentum'],
                                  nesterov=True,
                                  weight_decay=self.cfg['weight_decay'])
        else:
            raise ValueError(f'Specified wrong optimizer type. Availables optims ["adamw", "sgd"]')
        return optimizer

    def init_lr_scheduler(self, optimizer):
        lr_scheduler = self.cfg['sched_type'].lower()
        if lr_scheduler == "steplr":
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.cfg['lr_step'],
                                                  gamma=self.cfg['lr_gamma'])
        elif lr_scheduler == "cosinelr":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=self.cfg['cosine_lr_period'],
                                                             eta_min=self.cfg['lr_min'])
        elif lr_scheduler == "explr":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                         gamma=self.cfg['lr_gamma'])
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{self.cfg['sched_type']}'. Only StepLR (steplr), CosineAnnealingLR (cosinelr) "
                f"and ExponentialLR (explr) are supported.")
        return scheduler

    def init_device(self):
        gpus_avail = cuda.is_available()
        if gpus_avail and self.cfg['device'] == 'cuda':
            self.device = 'cuda'
        elif (self.cfg['device'] is None) or (self.cfg['device'] == 'cpu'):
            self.device = 'cpu'
        else:
            raise ValueError('wrong type device was specified. Available types: "cuda", "cpu"')