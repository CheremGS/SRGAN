from torch import optim, cuda
from torch.utils.data import DataLoader
from utils import transforms_init
from datasetCustom import SRDataset


def generate_dataloader(cfg: dict):
    dataset = SRDataset(root_dir=cfg['data_path_train'],
                        dir_in=cfg['train_in_dir'],
                        dir_out=cfg['train_out_dir'],
                        transforms=transforms_init(cfg=cfg))

    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=cfg['batch_size'],
                            pin_memory=True,
                            num_workers=cfg['workers'])
    return dataloader


def init_optimizator(cfg: dict, model: object):
    algorithm = cfg['opt_algo'].lower()
    if algorithm == 'adamw':
        optimizer = optim.AdamW(params=[x for x in model.parameters() if x.requires_grad],
                           lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'],
                           amsgrad=True)
    elif algorithm == 'sgd':
        optimizer = optim.SGD(params=[x for x in model.parameters() if x.requires_grad],
                              lr=cfg['lr'],
                              momentum=cfg['momentum'],
                              nesterov=True,
                              weight_decay=cfg['weight_decay'])
    else:
        raise ValueError(f'Specified wrong optimizer type. Availables optims ["adamw", "sgd"]')
    return optimizer


def init_lr_scheduler(cfg: dict, optimizer: object):
    lr_scheduler = cfg['sched_type'].lower()
    if lr_scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg['lr_step'],
                                              gamma=cfg['lr_gamma'])
    elif lr_scheduler == "cosinelr":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         eta_min=cfg['lr_min'])
    elif lr_scheduler == "explr":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=cfg['lr_gamma'])
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{cfg['sched_type']}'. Only StepLR (steplr), CosineAnnealingLR (cosinelr) and ExponentialLR (explr)"
            "are supported."
        )
    return scheduler


def init_device(cfg: dict):
    gpus_avail = cuda.is_available()
    if gpus_avail and cfg['device'] == 'cuda':
        dev = 'cuda'
    elif (cfg['device'] is None) or (cfg['device'] == 'cpu'):
        dev = 'cpu'
    else:
        raise ValueError('wrong type device was specified. Available types: "cuda", "cpu"')
    return dev