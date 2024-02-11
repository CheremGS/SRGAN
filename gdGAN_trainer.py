import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import os

from train_utils import *
from SRGAN_model import Generator, Discriminator
from utils import global_seed, yaml_read, \
                  custom_save_model, save_plot_hist, check_folder_name
from imgLoss import PerceptionLoss


def main(cfg):
    model_name = f"SRGAN_{cfg['n_blocks']}blocks_{cfg['upscale_factor']}x.pth"
    hist_plot_discr = 'discr_train_losses.png'
    gen_plot_discr = 'gen_train_losses.png'

    save_run_path = check_folder_name(os.path.join(cfg['save_dir'], cfg['save_run_name']))

    save_model_path = os.path.join(save_run_path, model_name)
    save_plot_hist_gen = os.path.join(save_run_path, gen_plot_discr)
    save_plot_hist_discr = os.path.join(save_run_path, hist_plot_discr)

    # toDo: этот код повторяется из обучения скелета (вынести куда-то общий код)
    genbb_weights = f'./runs/run/SRresnet{cfg["n_blocks"]}_{cfg["upscale_factor"]}upscale.pth'
    device = init_device(cfg=cfg)
    beta = cfg['beta']
    global_seed(cfg['deterministic'])

    dataloader = generate_dataloader(cfg=cfg)
    gen_model = Generator(n_blocks=cfg['n_blocks'],
                          scaling_factor=cfg['upscale_factor']).to(device)

    srresnet = torch.load(genbb_weights)
    gen_model.load_state_dict(srresnet['model_weights'])
    discr_model = Discriminator().to(device)

    perception_criterion = PerceptionLoss(device=device, num_feature_layer=cfg['num_vgg_layer_mse'])
    adversarial_criterion = nn.BCEWithLogitsLoss()

    discr_optimizer = init_optimizator(cfg=cfg, model=discr_model)
    discr_lr_scheduler = init_lr_scheduler(cfg=cfg, optimizer=discr_optimizer)

    gen_optimizer = init_optimizator(cfg=cfg, model=gen_model)
    gen_lr_scheduler = init_lr_scheduler(cfg=cfg, optimizer=gen_optimizer)

    scaler_enabled = cfg['amp']
    discr_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    gen_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    train_state = None
    discr_train_hist = []
    gen_train_hist = []

    try:
        gen_model.train()
        discr_model.train()
        for epoch in range(cfg['epochs']):
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            gen_epoch_loss = 0.0
            discr_epoch_loss = 0.0
            for i, (lr_imgs, hr_imgs) in progress_bar:
                lr_imgs = lr_imgs.to(device, non_blocking=True)
                hr_imgs = hr_imgs.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=scaler_enabled):
                    gen_imgs = gen_model(lr_imgs)
                    gen_labels = discr_model(gen_imgs)

                    perception_loss = perception_criterion(gen_imgs, hr_imgs)
                    adversarial_loss = adversarial_criterion(gen_labels, torch.ones_like(gen_labels))
                    perceptual_loss = perception_loss + beta * adversarial_loss

                gen_epoch_loss += perceptual_loss.item()
                gen_scaler.scale(perceptual_loss).backward()
                if (i+1) % cfg['accumulation_steps'] == 0:
                    gen_scaler.step(gen_optimizer)
                    gen_scaler.update()
                    gen_lr_scheduler.step()
                    gen_optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=scaler_enabled):
                    hr_labels = discr_model(hr_imgs)
                    gen_labels = discr_model(gen_imgs.detach())

                    adversarial_loss = adversarial_criterion(gen_labels, torch.zeros_like(gen_labels)) + \
                                       adversarial_criterion(hr_labels, torch.ones_like(hr_labels))

                discr_epoch_loss += adversarial_loss.item()
                discr_scaler.scale(adversarial_loss).backward()
                if (i+1) % cfg['accumulation_steps'] == 0:
                    discr_scaler.step(discr_optimizer)
                    discr_scaler.update()
                    discr_lr_scheduler.step()
                    discr_optimizer.zero_grad(set_to_none=True)

                progress_bar.set_description(f"[{epoch + 1}/{cfg['epochs']}] Loss_D: {adversarial_loss.item():.4f} Loss_G: {perceptual_loss.item():.4f} ")

            if (epoch + 1) % cfg['checkpoint_steps'] == 0:
                train_state = {'model_weights': gen_model.state_dict(),
                               'gen_loss': gen_epoch_loss,
                               'discr_loss': discr_epoch_loss}

            discr_train_hist.append(discr_epoch_loss)
            gen_train_hist.append(gen_epoch_loss)
    finally:
        custom_save_model(save_state=train_state,
                          model_name=save_model_path)
        save_plot_hist(hist=gen_train_hist,
                       plot_name=save_plot_hist_gen)
        save_plot_hist(hist=discr_train_hist,
                       plot_name=save_plot_hist_discr)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    cfg = yaml_read()
    main(cfg=cfg)
