import os
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from train_utils import *
from SRGAN_model import Generator
from utils import yaml_read, transforms_init, global_seed, \
                    check_folder_name, custom_save_model, save_plot_hist


def main(cfg: dict):
    model_name = f'SRresnet{cfg["n_blocks"]}_{cfg["upscale_factor"]}upscale.pth'
    hist_plot = 'train_mse_losses.png'

    device = init_device(cfg=cfg)
    global_seed(cfg['deterministic'])

    save_run_path = check_folder_name(os.path.join(cfg['save_dir'], cfg['save_run_name']))

    save_model_path = os.path.join(save_run_path, model_name)
    save_plot_hist_path = os.path.join(save_run_path, hist_plot)

    dataloader = generate_dataloader(cfg)

    gen_model = Generator(n_blocks=cfg['n_blocks'],
                          scaling_factor=cfg['upscale_factor']).to(device)
    pixel_mse = nn.MSELoss()

    optimizer = init_optimizator(cfg=cfg, model=gen_model)
    lr_scheduler = init_lr_scheduler(cfg=cfg, optimizer=optimizer)

    amp_enabled = cfg['amp']
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_state = None
    best_loss = torch.inf
    train_hist = []
    # train generator without validation
    # discriminator can suppress generator on early epochs
    # such pretrain allow avoid that case
    try:
        optimizer.zero_grad()
        gen_model.train()
        for epoch in range(cfg['epochs']):
            pbar = tqdm(dataloader, total=len(dataloader))
            avg_loss = 0.0
            for low_res_imgs, high_res_imgs in pbar:
                x = low_res_imgs.to(device, non_blocking=True)
                y = high_res_imgs.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    sr = gen_model(x)
                    loss = pixel_mse(sr, y)

                avg_loss += loss.item()
                scaler.scale(loss).backward()
                if (epoch+1) % cfg['accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                pbar.set_description(f"[{epoch}/{cfg['epochs']}] MSE loss: {loss.item():.4f}")

            if ((epoch + 1) % cfg['checkpoint_steps'] == 0) and (avg_loss < best_loss):
                best_loss = avg_loss
                train_state = {'model_weights': gen_model.state_dict(),
                               'epoch': epoch,
                               'loss': avg_loss}
            train_hist.append(avg_loss)
    finally:
        custom_save_model(save_state=train_state,
                          model_name=save_model_path)
        save_plot_hist(hist=train_hist,
                       plot_name=save_plot_hist_path)


if __name__ == "__main__":
    cfg = yaml_read()
    main(cfg=cfg)




