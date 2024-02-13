import os
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from utils import check_folder_name, custom_save_model, save_plot_hist
from BaseTrainer import Trainer
from SRGAN_model import Generator


class GeneratorTrainer(Trainer):
    def init_save_path(self) -> (str, str):
        model_name = f'SRresnet{self.cfg["n_blocks"]}_{self.cfg["upscale_factor"]}upscale.pth'
        hist_plot = 'train_mse_losses.png'

        # device = self.init_device()
        # global_seed(cfg['deterministic'])

        save_run_path = check_folder_name(os.path.join(self.cfg['save_dir'], self.cfg['save_run_name']))

        save_model_path = os.path.join(save_run_path, model_name)
        save_plot_hist_path = os.path.join(save_run_path, hist_plot)
        return save_model_path, save_plot_hist_path

    def train_loop(self):
        self.init_device()
        save_model_path, save_plot_hist_path = self.init_save_path()
        dataloader = self.init_dataloaders()

        gen_model = Generator(n_blocks=self.cfg['n_blocks'],
                              scaling_factor=self.cfg['upscale_factor']).to(self.device)
        pixel_mse = nn.MSELoss()

        optimizer = self.init_optimizer(model=gen_model)
        lr_scheduler = self.init_lr_scheduler(optimizer=optimizer)

        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['amp'])
        train_state = None
        best_loss = torch.inf
        train_hist = []
        # train generator without validation
        # discriminator can suppress generator on early epochs
        # such pretrain allow avoid that case
        try:
            for epoch in range(self.cfg['epochs']):
                avg_losses = self.train_step(model=gen_model,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             criterion=pixel_mse,
                                             i_epoch=epoch,
                                             data=dataloader,
                                             scaler=scaler)
                if ((epoch + 1) % self.cfg['checkpoint_steps'] == 0) and (avg_losses < best_loss):
                    best_loss = avg_losses
                    train_state = {'model_weights': gen_model.state_dict(),
                                   'epoch': epoch,
                                   'loss': avg_losses}
                train_hist.append(avg_losses)
        finally:
            custom_save_model(save_state=train_state,
                              model_name=save_model_path)
            save_plot_hist(hist=train_hist,
                           plot_name=save_plot_hist_path)

    def train_step(self, data, model, optimizer, lr_scheduler, criterion, i_epoch, scaler):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(data, total=len(data))
        avg_loss = 0.0
        for low_res_imgs, high_res_imgs in pbar:
            x = low_res_imgs.to(self.device, non_blocking=True)
            y = high_res_imgs.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
                sr = model(x)
                loss = criterion(sr, y)

            avg_loss += loss.item()
            scaler.scale(loss).backward()
            if (i_epoch + 1) % self.cfg['accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            pbar.set_description(f"[{i_epoch}/{self.cfg['epochs']}] MSE loss: {loss.item():.4f}")

        return avg_loss/len(data)