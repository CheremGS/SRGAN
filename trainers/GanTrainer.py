import os
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from utils import check_folder_name, custom_save_model, save_plot_hist
from BaseTrainer import Trainer
from SRGAN_model import Generator, Discriminator
from imgLoss import PerceptionLoss


class GANTrainer(Trainer):
    def train_loop(self) -> None:
        save_model_path, save_hist_gen, save_hist_discr = self.init_save_path()
        genbb_weights = self.cfg['pretrain_gen_net']
        self.init_device()

        dataloader = self.init_dataloaders()
        gen_model = Generator(n_blocks=self.cfg['n_blocks'],
                              scaling_factor=self.cfg['upscale_factor']).to(self.device)

        try:
            srresnet = torch.load(genbb_weights)
            gen_model.load_state_dict(srresnet['model_weights'])
        except Exception as e:
            raise RuntimeError(f'Generator backbone wasnt load. Error ocurred: {e}')

        discr_model = Discriminator().to(self.device)

        perception_criterion = PerceptionLoss(device=self.device, num_feature_layer=self.cfg['num_vgg_layer_mse'])
        adversarial_criterion = nn.BCEWithLogitsLoss()

        discr_optimizer = self.init_optimizer(model=discr_model)
        discr_lr_scheduler = self.init_lr_scheduler(optimizer=discr_optimizer)

        gen_optimizer = self.init_optimizer(model=gen_model)
        gen_lr_scheduler = self.init_lr_scheduler(optimizer=gen_optimizer)

        discr_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['amp'])
        gen_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['amp'])
        best_loss = torch.inf
        train_state = None
        discr_train_hist = []
        gen_train_hist = []

        try:
            self.model_summary(model=gen_model)
            self.model_summary(model=discr_model)
            print('Model train process')
            for epoch in range(self.cfg['epochs']):
                gl, dl = self.train_step(data=dataloader,
                                         model=[gen_model, discr_model],
                                         criterion=[perception_criterion, adversarial_criterion],
                                         optimizer=[gen_optimizer, discr_optimizer],
                                         scaler=[gen_scaler, discr_scaler],
                                         lr_scheduler=[gen_lr_scheduler, discr_lr_scheduler],
                                         i_epoch=epoch)

                if gl < best_loss:
                    best_loss = gl
                    train_state = {'model_weights': gen_model.state_dict(),
                                   'gen_loss': gl,
                                   'discr_loss': dl,
                                   'epochs_train': epoch}

                discr_train_hist.append(dl)
                gen_train_hist.append(gl)
        finally:
            custom_save_model(save_state=train_state,
                              model_name=save_model_path,
                              cfg=self.cfg)
            save_plot_hist(hist=gen_train_hist,
                           plot_name=save_hist_gen)
            save_plot_hist(hist=discr_train_hist,
                           plot_name=save_hist_discr)

        torch.cuda.empty_cache()

    def train_step(self, data, model, optimizer, lr_scheduler, criterion, i_epoch, scaler,
                   model_profile=False) -> (float, float):
        # models, scalers, optimizers, lr_schedulers contain two objects:
        # for generator(index 0) and discriminator(index 1)
        # criterions: perception_loss(index 0) and adversarial_loss(index 1)
        model[0].train()
        model[1].train()
        pbar = tqdm(enumerate(data), total=len(data))
        gen_epoch_loss = 0.0
        discr_epoch_loss = 0.0
        for i, (lr_imgs, hr_imgs) in pbar:
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
                gen_imgs = model[0](lr_imgs)
                gen_labels = model[1](gen_imgs)

                perception_loss = criterion[0](gen_imgs, hr_imgs)
                adversarial_loss = criterion[1](gen_labels, torch.ones_like(gen_labels))
                perceptual_loss = perception_loss + self.cfg['beta'] * adversarial_loss

            gen_epoch_loss += perceptual_loss.item()
            scaler[0].scale(perceptual_loss).backward()
            if (i + 1) % self.cfg['accumulation_steps'] == 0:
                scaler[0].step(optimizer[0])
                scaler[0].update()
                lr_scheduler[0].step()
                optimizer[0].zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.cfg['amp']):
                hr_labels = model[1](hr_imgs)
                gen_labels = model[1](gen_imgs.detach())

                adversarial_loss = criterion[1](gen_labels, torch.zeros_like(gen_labels)) + \
                                   criterion[1](hr_labels, torch.ones_like(hr_labels))

            discr_epoch_loss += adversarial_loss.item()
            scaler[1].scale(adversarial_loss).backward()
            if (i + 1) % self.cfg['accumulation_steps'] == 0:
                scaler[1].step(optimizer[1])
                scaler[1].update()
                lr_scheduler[1].step()
                optimizer[1].zero_grad(set_to_none=True)

            if model_profile:
                break

            pbar.set_description(
                f"[{i_epoch + 1}/{self.cfg['epochs']}] Loss_D: {discr_epoch_loss/(pbar.n+1):.5f} "
                f"Loss_G: {gen_epoch_loss/(pbar.n+1):.5f} ")

        return gen_epoch_loss/len(data), discr_epoch_loss/len(data)

    def init_save_path(self) -> (str, str, str):
        model_name = f"SRGAN_{self.cfg['n_blocks']}blocks_{self.cfg['upscale_factor']}x.pth"
        hist_plot_discr = 'discr_train_losses.png'
        gen_plot_discr = 'gen_train_losses.png'

        save_run_path = check_folder_name(os.path.join(self.cfg['save_dir'], self.cfg['save_run_name']))

        save_model_path = os.path.join(save_run_path, model_name)
        save_hist_gen = os.path.join(save_run_path, gen_plot_discr)
        save_hist_discr = os.path.join(save_run_path, hist_plot_discr)

        return save_model_path, save_hist_gen, save_hist_discr

    def run(self):
        self.train_loop()