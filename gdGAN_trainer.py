import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from datasetCustom import SRDataset
from SRGAN_model import Generator, Discriminator
from utils import global_seed, yaml_read, transforms_init
from imgLoss import PerceptionLoss


def main(cfg):
    # toDo: этот код повторяется из обучения скелета (вынести куда-то общий код)
    genbb_weights = f'./pth_models/SRresnet{cfg["n_blocks"]}_{cfg["upscale_factor"]}upscale.pth'
    gengan_weights = f"./pth_models/SRGAN_{cfg['n_blocks']}blocks_{cfg['upscale_factor']}x.pth"
    device = cfg['device'].lower()
    beta = cfg['beta']
    global_seed(cfg['deterministic'])

    dataset = SRDataset(root_dir=cfg['data_path_train'],
                        dir_in=cfg['train_in_dir'],
                        dir_out=cfg['train_out_dir'],
                        transforms=transforms_init(cfg=cfg))

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                             batch_size=cfg['batch_size'],
                                             pin_memory=True,
                                             num_workers=cfg['workers'])

    gen_model = Generator(n_blocks=cfg['n_blocks'],
                          scaling_factor=cfg['upscale_factor']).to(device)
    # ----------------------- общий код -------------------

    srresnet = torch.load(genbb_weights)
    gen_model.load_state_dict(srresnet['model_weights'])
    discr_model = Discriminator().to(device)

    # инициализируем loss-ы
    # toDo: add num parameter in cfg
    perception_criterion = PerceptionLoss(device=device, num_feature_layer=cfg['num_vgg_layer_mse'])
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # переводим в режим обучения
    gen_model.train()
    discr_model.train()

    # toDo: create different params for each optimizator
    gen_optimizer = torch.optim.AdamW(params=[x for x in gen_model.parameters() if x.requires_grad],
                                      lr=cfg['lr'],
                                      weight_decay=cfg['weight_decay'],
                                      amsgrad=True)
    discr_optimizer = torch.optim.AdamW(params=[x for x in discr_model.parameters() if x.requires_grad],
                                      lr=cfg['lr'],
                                      weight_decay=cfg['weight_decay'],
                                      amsgrad=True)

    discr_scaler = torch.cuda.amp.GradScaler()
    gen_scaler = torch.cuda.amp.GradScaler()
    train_state = None
    try:
        for epoch in range(cfg['start_epoch'], cfg['epochs']):
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            gen_epoch_loss = 0.0
            discr_epoch_loss = 0.0
            for i, (lr_imgs, hr_imgs) in progress_bar:
                lr_imgs = lr_imgs.to(device, non_blocking=True)
                hr_imgs = hr_imgs.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    gen_imgs = gen_model(lr_imgs)
                    gen_labels = discr_model(gen_imgs)

                    perception_loss = perception_criterion(gen_imgs, hr_imgs)
                    adversarial_loss = adversarial_criterion(gen_labels, torch.ones_like(gen_labels))
                    perceptual_loss = perception_loss + beta * adversarial_loss

                if (i + 1) % cfg['accumulation_steps'] == 0:
                    gen_scaler.scale(perceptual_loss).backward()
                    gen_scaler.step(gen_optimizer)
                    gen_scaler.update()
                    gen_optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    hr_labels = discr_model(hr_imgs)
                    gen_labels = discr_model(gen_imgs.detach())

                    adversarial_loss = adversarial_criterion(gen_labels, torch.zeros_like(gen_labels)) + \
                                       adversarial_criterion(hr_labels, torch.ones_like(hr_labels))

                if (i+1) % cfg['accumulation_steps'] == 0:
                    discr_scaler.scale(adversarial_loss).backward()
                    discr_scaler.step(discr_optimizer)
                    discr_scaler.update()
                    discr_optimizer.zero_grad(set_to_none=True)

                discr_epoch_loss += adversarial_loss.item()
                gen_epoch_loss += perceptual_loss.item()

                progress_bar.set_description(f"[{epoch + 1}/{cfg['epochs']}] Loss_D: {adversarial_loss.item():.4f} Loss_G: {perceptual_loss.item():.4f} ")

            if (epoch + 1) % cfg['checkpoint_steps'] == 0:
                train_state = {'model_weights': gen_model.state_dict(),
                               'gen_loss': gen_epoch_loss,
                               'discr_loss': discr_epoch_loss}
    except KeyboardInterrupt:
        print('Generator backbone train is over')
        if train_state is not None:
            print(f'Model was saved in {gengan_weights}')
            torch.save(train_state, gengan_weights)
        else:
            print('Model fitting was interrupted too early. Model wasnt save.')
    else:
        print(f'Model was saved in {gengan_weights}')
        torch.save(train_state, gengan_weights)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    cfg = yaml_read()
    main(cfg=cfg)
