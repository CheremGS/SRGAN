import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from datasetCustom import SRDataset
from SRGAN_model import Generator
from utils import yaml_read, transforms_init, global_seed


def main(cfg: dict):
    device = cfg['device'].lower()
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
    pixel_mse = nn.MSELoss()

    # toDo: all optimizer components must be contained in cfg
    optimizer = torch.optim.AdamW(params=[x for x in gen_model.parameters() if x.requires_grad],
                                  lr=cfg['lr'],
                                  weight_decay=1e-6,
                                  amsgrad=True)
    scaler = torch.cuda.amp.GradScaler()
    train_state = None
    best_loss = torch.inf
    model_save_path = f'./pth_models/SRresnet{cfg["n_blocks"]}_{cfg["upscale_factor"]}upscale.pth'
    # train generator without validation
    # discriminator can suppress generator on early epochs
    # such pretrain allow avoid that case
    try:
        gen_model.train()
        for epoch in range(cfg['start_epoch'], cfg['epochs']):
            pbar = tqdm(dataloader, total=len(dataloader))
            avg_loss = 0.0
            for low_res_imgs, high_res_imgs in pbar:
                x = low_res_imgs.to(device, non_blocking=True)
                y = high_res_imgs.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    sr = gen_model(x)
                    loss = pixel_mse(sr, y)

                if (epoch - cfg['start_epoch']+1) % cfg['accumulation_steps'] == 0:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                avg_loss += loss.item()
                pbar.set_description(f"[{epoch}/{cfg['epochs']}] MSE loss: {loss.item():.4f}")

            if ((epoch + 1) % cfg['checkpoint_steps'] == 0) and (avg_loss < best_loss):
                best_loss = avg_loss
                train_state = {'model_weights': gen_model.state_dict(),
                               'epoch': epoch,
                               'loss': avg_loss}

        else:
            train_state = {'model_weights': gen_model.state_dict(),
                           'epoch': epoch,
                           'loss': avg_loss}

    except KeyboardInterrupt:
        print('Generator backbone train is over')
        if train_state is not None:
            print(f'Model was saved in {model_save_path}')
            torch.save(train_state, model_save_path)
        else:
            print('Model fitting was interrupted too early. Model wasnt save.')
    else:
        print(f'Model was saved in {model_save_path}')
        torch.save(train_state, model_save_path)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    cfg = yaml_read()
    main(cfg=cfg)




