from utils import yaml_read
from GenPretrainer import GeneratorTrainer
from GanTrainer import GANTrainer
import warnings


def main(mode: str = 'gan_train') -> None:
    config = yaml_read(yaml_path='config.yaml')
    if mode.lower() == 'pretrain':
        gen_pretrainer = GeneratorTrainer(cfg=config)
        gen_pretrainer.run()
    elif mode.lower() == 'gan_train':
        gan_trainer = GANTrainer(cfg=config)
        gan_trainer.run()
    else:
        RuntimeError(f'Specify wrong train mode. (Available modes: "pretrain" or "gan_train")')


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    mod = 'gan_train'
    main(mode=mod)



