from utils import yaml_read
from GenPretrainer import GeneratorTrainer
from GanTrainer import GANTrainer

if __name__ == "__main__":

    train_modes = ['gen_pretrain', 'gan_train']
    mode = 1
    if mode == 0:
        config = yaml_read(yaml_path='train_config.yaml')
        genTrain = GeneratorTrainer(cfg=config)
        genTrain.run()
    elif mode == 1:
        config = yaml_read(yaml_path='train_config.yaml')
        ganTrain = GANTrainer(cfg=config)
        ganTrain.run()
    else:
        raise RuntimeError('Specify wrong train mode. Set 0 or 1 value in "mode" variable!')

