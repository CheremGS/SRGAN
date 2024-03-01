### Implemented features
- Trainers for pretrain generator model and GAN 
- Loggers for generator pretrain and GAN train
- Inference with comparison models (interpolation and gan)
- Dataset for GAN train and generator pretrain
- Adversarial and perception loss
- Configuration with yaml file
- Save train runs info in various folders 


#### ToDo:
- Implement thop: count params, layers, flops
- Advanced clean cuda cache (https://www.youtube.com/watch?v=6rWrKGH6suo)
- Implement pytorch profiler
- Implement new gan model and approach