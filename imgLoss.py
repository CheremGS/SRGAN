from torch import nn, Tensor
from torchvision.models import vgg19


class PerceptionLoss(nn.Module):
    def __init__(self, device: str, num_feature_layer: int):
        super().__init__()

        self.encoder = nn.Sequential(*list(vgg19(pretrained=True).modules())[2:num_feature_layer+2])
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.to(device)

    def forward(self, x: Tensor, y: Tensor):
        x_features = self.encoder(x)
        y_features = self.encoder(y)

        loss = nn.functional.mse_loss(input=x_features, target=y_features)
        return loss


