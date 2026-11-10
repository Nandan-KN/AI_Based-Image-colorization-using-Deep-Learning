import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNColorizationModel(nn.Module):
    """
    Encoder-Decoder CNN Architecture matching models/cnn_best.pth.
    Takes 1-channel Lightness (L in [0, 1]) and predicts 2-channel chrominance (AB in [-1, 1]).
    """
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def AB_pred(self, L):
        x = self.forward(L)
        return x[:, 0, :, :], x[:, 1, :, :]


if __name__ == "__main__":
    model = CNNColorizationModel()
    L = torch.randn(2, 1, 256, 256)
    AB_pred = model(L)
    print("Input L:", L.shape)
    print("CNN output AB:", AB_pred.shape)
