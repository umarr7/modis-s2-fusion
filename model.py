import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),   # 2 MODIS bands
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=4),  # upsample 32→128
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1)  # 2 Sentinel bands (B4, B8)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.up(x)
        x = self.decoder(x)
        return x
