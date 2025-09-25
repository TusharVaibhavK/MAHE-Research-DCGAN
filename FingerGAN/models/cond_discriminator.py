import torch.nn as nn


class CondDiscriminator(nn.Module):
    def __init__(self, n_labels=2, ndf=64, nc=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Real/Fake (4x4 after 4 conv layers with stride 2)
        self.fc_real = nn.Linear(ndf*8*4*4, 1)
        self.fc_aux = nn.Linear(ndf*8*4*4, n_labels)   # Region classification

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        real_fake = self.fc_real(feat)
        region_logits = self.fc_aux(feat)
        return real_fake, region_logits
