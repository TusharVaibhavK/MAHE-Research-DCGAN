import torch
import torch.nn as nn


class CondGenerator(nn.Module):
    def __init__(self, nz=100, n_labels=2, ngf=64, nc=1):
        super().__init__()
        self.nz = nz
        self.n_labels = n_labels
        # embedding for labels
        self.label_emb = nn.Embedding(n_labels, n_labels)

        input_dim = nz + n_labels
        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # labels: (B,)
        lbl = self.label_emb(labels)
        x = torch.cat([z, lbl], 1)  # (B, nz+n_labels)
        x = x.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        return self.net(x)
