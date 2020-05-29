from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(din, dout, 6, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(dout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


e_chanel = 64
d_chanel = 64
latent_dim = 512


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, e_chanel),  # (3,64,64) => (64,32,32)
            ConvBlock(e_chanel, 2 * e_chanel),  # (64,32,32)=> (128,16,16)
            ConvBlock(2 * e_chanel, 4 * e_chanel),  # (128,16,16) => (256,8,8)
            ConvBlock(4 * e_chanel, 8 * e_chanel),  # (256,8,8) => (512,4,4)
            Flatten(),
            nn.Linear(8 * e_chanel * (4) ** 2, latent_dim * 2),  # (512,4,4) => 1024
            nn.BatchNorm1d(latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * d_chanel * (8) ** 2),  # 512 = > 256*8*8
            Reshape(-1, 4 * d_chanel, 8, 8),  # => (256, 8, 8)
            nn.BatchNorm2d(4 * d_chanel), nn.ReLU(),
            DeconvBlock(4 * d_chanel, 4 * d_chanel),  # => (256, 16, 16)
            DeconvBlock(4 * d_chanel, 2 * d_chanel),  # => (128, 32, 32)
            DeconvBlock(2 * d_chanel, d_chanel // 2),  # => (32, 64, 64)
            nn.ConvTranspose2d(d_chanel // 2, 3, 5, padding=2),  # => (3, 64, 64)
            nn.Sigmoid()
        )

    def reparameterize(self, z):

        mu = z[:, :latent_dim]
        logvar = z[:, latent_dim:]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latent = self.encoder(x)
        latent_mu, latent_logvar = latent[:, :latent_dim], latent[:, latent_dim:]

        latent_code = self.reparameterize(latent)
        reconstructed = self.decoder(latent_code)
        return reconstructed, latent_mu, latent_logvar
