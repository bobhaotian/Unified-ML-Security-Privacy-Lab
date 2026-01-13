"""
Neural Network Architectures

CVAE: Conditional Variational Autoencoder
    ELBO = E_{z~q}[log p(x|z,y)] - D_KL(q(z|x,y) || p(z))
    
    Encoder: q_φ(z|x,y) = N(μ_φ(x,y), diag(σ²_φ(x,y)))
    Decoder: p_θ(x|z,y) = Bernoulli(D_θ(z,y))
    Prior:   p(z) = N(0, I)

References:
    [1] Kingma & Welling, "Auto-Encoding Variational Bayes" (ICLR 2014)
    [2] Sohn et al., "Learning Structured Output Representation using DCGM" (NeurIPS 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Compact CNN classifier: f_θ : ℝ^{1×28×28} → ℝ^K
    
    Architecture: Conv(1→32→64→128) → FC(6272→256→K)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CVAE(nn.Module):
    """
    Conditional VAE for class-conditional generation.
    
    Encoder: (x, y) → (μ, log σ²)
    Decoder: (z, y) → x̂
    Reparameterization: z = μ + σ ⊙ ε, where ε ~ N(0, I)
    """
    def __init__(self, z_dim: int = 32, num_classes: int = 10):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes

        # Encoder q_φ(z|x, y)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),   # 28→14
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 14→7
        )
        self.enc_fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 + num_classes, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

        # Decoder p_θ(x|z, y)
        self.dec_fc = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256), nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7), nn.ReLU()
        )
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 7→14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid() # 14→28
        )

    def one_hot(self, y: torch.Tensor):
        return F.one_hot(y, num_classes=self.num_classes).float()

    def encode(self, x, y):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, self.one_hot(y)], dim=1)
        h = self.enc_fc(h)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        """z = μ + σ ⊙ ε, where σ = exp(log σ² / 2), ε ~ N(0, I)"""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h = torch.cat([z, self.one_hot(y)], dim=1)
        h = self.dec_fc(h)
        h = h.view(h.size(0), 64, 7, 7)
        return self.dec_deconv(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z, y)
        return xhat, mu, logvar


def cvae_loss(x, xhat, mu, logvar):
    """
    Negative ELBO: L = BCE(x, x̂) + D_KL(q||p)
    
    BCE = -Σ_{ij} [x_{ij} log x̂_{ij} + (1-x_{ij}) log(1-x̂_{ij})]
    D_KL = (1/2) Σ_j [μ_j² + σ_j² - log σ_j² - 1]
    """
    bce = F.binary_cross_entropy(xhat, x, reduction="sum") / x.size(0)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return bce + kl, {"bce": float(bce.item()), "kl": float(kl.item())}
