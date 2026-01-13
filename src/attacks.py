"""
Adversarial Attack Implementations (ℓ∞ threat model)

References:
    [1] Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015)
    [2] Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)
"""

import torch
import torch.nn.functional as F


def fgsm(model, x, y, eps: float):
    """
    Fast Gradient Sign Method (FGSM) [1].
    
    x_adv = x + ε · sign(∇_x L(f_θ(x), y))
    
    where L is cross-entropy and ||x_adv - x||_∞ ≤ ε.
    """
    x = x.detach().clone()
    x.requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    # δ = ε · sign(∇_x L)
    adv = x + eps * x.grad.sign()
    return adv.detach().clamp(0.0, 1.0)


def pgd(model, x, y, eps: float, alpha: float, steps: int, random_start: bool = True):
    """
    Projected Gradient Descent (PGD) attack [2].
    
    Iteratively solves: max_{||δ||_∞ ≤ ε} L(f_θ(x + δ), y)
    
    Update rule:
        δ^{(t+1)} = Π_{B_∞(x,ε)}[δ^{(t)} + α · sign(∇_x L)]
    
    where Π projects onto the ℓ∞-ball of radius ε centered at x.
    """
    x0 = x.detach()
    if random_start:
        # δ^{(0)} ~ Uniform(-ε, ε)
        adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        adv = x0.clone()

    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model(adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            # Gradient ascent: δ ← δ + α · sign(∇_x L)
            adv = adv + alpha * adv.grad.sign()
            # Projection: Π_{B_∞(x, ε)}
            adv = torch.max(torch.min(adv, x0 + eps), x0 - eps)
            adv = adv.clamp(0.0, 1.0)
        adv = adv.detach()
    return adv
