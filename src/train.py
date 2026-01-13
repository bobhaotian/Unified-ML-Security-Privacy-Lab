"""
Training Procedures

ERM: min_θ (1/n) Σ_i L(f_θ(x_i), y_i)

Adversarial Training [1]: min_θ (1/n) Σ_i max_{||δ||_∞ ≤ ε} L(f_θ(x_i + δ), y_i)

DP-SGD [2]: Gradient clipping + Gaussian noise for differential privacy
    g̃ = clip(g, C) + N(0, σ²C²I)

References:
    [1] Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)
    [2] Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _forward_eval(model, x, y):
    logits = model(x)
    loss = F.cross_entropy(logits, y, reduction="sum").item()
    acc = (logits.argmax(dim=1) == y).float().sum().item()
    return loss, acc


def eval_classifier(model, loader, device, attack_fn=None):
    """Evaluate classifier on clean or adversarial examples."""
    model.eval()
    total_acc, total_loss, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if attack_fn is not None:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                x_adv = attack_fn(model, x, y)
            x_eval = x_adv.detach()
        else:
            x_eval = x

        with torch.no_grad():
            loss, acc = _forward_eval(model, x_eval, y)

        total_loss += loss
        total_acc += acc
        n += y.size(0)

    return {"loss": total_loss / n, "acc": total_acc / n}


def train_classifier(model, train_loader, test_loader, device, epochs=5, lr=1e-3, weight_decay=1e-4):
    """
    Standard ERM training.
    
    Objective: min_θ (1/n) Σ_i L(f_θ(x_i), y_i) + (λ/2)||θ||²
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, tot_acc, n = 0.0, 0.0, 0

        for x, y in tqdm(train_loader, desc=f"train ep {ep}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            tot_loss += loss.item() * y.size(0)
            tot_acc += (logits.argmax(dim=1) == y).float().sum().item()
            n += y.size(0)

        train_metrics = {"loss": tot_loss / n, "acc": tot_acc / n}
        test_metrics = eval_classifier(model, test_loader, device)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])

    return history


def dp_train_classifier(model, train_loader, test_loader, device, epochs=5, lr=1e-3, weight_decay=1e-4,
                        clip_norm=1.0, noise_multiplier=1.0):
    """
    DP-SGD style training (empirical, no formal ε-δ accounting).
    
    1. Clip gradient: g ← g · min(1, C/||g||₂)
    2. Add noise: g̃ ← g + N(0, (σC/B)² I)
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, tot_acc, n = 0.0, 0.0, 0

        for x, y in tqdm(train_loader, desc=f"dp-train ep {ep}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            # Clip gradient by L2 norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            # Add Gaussian noise
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    noise = torch.randn_like(p.grad) * (noise_multiplier * clip_norm / x.size(0))
                    p.grad.add_(noise)

            opt.step()

            tot_loss += loss.item() * y.size(0)
            tot_acc += (logits.argmax(dim=1) == y).float().sum().item()
            n += y.size(0)

        train_metrics = {"loss": tot_loss / n, "acc": tot_acc / n}
        test_metrics = eval_classifier(model, test_loader, device)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])

    return history


def adversarial_train_pgd(model, train_loader, test_loader, device, epochs=5, lr=1e-3, weight_decay=1e-4,
                          eps=0.2, alpha=0.03, steps=5):
    """
    Adversarial training with PGD.
    
    Objective: min_θ (1/n) Σ_i max_{||δ||_∞ ≤ ε} L(f_θ(x_i + δ), y_i)
    """
    from .attacks import pgd
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, tot_acc, n = 0.0, 0.0, 0

        for x, y in tqdm(train_loader, desc=f"pgd-train ep {ep}", leave=False):
            x, y = x.to(device), y.to(device)

            # Generate adversarial examples
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                adv = pgd(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True).detach()

            # Train on adversarial examples
            opt.zero_grad(set_to_none=True)
            logits = model(adv)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            tot_loss += loss.item() * y.size(0)
            tot_acc += (logits.argmax(dim=1) == y).float().sum().item()
            n += y.size(0)

        train_metrics = {"loss": tot_loss / n, "acc": tot_acc / n}
        test_metrics = eval_classifier(model, test_loader, device)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])

    return history
