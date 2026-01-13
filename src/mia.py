"""
Membership Inference Attack (MIA)

Given target model f_θ and sample (x, y), predict membership m ∈ {0, 1}.

References:
    [1] Shokri et al., "Membership Inference Attacks Against ML Models" (S&P 2017)
    [2] Salem et al., "ML-Leaks: Model and Data Independent MIA" (NDSS 2019)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def collect_attack_features(model, x, y, device):
    """
    Extract MIA features φ(x, y, f_θ) ∈ ℝ^5:
    
        φ = [L, conf, H, margin, correct]
    
    where:
        L       = -log p_θ(y|x)                     (cross-entropy loss)
        conf    = max_k p_θ(k|x)                    (confidence)
        H       = -Σ_k p_θ(k|x) log p_θ(k|x)        (entropy)
        margin  = p_θ(ŷ₁|x) - p_θ(ŷ₂|x)            (top-2 gap)
        correct = 𝟙[argmax_k p_θ(k|x) = y]         (correctness)
    """
    model.eval()
    x, y = x.to(device), y.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    
    loss = F.cross_entropy(logits, y, reduction="none")
    p_sorted, _ = probs.sort(dim=1, descending=True)
    p1, p2 = p_sorted[:, 0], p_sorted[:, 1]
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=1)
    correct = (logits.argmax(dim=1) == y).float()
    
    feats = torch.stack([loss, p1, entropy, (p1 - p2), correct], dim=1).cpu().numpy()
    return feats


def threshold_attack_correctness(correct_in: np.ndarray, correct_out: np.ndarray):
    """
    Threshold attack: A(x, y) = 𝟙[f_θ(x) correctly classifies y].
    """
    in_pred = (correct_in >= 0.5).astype(int)
    out_pred = (correct_out >= 0.5).astype(int)
    y_true = np.concatenate([np.ones_like(in_pred), np.zeros_like(out_pred)])
    y_hat = np.concatenate([in_pred, out_pred])
    return float((y_true == y_hat).mean())


def train_attack_model(shadow_in_feats, shadow_out_feats):
    """
    Train attack classifier A_ω: P(member|φ) = σ(ω^T φ + b).
    """
    X = np.vstack([shadow_in_feats, shadow_out_feats])
    y = np.concatenate([np.ones(len(shadow_in_feats)), np.zeros(len(shadow_out_feats))])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    return clf


def eval_attack_model(clf, target_in_feats, target_out_feats):
    """
    Evaluate attack model. AUC = 0.5 indicates no leakage.
    """
    X = np.vstack([target_in_feats, target_out_feats])
    y = np.concatenate([np.ones(len(target_in_feats)), np.zeros(len(target_out_feats))])
    proba = clf.predict_proba(X)[:, 1]
    y_hat = (proba >= 0.5).astype(int)
    acc = float((y_hat == y).mean())
    auc = float(roc_auc_score(y, proba))
    return {"acc": acc, "auc": auc, "proba": proba, "y": y}
