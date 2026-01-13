import argparse, os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from .data import seed_all, get_device, load_fashion_mnist, dataset_to_tensors
from .models import SmallCNN, CVAE, cvae_loss
from .train import train_classifier, dp_train_classifier, adversarial_train_pgd, eval_classifier
from .attacks import fgsm, pgd
from .mia import collect_attack_features, threshold_attack_correctness, train_attack_model, eval_attack_model
from .gmm_em import fit_diag_gmm, loglik_diag_gmm

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def plot_curves(xs, ys_dict, title, xlabel, ylabel, outpath):
    plt.figure()
    for k, y in ys_dict.items():
        plt.plot(xs, y, label=k)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def run_gmm_baseline(train_ds, test_ds, K=5, pca_dim=30, seed=0, max_per_class=2000):
    xtr, ytr = dataset_to_tensors(train_ds, max_items=min(len(train_ds), max_per_class * 10))
    xte, yte = dataset_to_tensors(test_ds, max_items=min(len(test_ds), 2000))
    Xtr = xtr.view(xtr.size(0), -1).numpy()
    Xte = xte.view(xte.size(0), -1).numpy()
    ytr = ytr.numpy()
    yte = yte.numpy()

    pca = PCA(n_components=pca_dim, random_state=seed)
    Xtr_p = pca.fit_transform(Xtr)
    Xte_p = pca.transform(Xte)

    priors = np.bincount(ytr, minlength=10).astype(np.float64)
    priors /= priors.sum()

    models = {}
    for c in range(10):
        idx = np.where(ytr == c)[0]
        Xc = Xtr_p[idx]
        if len(Xc) > max_per_class:
            rng = np.random.default_rng(seed + c)
            Xc = Xc[rng.choice(len(Xc), size=max_per_class, replace=False)]
        models[c] = fit_diag_gmm(Xc, K=K, seed=seed + c)

    logpost = []
    for c in range(10):
        ll = loglik_diag_gmm(Xte_p, models[c])
        logpost.append(ll + np.log(priors[c] + 1e-12))
    logpost = np.stack(logpost, axis=1)
    yhat = logpost.argmax(axis=1)
    acc = float((yhat == yte).mean())
    return {"gmm_test_acc": acc}

def train_cvae(train_loader, device, epochs=10, z_dim=32, lr=1e-3):
    cvae = CVAE(z_dim=z_dim).to(device)
    opt = torch.optim.Adam(cvae.parameters(), lr=lr)
    hist = {"loss": [], "bce": [], "kl": []}
    for _ in range(epochs):
        cvae.train()
        tot, tot_bce, tot_kl, n = 0.0, 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            xhat, mu, logvar = cvae(x, y)
            loss, parts = cvae_loss(x, xhat, mu, logvar)
            loss.backward()
            opt.step()
            bs = y.size(0)
            tot += loss.item() * bs
            tot_bce += parts["bce"] * bs
            tot_kl += parts["kl"] * bs
            n += bs
        hist["loss"].append(tot / n)
        hist["bce"].append(tot_bce / n)
        hist["kl"].append(tot_kl / n)
    return cvae, hist

@torch.no_grad()
def cvae_recon_errors(cvae, x, y, device):
    cvae.eval()
    x, y = x.to(device), y.to(device)
    xhat, _, _ = cvae(x, y)
    mse = ((xhat - x) ** 2).view(x.size(0), -1).mean(dim=1)
    return mse.detach().cpu().numpy()

def save_samples_cvae(cvae, device, outpath, n_per_class=8, z_dim=32):
    cvae.eval()
    zs, ys = [], []
    for c in range(10):
        z = torch.randn(n_per_class, z_dim, device=device)
        y = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        zs.append(z); ys.append(y)
    z = torch.cat(zs, dim=0)
    y = torch.cat(ys, dim=0)
    with torch.no_grad():
      x = cvae.decode(z, y).detach().cpu()


    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class, 10))
    idx = 0
    for r in range(10):
        for c in range(n_per_class):
            axes[r, c].imshow(x[idx, 0], cmap="gray")
            axes[r, c].axis("off")
            idx += 1
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def run_membership_inference(target_model, shadow_models, train_ds, test_ds, device, results_dir, max_points=3000):
    xin, yin = dataset_to_tensors(train_ds, max_items=min(len(train_ds), max_points))
    xout, yout = dataset_to_tensors(test_ds, max_items=min(len(test_ds), max_points))

    # Build one shadow split from train_ds
    n_shadow_total = min(len(train_ds), max_points)
    perm = torch.randperm(n_shadow_total)
    half = n_shadow_total // 2
    shadow_train_idx = perm[:half].tolist()
    shadow_out_idx = perm[half:2*half].tolist()

    shadow_in_x = torch.stack([train_ds[i][0] for i in shadow_train_idx], dim=0)
    shadow_in_y = torch.tensor([train_ds[i][1] for i in shadow_train_idx], dtype=torch.long)
    shadow_out_x = torch.stack([train_ds[i][0] for i in shadow_out_idx], dim=0)
    shadow_out_y = torch.tensor([train_ds[i][1] for i in shadow_out_idx], dtype=torch.long)

    shadow_in_feats, shadow_out_feats = [], []
    for sm in shadow_models:
        shadow_in_feats.append(collect_attack_features(sm, shadow_in_x, shadow_in_y, device))
        shadow_out_feats.append(collect_attack_features(sm, shadow_out_x, shadow_out_y, device))
    shadow_in_feats = np.vstack(shadow_in_feats)
    shadow_out_feats = np.vstack(shadow_out_feats)

    clf = train_attack_model(shadow_in_feats, shadow_out_feats)

    target_in_feats = collect_attack_features(target_model, xin, yin, device)
    target_out_feats = collect_attack_features(target_model, xout, yout, device)
    learned = eval_attack_model(clf, target_in_feats, target_out_feats)

    # Simple threshold attack
    thresh_acc = threshold_attack_correctness(target_in_feats[:, -1], target_out_feats[:, -1])

    # ROC curve
    fpr, tpr, _ = roc_curve(learned["y"], learned["proba"])
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Membership Inference ROC (learned attacker)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mia_roc.png"), dpi=160)
    plt.close()

    return {"mia_threshold_acc": thresh_acc, "mia_learned_acc": learned["acc"], "mia_learned_auc": learned["auc"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--cvae_epochs", type=int, default=10)
    ap.add_argument("--shadow_models", type=int, default=2)
    ap.add_argument("--eps_list", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    args = ap.parse_args()

    seed_all(args.seed)
    device = get_device()

    results_dir = os.path.abspath("results")
    ensure_dir(results_dir)

    limit_train = 12000 if args.quick else None
    limit_test = 3000 if args.quick else None
    epochs = 3 if args.quick else args.epochs
    cvae_epochs = 6 if args.quick else args.cvae_epochs

    train_ds, test_ds, train_loader, test_loader = load_fashion_mnist(
        batch_size=args.batch_size, limit_train=limit_train, limit_test=limit_test
    )

    # 1) Standard training
    std = SmallCNN().to(device)
    hist_std = train_classifier(std, train_loader, test_loader, device, epochs=epochs)
    std_test = eval_classifier(std, test_loader, device)

    # 2) PGD adversarial training
    adv = SmallCNN().to(device)
    hist_adv = adversarial_train_pgd(adv, train_loader, test_loader, device, epochs=epochs, eps=0.2, alpha=0.03, steps=5)
    adv_test = eval_classifier(adv, test_loader, device)

    # 3) DP-ish training
    dp = SmallCNN().to(device)
    hist_dp = dp_train_classifier(dp, train_loader, test_loader, device, epochs=epochs, clip_norm=1.0, noise_multiplier=1.0)
    dp_test = eval_classifier(dp, test_loader, device)

    # Curves
    xs = list(range(1, epochs + 1))
    plot_curves(xs, {"standard": hist_std["test_acc"], "pgd-train": hist_adv["test_acc"], "dp-train": hist_dp["test_acc"]},
                "Test Accuracy vs Epoch", "epoch", "accuracy", os.path.join(results_dir, "test_acc_vs_epoch.png"))
    plot_curves(xs, {"standard": hist_std["test_loss"], "pgd-train": hist_adv["test_loss"], "dp-train": hist_dp["test_loss"]},
                "Test Loss vs Epoch", "epoch", "loss", os.path.join(results_dir, "test_loss_vs_epoch.png"))

    # Robustness eval
    eps_list = args.eps_list

    def fgsm_attack(eps):
        return lambda m, x, y: fgsm(m, x, y, eps=eps)

    def pgd_attack(eps):
        alpha = max(1e-3, eps / 6)
        return lambda m, x, y: pgd(m, x, y, eps=eps, alpha=alpha, steps=10, random_start=True)

    robust = {"fgsm": {}, "pgd": {}}
    for eps in eps_list:
        robust["fgsm"][eps] = {
            "standard": eval_classifier(std, test_loader, device, attack_fn=fgsm_attack(eps))["acc"],
            "pgd-train": eval_classifier(adv, test_loader, device, attack_fn=fgsm_attack(eps))["acc"],
            "dp-train": eval_classifier(dp, test_loader, device, attack_fn=fgsm_attack(eps))["acc"],
        }
        robust["pgd"][eps] = {
            "standard": eval_classifier(std, test_loader, device, attack_fn=pgd_attack(eps))["acc"],
            "pgd-train": eval_classifier(adv, test_loader, device, attack_fn=pgd_attack(eps))["acc"],
            "dp-train": eval_classifier(dp, test_loader, device, attack_fn=pgd_attack(eps))["acc"],
        }

    plot_curves(eps_list,
                {"standard": [robust["fgsm"][e]["standard"] for e in eps_list],
                 "pgd-train": [robust["fgsm"][e]["pgd-train"] for e in eps_list],
                 "dp-train": [robust["fgsm"][e]["dp-train"] for e in eps_list]},
                "Robust Accuracy under FGSM", "epsilon", "accuracy", os.path.join(results_dir, "robust_fgsm.png"))

    plot_curves(eps_list,
                {"standard": [robust["pgd"][e]["standard"] for e in eps_list],
                 "pgd-train": [robust["pgd"][e]["pgd-train"] for e in eps_list],
                 "dp-train": [robust["pgd"][e]["dp-train"] for e in eps_list]},
                "Robust Accuracy under PGD", "epsilon", "accuracy", os.path.join(results_dir, "robust_pgd.png"))

    # Membership inference
    shadow_models = []
    for _ in range(args.shadow_models):
        sm = SmallCNN().to(device)
        _ = train_classifier(sm, train_loader, test_loader, device, epochs=max(1, epochs - 1))
        shadow_models.append(sm)

    mia_std = run_membership_inference(std, shadow_models, train_ds, test_ds, device, results_dir)
    mia_adv = run_membership_inference(adv, shadow_models, train_ds, test_ds, device, results_dir)
    mia_dp = run_membership_inference(dp, shadow_models, train_ds, test_ds, device, results_dir)

    # CVAE + recon membership
    cvae, cvae_hist = train_cvae(train_loader, device, epochs=cvae_epochs, z_dim=32)
    save_samples_cvae(cvae, device, os.path.join(results_dir, "cvae_samples.png"), n_per_class=8, z_dim=32)

    xin, yin = dataset_to_tensors(train_ds, max_items=min(3000, len(train_ds)))
    xout, yout = dataset_to_tensors(test_ds, max_items=min(3000, len(test_ds)))
    ein = cvae_recon_errors(cvae, xin, yin, device)
    eout = cvae_recon_errors(cvae, xout, yout, device)
    recon_gap = float(np.mean(eout) - np.mean(ein))

    plt.figure()
    plt.hist(ein, bins=40, alpha=0.6, label="IN (train)")
    plt.hist(eout, bins=40, alpha=0.6, label="OUT (test)")
    plt.xlabel("reconstruction MSE"); plt.ylabel("count")
    plt.title("CVAE Reconstruction Error: IN vs OUT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cvae_recon_in_out.png"), dpi=160)
    plt.close()

    # GMM baseline
    gmm_metrics = run_gmm_baseline(train_ds, test_ds, K=5, pca_dim=30, seed=args.seed)

    summary = {
        "device": str(device),
        "standard": {"test": std_test, "mia": mia_std},
        "pgd_trained": {"test": adv_test, "mia": mia_adv},
        "dp_trained": {"test": dp_test, "mia": mia_dp},
        "robustness": robust,
        "cvae": {"epochs": cvae_epochs, "recon_gap_out_minus_in": recon_gap, "loss_curve": cvae_hist},
        "gmm": gmm_metrics,
        "notes": {
            "dp_warning": "DP-SGD here is minimal and has no epsilon/delta accounting; use it to study leakage trends.",
            "quick_mode": bool(args.quick)
        }
    }
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved results to:", results_dir)
    print("standard test acc:", std_test["acc"], "mia auc:", mia_std["mia_learned_auc"])
    print("pgd-train test acc:", adv_test["acc"], "mia auc:", mia_adv["mia_learned_auc"])
    print("dp-train test acc:", dp_test["acc"], "mia auc:", mia_dp["mia_learned_auc"])
    print("cvae recon gap (OUT-IN):", recon_gap)
    print("gmm test acc:", gmm_metrics["gmm_test_acc"])

if __name__ == "__main__":
    main()
