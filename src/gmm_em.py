"""
Gaussian Mixture Model with EM Algorithm

Model: p(x) = Σ_{k=1}^K π_k · N(x | μ_k, Σ_k)

For diagonal covariance Σ_k = diag(σ²_k):
    N(x|μ_k, Σ_k) = (2π)^{-d/2} |Σ_k|^{-1/2} exp(-½ Σ_j (x_j - μ_{kj})² / σ²_{kj})

EM Algorithm:
    E-step: r_{ik} = π_k N(x_i|μ_k,Σ_k) / Σ_j π_j N(x_i|μ_j,Σ_j)
    M-step: π_k = N_k/n,  μ_k = Σ_i r_{ik} x_i / N_k,  σ²_k = Σ_i r_{ik}(x_i-μ_k)² / N_k

References:
    [1] Bishop, "Pattern Recognition and Machine Learning", Ch. 9 (2006)
"""

import numpy as np


def _logsumexp(a, axis=1):
    """log(Σ_j exp(a_j)) = max(a) + log(Σ_j exp(a_j - max(a)))"""
    amax = np.max(a, axis=axis, keepdims=True)
    return amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True) + 1e-12)


def fit_diag_gmm(X, K: int, max_iter: int = 200, tol: float = 1e-4, seed: int = 0):
    """
    Fit diagonal-covariance GMM via EM.
    
    Returns: {π_k, μ_k, σ²_k, log-likelihood}
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # Initialize: π_k = 1/K, μ_k from data, σ²_k from data variance
    pi = np.ones(K) / K
    mu = X[rng.choice(n, size=K, replace=False)].copy()
    var = np.ones((K, d)) * (X.var(axis=0, keepdims=True) + 1e-3)

    prev_ll = None
    for _ in range(max_iter):
        # E-step: compute r_{ik} = P(z_i = k | x_i)
        # log N(x|μ,Σ) = -d/2 log(2π) - 1/2 log|Σ| - 1/2 (x-μ)^T Σ^{-1} (x-μ)
        log_det = np.sum(np.log(var + 1e-12), axis=1)  # (K,)
        diff = X[:, None, :] - mu[None, :, :]          # (n, K, d)
        maha = np.sum((diff * diff) / (var[None, :, :] + 1e-12), 2)  # (n, K)

        log_prob = np.log(pi + 1e-12)[None, :] - 0.5 * (d * np.log(2 * np.pi) + log_det[None, :] + maha)
        lse = _logsumexp(log_prob, axis=1)  # (n, 1)
        ll = float(np.sum(lse))
        log_r = log_prob - lse
        r = np.exp(log_r)  # (n, K)

        # M-step: update π_k, μ_k, σ²_k
        Nk = r.sum(axis=0) + 1e-12  # (K,)
        pi = Nk / n
        mu = (r.T @ X) / Nk[:, None]
        Ex2 = (r.T @ (X * X)) / Nk[:, None]
        var = np.maximum(Ex2 - mu * mu, 1e-4)

        # Convergence check
        if prev_ll is not None and abs(ll - prev_ll) <= tol * (abs(ll) + 1e-12):
            break
        prev_ll = ll

    return {"pi": pi, "mu": mu, "var": var, "loglik": prev_ll if prev_ll is not None else ll}


def loglik_diag_gmm(X, params):
    """
    Per-sample log-likelihood: log p(x_i) = log Σ_k π_k N(x_i | μ_k, Σ_k)
    """
    pi, mu, var = params["pi"], params["mu"], params["var"]
    n, d = X.shape
    log_det = np.sum(np.log(var + 1e-12), axis=1)
    diff = X[:, None, :] - mu[None, :, :]
    maha = np.sum((diff * diff) / (var[None, :, :] + 1e-12), 2)
    log_prob = np.log(pi + 1e-12)[None, :] - 0.5 * (d * np.log(2 * np.pi) + log_det[None, :] + maha)
    lse = _logsumexp(log_prob, axis=1)
    return lse.squeeze(1)
