"""LSTM Autoencoder (PyTorch) and a test-only stand-in model.

The LSTM-AE is identical for every threshold method: it is trained once per
(file, window size W, seed) and its reconstruction errors are saved. All
threshold methods then read the same saved errors.
"""
from __future__ import annotations

import time

import numpy as np


def _set_seed(seed: int):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_lstm_ae(hidden: int, latent: int):
    import torch
    from torch import nn

    class LSTMAutoencoder(nn.Module):
        """Encoder LSTM -> latent vector -> repeated -> decoder LSTM -> value."""

        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(1, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.decoder = nn.LSTM(latent, hidden, batch_first=True)
            self.output = nn.Linear(hidden, 1)

        def forward(self, x):                      # x: (batch, W, 1)
            _, (h, _) = self.encoder(x)
            z = torch.tanh(self.to_latent(h[-1]))  # (batch, latent)
            z = z.unsqueeze(1).repeat(1, x.shape[1], 1)
            out, _ = self.decoder(z)
            return self.output(out)

    return LSTMAutoencoder()


def train_and_score(train_windows: np.ndarray, all_windows: np.ndarray, *, seed: int,
                    hidden=64, latent=32, lr=1e-3, batch_size=128, max_epochs=30,
                    patience=4, holdout=0.1, threads=2) -> tuple[np.ndarray, dict]:
    """Train on train_windows, return per-window MSE for all_windows."""
    import torch
    torch.set_num_threads(threads)
    _set_seed(seed)

    n_hold = max(1, int(round(holdout * len(train_windows))))
    tr = torch.from_numpy(train_windows[:-n_hold]).unsqueeze(-1)
    ho = torch.from_numpy(train_windows[-n_hold:]).unsqueeze(-1)  # chronologically last part
    model = build_lstm_ae(hidden, latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(seed)

    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g)
        tot = 0.0
        for i in range(0, len(tr), batch_size):
            xb = tr[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            hl = loss_fn(model(ho), ho).item()
        history.append((tot / len(tr), hl))
        if hl < best - 1e-7:
            best, bad = hl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(all_windows), 2048):
            xb = torch.from_numpy(all_windows[i:i + 2048]).unsqueeze(-1)
            errs.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).numpy())
    meta = {"epochs_run": len(history), "best_holdout_mse": float(best),
            "train_seconds": round(time.time() - t0, 1),
            "loss_history": [[round(a, 7), round(b, 7)] for a, b in history]}
    return np.concatenate(errs).astype(np.float32), meta


def standin_score(train_windows: np.ndarray, all_windows: np.ndarray, *, seed: int, **_) -> tuple[np.ndarray, dict]:
    """TEST-ONLY stand-in (PCA reconstruction). Used to smoke-test the
    pipeline where PyTorch is unavailable. Never used for reported results."""
    from sklearn.decomposition import PCA
    k = max(1, min(8, train_windows.shape[1] // 6))
    pca = PCA(n_components=k, random_state=seed).fit(train_windows)
    rec = pca.inverse_transform(pca.transform(all_windows))
    return ((rec - all_windows) ** 2).mean(axis=1).astype(np.float32), {"model": "pca_standin", "k": k}
