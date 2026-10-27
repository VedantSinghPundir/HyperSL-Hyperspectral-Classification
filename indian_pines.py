"""
Indian Pines pipeline for SpectralMAE.

  python indian_pines.py explore  --data Indian_pines_corrected.mat --gt Indian_pines_gt.mat
  python indian_pines.py pretrain --data ... --gt ... --epochs 30
  python indian_pines.py evaluate --data ... --gt ...

Steps
  explore  : prints shapes / class counts and saves pictures (false-colour image, ground truth,
             mean spectrum per class).
  pretrain : self-supervised masked reconstruction on ALL pixels (labels are NOT used).
             Saves ip_checkpoint.pt and a reconstruction plot.
  evaluate : freezes the encoder, turns every labelled pixel into a representation vector,
             trains a linear classifier on a small labelled subset, reports OA / AA / Kappa,
             compares with the same classifier on raw spectra, and saves a classification map.

All pictures go to --out (default: ./ip_outputs).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from spectral_mae import SpectralMAE, SpectralMAEConfig

CLASS_NAMES = [
    "Background", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
    "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill",
    "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Bldg-Grass-Tree-Drives",
    "Stone-Steel-Towers",
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_indian_pines(data_path, gt_path):
    """Return cube (H, W, C) float32, labels (H, W) int. Finds the arrays by shape, not name."""
    def pick(path, ndim):
        m = sio.loadmat(path)
        for k, v in m.items():
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == ndim:
                return v
        raise ValueError(f"No {ndim}-D array found in {path}; keys: {list(m)}")
    cube = pick(data_path, 3).astype(np.float32)
    gt = pick(gt_path, 2).astype(np.int64)
    assert cube.shape[:2] == gt.shape, f"cube {cube.shape} and gt {gt.shape} do not match"
    return cube, gt


def ip_wavelengths(n_bands):
    """
    Approximate AVIRIS band centres in nm.  AVIRIS has 220 bands over ~400-2500 nm.  The
    'corrected' Indian Pines has 200 bands: the 20 water-absorption bands
    [104-108], [150-163], 220 (1-indexed) were removed.  Exact centres are in the AVIRIS
    calibration file; this uniform grid is accurate to a few nm.
    """
    full = np.linspace(400.0, 2500.0, 220)
    if n_bands == 220:
        return full.astype(np.float32)
    if n_bands == 200:
        drop = list(range(103, 108)) + list(range(149, 163)) + [219]
        keep = np.setdiff1d(np.arange(220), drop)
        return full[keep].astype(np.float32)
    return np.linspace(400.0, 2500.0, n_bands).astype(np.float32)


def normalise(cube):
    """Per-band z-score using statistics over the whole image."""
    mu, sd = cube.mean((0, 1), keepdims=True), cube.std((0, 1), keepdims=True) + 1e-6
    return (cube - mu) / sd


def false_colour(cube, wl):
    idx = [int(np.abs(wl - w).argmin()) for w in (650, 550, 480)]   # R, G, B
    rgb = cube[:, :, idx].copy()
    for i in range(3):
        lo, hi = np.percentile(rgb[:, :, i], (2, 98))
        rgb[:, :, i] = np.clip((rgb[:, :, i] - lo) / (hi - lo + 1e-9), 0, 1)
    return rgb


def label_cmap():
    rng = np.random.RandomState(3)
    colors = rng.rand(17, 3) * 0.8 + 0.2
    colors[0] = 0
    return matplotlib.colors.ListedColormap(colors)


# --------------------------------------------------------------------------- #
# explore
# --------------------------------------------------------------------------- #
def explore(args):
    cube, gt = load_indian_pines(args.data, args.gt)
    wl = ip_wavelengths(cube.shape[2])
    os.makedirs(args.out, exist_ok=True)
    print(f"cube {cube.shape}  (H x W x bands), value range {cube.min():.0f}..{cube.max():.0f}")
    print(f"labelled pixels: {(gt > 0).sum()} of {gt.size}")
    for c in range(1, int(gt.max()) + 1):
        print(f"  class {c:2d} {CLASS_NAMES[c] if c < 17 else '':24s} {(gt == c).sum():5d} pixels")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(false_colour(cube, wl)); ax[0].set_title("False-colour image")
    ax[1].imshow(gt, cmap=label_cmap(), vmin=0, vmax=16, interpolation="nearest")
    ax[1].set_title("Ground truth (black = unlabelled)")
    for c in range(1, int(gt.max()) + 1):
        ax[2].plot(wl, cube[gt == c].mean(0), label=CLASS_NAMES[c] if c < 17 else str(c))
    ax[2].set_title("Mean spectrum per class"); ax[2].set_xlabel("wavelength (nm)")
    ax[2].legend(fontsize=6, ncol=2)
    for a in ax[:2]:
        a.axis("off")
    plt.tight_layout(); plt.savefig(f"{args.out}/explore.png", dpi=130); plt.close()
    print(f"saved {args.out}/explore.png")


# --------------------------------------------------------------------------- #
# pretrain
# --------------------------------------------------------------------------- #
def get_device(args):
    return args.device or ("cuda" if torch.cuda.is_available() else "cpu")


def pretrain(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cube, gt = load_indian_pines(args.data, args.gt)
    wl = torch.from_numpy(ip_wavelengths(cube.shape[2]))
    X = torch.from_numpy(normalise(cube).reshape(-1, cube.shape[2]))     # (H*W, C) every pixel
    dev = get_device(args)
    print(f"device {dev}; {len(X)} pixel spectra of {X.shape[1]} bands (labels unused)")

    cfg = SpectralMAEConfig(embed_dim=args.dim, enc_depth=args.depth, num_heads=args.heads,
                            mask_ratio=args.mask_ratio)
    model = SpectralMAE(cfg).to(dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)  # L2 is in loss
    steps = args.epochs * (len(X) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    wl_dev = wl.to(dev)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = tc = 0.0
        nb = len(X) // args.bs
        for b in range(nb):
            x = X[perm[b * args.bs:(b + 1) * args.bs]].to(dev)
            out = model(x, wl_dev)
            loss, parts = model.loss(x, out, alpha=args.alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += parts["mse"]; tc += parts["cos"]
        print(f"epoch {ep + 1:3d}/{args.epochs}  mse {tot / nb:.4f}  (1-cos) {tc / nb:.4f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"cfg": cfg, "state_dict": model.state_dict()}, args.ckpt)
    print(f"saved {args.ckpt}")

    # reconstruction picture: original vs reconstruction from 25% visible bands
    model.eval()
    torch.manual_seed(0)
    idx = np.random.choice(np.flatnonzero(gt.reshape(-1) > 0), 4, replace=False)
    with torch.no_grad():
        x = X[idx].to(dev)
        out = model(x, wl_dev)
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    for i, a in enumerate(ax.ravel()):
        m = out["mask"][i].cpu().numpy().astype(bool)
        a.plot(wl, x[i].cpu(), "k", lw=1, label="original")
        a.plot(wl, out["recon"][i].cpu(), "r", lw=1, label="reconstruction")
        a.scatter(wl[~m], x[i].cpu()[~m], s=12, c="b", label="visible bands", zorder=3)
        a.set_title(f"pixel class: {CLASS_NAMES[gt.reshape(-1)[idx[i]]]}")
        if i == 0:
            a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{args.out}/reconstruction.png", dpi=130); plt.close()
    print(f"saved {args.out}/reconstruction.png")


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred):
    classes = np.unique(y_true)
    per_class = np.array([(y_pred[y_true == c] == c).mean() for c in classes])
    return (y_pred == y_true).mean(), per_class.mean(), cohen_kappa_score(y_true, y_pred), per_class


def evaluate(args):
    cube, gt = load_indian_pines(args.data, args.gt)
    wl = torch.from_numpy(ip_wavelengths(cube.shape[2]))
    Xn = normalise(cube)
    H, W, C = Xn.shape
    X = Xn.reshape(-1, C)
    y = gt.reshape(-1)
    dev = get_device(args)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = SpectralMAE(ck["cfg"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    # embed every pixel (no masking) in chunks
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            xb = torch.from_numpy(X[i:i + 1024]).to(dev)
            feats.append(model.encoder(xb, wl.to(dev), mask_ratio=0.0)["rep"].cpu())
    F_all = torch.cat(feats).numpy()
    print(f"embeddings: {F_all.shape}")

    lab = np.flatnonzero(y > 0)
    tr, te = train_test_split(lab, train_size=args.train_frac, stratify=y[lab], random_state=args.seed)
    print(f"labelled pixels {len(lab)}  ->  train {len(tr)} ({args.train_frac:.0%}), test {len(te)}")

    def fit_eval(name, feat):
        mu, sd = feat[tr].mean(0), feat[tr].std(0) + 1e-6
        f = (feat - mu) / sd
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(f[tr], y[tr])
        oa, aa, kappa, pc = metrics(y[te], clf.predict(f[te]))
        print(f"{name:28s} OA {oa * 100:5.2f}  AA {aa * 100:5.2f}  Kappa {kappa:.4f}")
        return clf, f, pc

    clf, f_all, pc = fit_eval("SpectralMAE embeddings", F_all)
    _, _, pc_raw = fit_eval("raw spectra (baseline)", X)

    print("\nper-class test accuracy (embeddings / raw):")
    for c, a, b in zip(np.unique(y[te]), pc, pc_raw):
        print(f"  {CLASS_NAMES[c]:24s} {a * 100:5.1f}  /  {b * 100:5.1f}")

    # classification map over the whole image
    os.makedirs(args.out, exist_ok=True)
    pred = clf.predict(f_all).reshape(H, W)
    pred_masked = np.where(gt > 0, pred, 0)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(gt, cmap=label_cmap(), vmin=0, vmax=16, interpolation="nearest"); ax[0].set_title("Ground truth")
    ax[1].imshow(pred_masked, cmap=label_cmap(), vmin=0, vmax=16, interpolation="nearest")
    ax[1].set_title("Prediction (labelled pixels)")
    ax[2].imshow(pred, cmap=label_cmap(), vmin=0, vmax=16, interpolation="nearest")
    ax[2].set_title("Prediction (all pixels)")
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(f"{args.out}/classification_map.png", dpi=130); plt.close()
    print(f"saved {args.out}/classification_map.png")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["explore", "pretrain", "evaluate"])
    p.add_argument("--data", default="Indian_pines_corrected.mat")
    p.add_argument("--gt", default="Indian_pines_gt.mat")
    p.add_argument("--out", default="ip_outputs")
    p.add_argument("--ckpt", default="ip_checkpoint.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    # model / training
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1e-6)
    # evaluation
    p.add_argument("--train_frac", type=float, default=0.10)
    args = p.parse_args()
    {"explore": explore, "pretrain": pretrain, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()