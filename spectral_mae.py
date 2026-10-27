"""
Masked spectral autoencoder (MAE-style) for hyperspectral signals.

Implements the architecture described in "Understanding_HyperSL.docx":

  1. Universal Spectral Tokenizer   (multiscale 1-D convs -> one token per band)
  2. Spectral Wavelength Position Encoding  (sin/cos of (wavelength - 400) / 10000^(2j/d))
  3. Encoder   : high-ratio random masking + CLS token + Transformer blocks + Aggregate
                 (outputs a representation vector plus K and V of the visible tokens)
  4. Decoder   : "decouple visible information" (K, V, representation vector),
                 restore token order, fill with mask tokens, add positions,
                 lightweight Transformer (N/2 blocks), output layer
  5. Loss      : MSE + cosine term + alpha * L2 regularisation

Input is a batch of spectra x with shape (B, c), plus the centre wavelength (nm)
of each of the c bands. c may differ between datasets / calls (it only changes
the token-sequence length).

Where the document was ambiguous, the choice made is marked with  # ASSUMPTION.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SpectralMAEConfig:
    embed_dim: int = 128            # N: tokenizer output channels == token dim == d of the PE
    enc_depth: int = 6              # N Transformer blocks in the encoder  (figure: "x N")
    dec_depth: Optional[int] = None  # decoder blocks; None -> enc_depth // 2   (figure: "x N/2")
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    mask_ratio: float = 0.75        # "high masking ratio"
    norm_first: bool = True         # ASSUMPTION: pre-norm, as in the equations (see notes)
    # Tokenizer: (kernel, dilation) for the three branches Conv1D(N,3,1,1), (N,5,1,2), (N,9,1,8)
    tok_kernels: tuple = (3, 5, 9)
    tok_dilations: tuple = (1, 2, 8)  # ASSUMPTION: last number in the figure labels is the dilation
    wavelength_offset: float = 400.0  # the "400" bias in the position encoding
    pe_base: float = 10000.0
    decouple_softmax: bool = False  # ASSUMPTION: plain scaled dot product, no softmax


# --------------------------------------------------------------------------- #
# 1. Universal spectral tokenizer
# --------------------------------------------------------------------------- #
class UniversalSpectralTokenizer(nn.Module):
    """
    x (B, c)  ->  tokens (B, c, N)

    x^(i)  = Conv1D_i(x)                  in R^{N x c}   (i = 1, 2, 3; different receptive fields)
    x^(4)  = Conv1D(x^(1)+x^(2)+x^(3))^T  in R^{c x N}

    The number of bands c becomes the sequence length, so any number of bands works.
    """

    def __init__(self, embed_dim: int, kernels=(3, 5, 9), dilations=(1, 2, 8)):
        super().__init__()
        assert len(kernels) == len(dilations)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    1, embed_dim, kernel_size=k, dilation=d,
                    padding=d * (k - 1) // 2,  # 'same' length so the branches can be added
                )
                for k, d in zip(kernels, dilations)
            ]
        )
        # ASSUMPTION: the 4th Conv1D in the document is a pointwise (kernel_size=1) conv N -> N.
        self.fuse = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                                  # (B, 1, c)
        x = sum(branch(x) for branch in self.branches)      # (B, N, c)  "Add"
        x = self.fuse(x)                                    # (B, N, c)
        return x.transpose(1, 2).contiguous()               # (B, c, N)  "Transpose"


# --------------------------------------------------------------------------- #
# 2. Spectral wavelength position encoding
# --------------------------------------------------------------------------- #
class SpectralWavelengthPE(nn.Module):
    """
    PE(w, 2j)   = sin((w - 400) / 10000^(2j/d))
    PE(w, 2j+1) = cos((w - 400) / 10000^(2j/d))
    """

    def __init__(self, dim: int, offset: float = 400.0, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "embed_dim must be even"
        self.dim, self.offset = dim, offset
        j2 = torch.arange(0, dim, 2, dtype=torch.float32)           # 2j
        self.register_buffer("inv_freq", 1.0 / (base ** (j2 / dim)), persistent=False)

    def forward(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """wavelengths: (c,) or (B, c) in nm  ->  (c, d) or (B, c, d)"""
        pos = (wavelengths.float() - self.offset).unsqueeze(-1)     # (..., c, 1)
        ang = pos * self.inv_freq                                   # (..., c, d/2)
        pe = torch.zeros(*ang.shape[:-1], self.dim, device=ang.device, dtype=ang.dtype)
        pe[..., 0::2] = torch.sin(ang)
        pe[..., 1::2] = torch.cos(ang)
        return pe


# --------------------------------------------------------------------------- #
# Transformer block (MSA + MLP with residuals and LayerNorm)
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """
    pre-norm  (norm_first=True):   H' = MSA(LN(H)) + H ;   H = MLP(LN(H')) + H'
    post-norm (norm_first=False):  H' = LN(MSA(H) + H) ;   H = LN(MLP(H') + H')
    """

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            y = self.ln1(h)
            h = h + self.attn(y, y, y, need_weights=False)[0]
            h = h + self.mlp(self.ln2(h))
        else:
            h = self.ln1(h + self.attn(h, h, h, need_weights=False)[0])
            h = self.ln2(h + self.mlp(h))
        return h


# --------------------------------------------------------------------------- #
# 3. Encoder
# --------------------------------------------------------------------------- #
class SpectralEncoder(nn.Module):
    def __init__(self, cfg: SpectralMAEConfig):
        super().__init__()
        d = cfg.embed_dim
        self.cfg = cfg
        self.tokenizer = UniversalSpectralTokenizer(d, cfg.tok_kernels, cfg.tok_dilations)
        self.pos = SpectralWavelengthPE(d, cfg.wavelength_offset, cfg.pe_base)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.norm_first)
             for _ in range(cfg.enc_depth)]
        )
        self.norm = nn.LayerNorm(d)
        # "Aggregate": ASSUMPTION - K and V are linear projections of the final visible tokens,
        # and the representation vector is the mean of CLS + visible tokens.
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)

    @staticmethod
    def random_masking(h: torch.Tensor, mask_ratio: float):
        """Per-sample random masking (MAE style). Returns visible tokens + bookkeeping."""
        B, L, D = h.shape
        n_keep = max(1, int(round(L * (1.0 - mask_ratio))))
        noise = torch.rand(B, L, device=h.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        h_vis = torch.gather(h, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, L, device=h.device)          # 1 = masked, 0 = visible
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return h_vis, mask, ids_keep, ids_restore

    def forward(self, x, wavelengths, mask_ratio: Optional[float] = None):
        mask_ratio = self.cfg.mask_ratio if mask_ratio is None else mask_ratio

        t = self.tokenizer(x)                              # T: (B, L, d)
        p = self.pos(wavelengths)                          # P: (L, d) or (B, L, d)
        h = t + p                                          # H = T + P

        h_vis, mask, ids_keep, ids_restore = self.random_masking(h, mask_ratio)
        cls = self.cls_token.expand(h_vis.size(0), -1, -1)
        h_vis = torch.cat([cls, h_vis], dim=1)             # {CLS, h_m}

        for blk in self.blocks:
            h_vis = blk(h_vis)
        h_vis = self.norm(h_vis)

        # Aggregate
        rep = h_vis.mean(dim=1)                            # (B, d) representation vector
        vis = h_vis[:, 1:]                                 # visible tokens (without CLS)
        k, v = self.to_k(vis), self.to_v(vis)              # (B, m, d) each
        return dict(rep=rep, k=k, v=v, mask=mask, ids_keep=ids_keep, ids_restore=ids_restore)


# --------------------------------------------------------------------------- #
# 4. Decoder
# --------------------------------------------------------------------------- #
class SpectralDecoder(nn.Module):
    def __init__(self, cfg: SpectralMAEConfig):
        super().__init__()
        d = cfg.embed_dim
        depth = cfg.dec_depth if cfg.dec_depth is not None else max(1, cfg.enc_depth // 2)
        self.cfg = cfg
        self.pos = SpectralWavelengthPE(d, cfg.wavelength_offset, cfg.pe_base)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.norm_first)
             for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)          # output layer: each band token -> one reconstructed value

    def forward(self, enc: dict, wavelengths: torch.Tensor, L: int):
        rep, k, v = enc["rep"], enc["k"], enc["v"]
        ids_keep, ids_restore = enc["ids_keep"], enc["ids_restore"]
        B, m, d = k.shape

        # Decouple visible information: dot product of V with the representation vector gives
        # one score per visible band, which scales K element-wise.   H_v = K * (V . rep)
        scores = torch.einsum("bmd,bd->bm", v, rep) / math.sqrt(d)
        if self.cfg.decouple_softmax:
            scores = scores.softmax(dim=-1)
        h_vis = k * scores.unsqueeze(-1)                                # (B, m, d)

        # Restore original order, fill masked positions with the mask token
        full = self.mask_token.expand(B, L, d).clone()
        full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, d), h_vis)
        # (equivalent to concat + gather with ids_restore; ids_restore kept for API symmetry)

        full = full + self.pos(wavelengths)                             # positions for ALL tokens
        for blk in self.blocks:
            full = blk(full)
        return self.out(self.norm(full)).squeeze(-1)                    # (B, L) reconstruction


# --------------------------------------------------------------------------- #
# 5. Full model + loss
# --------------------------------------------------------------------------- #
class SpectralMAE(nn.Module):
    def __init__(self, cfg: Optional[SpectralMAEConfig] = None):
        super().__init__()
        self.cfg = cfg or SpectralMAEConfig()
        self.encoder = SpectralEncoder(self.cfg)
        self.decoder = SpectralDecoder(self.cfg)

    def forward(self, x: torch.Tensor, wavelengths: torch.Tensor,
                mask_ratio: Optional[float] = None):
        """
        x           : (B, c) spectra
        wavelengths : (c,) band centre wavelengths in nm (or (B, c))
        returns     : dict(recon (B, c), mask (B, c) with 1 = masked, rep (B, d))
        """
        enc = self.encoder(x, wavelengths, mask_ratio)
        recon = self.decoder(enc, wavelengths, x.size(1))
        return dict(recon=recon, mask=enc["mask"], rep=enc["rep"])

    @torch.no_grad()
    def embed(self, x, wavelengths):
        """Representation vector for downstream tasks (no masking)."""
        return self.encoder(x, wavelengths, mask_ratio=0.0)["rep"]

    def loss(self, x, out, alpha: float = 1e-4, on: str = "all"):
        """
        Loss = MSE(x, x_hat) + CosSim term + alpha * ||theta||^2

        on = "all"    : loss over every band (as written in the document)
        on = "masked" : loss only over masked bands (classic MAE)
        The cosine term is 1 - cos(x, x_hat) so that minimising it aligns directions.
        """
        x_hat, mask = out["recon"], out["mask"]
        if on == "masked":
            w = mask
            mse = (((x_hat - x) ** 2) * w).sum(1) / w.sum(1).clamp(min=1)
            cos = 1 - F.cosine_similarity(x_hat * w, x * w, dim=1, eps=1e-8)
        else:
            mse = ((x_hat - x) ** 2).mean(1)
            cos = 1 - F.cosine_similarity(x_hat, x, dim=1, eps=1e-8)
        l2 = sum((p ** 2).sum() for p in self.parameters() if p.requires_grad)
        total = mse.mean() + cos.mean() + alpha * l2
        return total, dict(mse=mse.mean().item(), cos=cos.mean().item(), l2=float(l2.detach()))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = SpectralMAE()
    for c in (200, 224):                       # different band counts, same model
        wl = torch.linspace(400.1, 2498.8, c)
        x = torch.randn(4, c)
        out = model(x, wl)
        loss, parts = model.loss(x, out)
        print(f"c={c}: recon {tuple(out['recon'].shape)}, rep {tuple(out['rep'].shape)}, "
              f"masked {out['mask'].mean():.2f}, loss {loss.item():.3f}")
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")