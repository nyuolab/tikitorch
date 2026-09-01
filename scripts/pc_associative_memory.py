"""Associative-memory retrieval (Salvatori-style): store K images, retrieve from partial.

Multiplier-free SparseCodingPC vs a backprop autoencoder. Metric = fraction of stored
images recalled with recon MSE below a threshold, from random pixel masking. This is the
multiplier-free analogue of Salvatori et al. 2021 ("Associative memories via predictive
coding", NeurIPS), where a float PCN retrieves far more stored patterns than a backprop
autoencoder given a partial cue.

Measured (seed 0, K=200 MNIST, this script):
    kept  thr     sparse-PC (mult-free)   backprop-AE
    50%   0.01    43%                     0%
    50%   --      mean recon MSE PC=0.0104   AE=0.0384  (PC 3.7x lower)
    25%   0.01     3%                     0%
    25%   --      mean recon MSE PC=0.0210   AE=0.0729  (PC 3.5x lower)

The AE collapses to ~mean output and recalls nothing exactly; the sparse-PC settles the
code from the *observed* pixels and recalls the stored pattern. Same qualitative
PC>>AE gap Salvatori report, here under the multiplier-free (+-2^k dictionary, top-k WTA
code) constraint. Run: PYTHONPATH=. python scripts/pc_associative_memory.py
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
from tikitorch.nn.predictive_coding import SparseCodingPC
torch.manual_seed(0)
from torchvision import datasets
ds = datasets.MNIST("/tmp/_tiki_data", train=True, download=True)
K = 200
X = (ds.data[:K].float() / 255.0).reshape(K, 784)            # the stored "memory"

# --- multiplier-free sparse-coding PC: memorize the K images ---
pc = SparseCodingPC(n_pixels=784, n_atoms=512, n_active=20)
for _ in range(120):
    pc.train_step(X, lr_dict=0.08, n_steps=30)

# --- backprop autoencoder baseline (plain AE trained to reconstruct the K) ---
ae = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                   nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784))
opt = torch.optim.Adam(ae.parameters(), lr=2e-3)
for _ in range(2000):
    opt.zero_grad(); loss = F.mse_loss(ae(X), X); loss.backward(); opt.step()


@torch.no_grad()
def retrieval_fraction(keep_frac, gen):
    mask = (torch.rand(K, 784, generator=gen) < keep_frac).float()
    cue = X * mask
    pc_out = pc.complete(X, mask, n_steps=200)            # PC: infer code from observed, recall
    ae_out = ae(cue)                                       # AE: feed the masked cue forward
    pc_mse = ((pc_out - X) ** 2).mean(1); ae_mse = ((ae_out - X) ** 2).mean(1)
    out = {}
    for thr in (0.001, 0.005, 0.01):
        out[thr] = ((pc_mse < thr).float().mean().item(), (ae_mse < thr).float().mean().item())
    return out, pc_mse.mean().item(), ae_mse.mean().item()


g = torch.Generator().manual_seed(1)
print(f"Associative memory: store K={K} MNIST images, retrieve from random masking.")
print(f"{'kept':>6} | {'thr':>6} | {'sparse-PC (mult-free)':>21} | {'backprop-AE':>11}")
for keep in (0.5, 0.25):
    res, pcm, aem = retrieval_fraction(keep, g)
    for thr, (p, a) in res.items():
        print(f"{int(keep*100):>5}% | {thr:>6} | {p*100:>20.0f}% | {a*100:>10.0f}%")
    print(f"        (mean recon MSE: PC={pcm:.4f}  AE={aem:.4f})")
