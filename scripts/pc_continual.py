"""Where ITERATIVE tropical PC earns its keep: catastrophic forgetting.

Prospective configuration (Song et al. 2024) — settling latents to equilibrium
BEFORE updating weights — interferes less with prior knowledge than backprop's
direct gradient step. We test this on the tropical substrate: same DyadicPCNet,
same Adam optimizer, same permuted-MNIST task sequence; the ONLY difference is the
learning rule —
  iterative : train_step           (settle to equilibrium, then local update)
  cascade   : train_step_cascade   (single-pass = backprop, no relaxation)
If iterative retains earlier tasks better, iteration pays off (and it's
multiplier-free, depth-practical to ~32). Permuted-MNIST = the canonical
continual-learning forgetting benchmark.
"""
import sys
import time
import torch

from tikitorch.nn import DyadicPCNet


def get_mnist(down, n_train, n_test):
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((down, down), antialias=True),
        transforms.Lambda(lambda t: t.view(-1)),
    ])
    tr = datasets.MNIST("data", train=True, download=True, transform=tf)
    te = datasets.MNIST("data", train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(n_train)])
    Ytr = torch.tensor([tr[i][1] for i in range(n_train)])
    Xte = torch.stack([te[i][0] for i in range(n_test)])
    Yte = torch.tensor([te[i][1] for i in range(n_test)])
    return Xtr, Ytr, Xte, Yte


def make_tasks(Xtr, Ytr, Xte, Yte, n_tasks, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    perms = [torch.arange(d)] + [torch.randperm(d, generator=g) for _ in range(n_tasks - 1)]
    return [(Xtr[:, p], Ytr, Xte[:, p], Yte) for p in perms]


@torch.no_grad()
def acc(model, X, Y):
    return (model.predict_logits(X).argmax(-1) == Y).float().mean().item()


def continual(model, tasks, learner, epochs, bs=64):
    """Train tasks sequentially; after each, record test acc on ALL tasks."""
    history = []
    for t, (Xtr, Ytr, _, _) in enumerate(tasks):
        n = Xtr.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n - bs, bs):
                idx = perm[i:i + bs]
                learner(model, Xtr[idx], Ytr[idx])
        history.append([acc(model, tasks[j][2], tasks[j][3]) for j in range(len(tasks))])
    return history


def report(name, history):
    final = history[-1]
    avg = sum(final) / len(final)
    # forgetting: for each earlier task, (best acc right after learning it) - (final acc)
    forget = []
    for j in range(len(history) - 1):
        forget.append(history[j][j] - final[j])
    mean_forget = sum(forget) / max(len(forget), 1)
    print(f"  {name:>10}: final per-task = {[f'{a:.2f}' for a in final]}  "
          f"avg={avg:.3f}  mean-forgetting={mean_forget:.3f}")
    return avg, mean_forget


if __name__ == "__main__":
    down, n_train, n_test = 12, 5000, 2000
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    epochs, width, n_steps, lr = 6, 64, 40, 1e-3
    d = down * down
    Xtr, Ytr, Xte, Yte = get_mnist(down, n_train, n_test)
    tasks = make_tasks(Xtr, Ytr, Xte, Yte, n_tasks, d, seed=0)
    dims = [d] + [width] * (depth - 1) + [10]
    print(f"== permuted-MNIST continual learning: depth {depth}, {n_tasks} tasks, "
          f"{epochs} ep/task, settle {n_steps} ==")

    # iterative-equilibrium PC
    torch.manual_seed(0)
    net = DyadicPCNet(dims, gamma=0.125)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time()
    h_iter = continual(net, tasks, lambda m, x, y: m.train_step(x, y, opt, n_steps=n_steps),
                       epochs)
    a_i, f_i = report("iterative", h_iter)
    print(f"             ({time.time()-t0:.0f}s)")

    # cascade (= backprop)
    torch.manual_seed(0)
    net = DyadicPCNet(dims, gamma=0.125)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time()
    h_casc = continual(net, tasks,
                       lambda m, x, y: m.train_step_cascade(x, y, opt), epochs)
    a_c, f_c = report("cascade", h_casc)
    print(f"             ({time.time()-t0:.0f}s)")

    print(f"\n  VERDICT: iterative avg {a_i:.3f} (forget {f_i:.3f}) vs "
          f"cascade avg {a_c:.3f} (forget {f_c:.3f}) -> "
          f"iterative {'WINS' if a_i > a_c else 'loses'} by {a_i-a_c:+.3f}")
