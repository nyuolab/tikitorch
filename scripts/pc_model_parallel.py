"""Test 1 (training-at-scale): does PC's local credit train an LM under REAL model-parallel
sharding — each layer a separate PROCESS exchanging only NEIGHBOUR messages, no all-reduce,
no global barrier, no process ever holding the global model or gradient?

This is the correctness gate for the fabric vision (TRAINING_AT_SCALE.md). The staleness-
frontier showed async credit converges in a SINGLE process (could be a simulation artifact);
this shards one MLP char-LM across D=depth real `torch.distributed` (gloo) ranks, one Linear
layer per rank, and trains it with LOCAL PC credit:

  forward : rank l recv a from l-1 (point-to-point), a_next = relu(W_l a), send to l+1.
  backward: rank D-1 computes top error (softmax - target); each rank l recv error from l+1,
            computes its LOCAL grad gW = (err*relu_mask) outer a_l (no global gradient),
            updates W_l with its OWN optimizer, routes error e = W_l^T (err*relu_mask),
            sends to l-1. NO all-reduce, NO barrier — credit is neighbour-only.

Each rank holds ONLY its layer's weight + the neighbour messages. If the sharded val bpc
matches the single-process backprop baseline, the carrier survives real sharding.
Run: PYTHONPATH=. python scripts/pc_model_parallel.py --depth 4
"""
import os, sys, math, time, argparse, urllib.request
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

PATH = "/tmp/_tiki_data/tinyshakespeare.txt"


def load(ctx, V_box):
    os.makedirs("/tmp/_tiki_data", exist_ok=True)
    if not os.path.exists(PATH):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", PATH)
    text = open(PATH).read()[:300000]
    chars = sorted(set(text)); V = len(chars); V_box[0] = V
    stoi = {c: i for i, c in enumerate(chars)}
    d = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    N = len(d) - ctx
    X = torch.stack([d[i:i + ctx] for i in range(N)])
    y = d[ctx:ctx + N]
    n = int(0.9 * N)
    return (X[:n], y[:n]), (X[n:], y[n:]), V


def onehot(xb, V):
    return F.one_hot(xb, V).float().reshape(xb.size(0), -1)


def worker(rank, world, args, V_box):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29611"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(1234)
    (Xtr, ytr), (Xva, yva), V = load(args.ctx, V_box)
    W = args.width
    in_dim = args.ctx * V if rank == 0 else W
    out_dim = V if rank == world - 1 else W
    last = rank == world - 1
    torch.manual_seed(100 + rank)                                   # distinct per-layer init
    lin = torch.nn.Linear(in_dim, out_dim)
    opt = torch.optim.Adam(lin.parameters(), lr=args.lr)            # LOCAL optimizer — only this layer
    gen = torch.Generator().manual_seed(7)

    def fwd_recv(B):                                                # get this rank's input activation
        if rank == 0:
            return None                                            # rank 0 reads data directly (set by caller)
        a = torch.empty(B, in_dim)
        dist.recv(a, src=rank - 1)
        return a

    best = [1e9]

    @torch.no_grad()
    def validate(step):
        # full forward through the pipeline on a val chunk; last rank computes bpc
        nb, tot, ntok = 256, 0.0, 0
        for i in range(0, min(Xva.size(0), 4096) - nb, nb):
            if rank == 0:
                a = torch.relu(lin(onehot(Xva[i:i + nb], V)))
                dist.send(a, dst=1)
            else:
                a = torch.empty(nb, in_dim); dist.recv(a, src=rank - 1)
                z = lin(a)
                if not last:
                    dist.send(torch.relu(z), dst=rank + 1)
                else:
                    tot += F.cross_entropy(z, yva[i:i + nb], reduction="sum").item(); ntok += nb
        if last:
            bpc = tot / ntok / math.log(2)
            best[0] = min(best[0], bpc)
            print(f"  [shard-PC] step {step} val_bpc {bpc:.3f} (best {best[0]:.3f})", flush=True)

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(Xtr.size(0), (args.batch,), generator=gen)
        # ---- FORWARD (neighbour send/recv) ----
        if rank == 0:
            a_in = onehot(Xtr[idx], V)
        else:
            a_in = fwd_recv(args.batch)
        pre = lin(a_in)
        mask = (pre > 0).float() if not last else torch.ones_like(pre)
        a_out = torch.relu(pre) if not last else pre
        if not last:
            dist.send(a_out, dst=rank + 1)
        # ---- BACKWARD (neighbour send/recv) + LOCAL update ----
        if last:
            err = torch.softmax(a_out, -1) - F.one_hot(ytr[idx], V).float()   # top error
        else:
            err = torch.empty(args.batch, out_dim); dist.recv(err, src=rank + 1)
        delta = err * mask                                          # gate by this rank's relu mask
        gW = delta.t() @ a_in / args.batch                         # LOCAL grad — only neighbour error + own activation
        gb = delta.mean(0)
        opt.zero_grad(set_to_none=True)
        lin.weight.grad = gW; lin.bias.grad = gb
        opt.step()                                                  # LOCAL update; NO all-reduce, NO barrier
        if rank > 0:
            e_down = delta @ lin.weight.detach()                   # route error to neighbour (same weight, no transpose copy)
            dist.send(e_down, dst=rank - 1)
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            validate(step)
    if last:
        print(f"==> sharded-PC best val_bpc {best[0]:.4f} (D={world} ranks, {time.time()-t0:.0f}s)", flush=True)
    dist.destroy_process_group()


def single_proc_backprop(args, V):
    """Baseline: the SAME MLP in one process, ordinary backprop — the quality bar."""
    torch.manual_seed(1234)
    (Xtr, ytr), (Xva, yva), _ = load(args.ctx, [V])
    dims = [args.ctx * V] + [args.width] * (args.depth - 1) + [V]
    layers = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i + 1]) for i in range(args.depth)])
    opt = torch.optim.Adam(layers.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(7)

    def fwd(xb):
        h = onehot(xb, V)
        for i, L in enumerate(layers):
            h = L(h)
            if i < len(layers) - 1:
                h = torch.relu(h)
        return h

    best = 1e9
    for step in range(args.steps):
        idx = torch.randint(Xtr.size(0), (args.batch,), generator=gen)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(fwd(Xtr[idx]), ytr[idx]).backward(); opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            with torch.no_grad():
                v = F.cross_entropy(fwd(Xva[:4096]), yva[:4096]).item() / math.log(2)
                best = min(best, v)
                print(f"  [single-bp] step {step} val_bpc {v:.3f} (best {best:.3f})", flush=True)
    print(f"==> single-process backprop best val_bpc {best:.4f}", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    V_box = [0]
    _, _, V = load(args.ctx, V_box); V = V_box[0]
    print("=== single-process backprop baseline (the quality bar) ===", flush=True)
    single_proc_backprop(args, V)
    print(f"\n=== model-parallel PC: {args.depth} real processes, 1 layer each, neighbour-only, no barrier ===", flush=True)
    mp.spawn(worker, args=(args.depth, args, V_box), nprocs=args.depth, join=True)
