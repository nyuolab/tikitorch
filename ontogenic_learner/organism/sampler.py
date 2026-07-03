"""Autoregressive sampling from the live net (pure forward, no_grad, dedicated generator).
Lifted from b1_shakespeare.sample, vocab-aware (prompts may contain pixel symbols)."""
import torch
import torch.nn.functional as F

from .policy import onehot


@torch.no_grad()
def sample(net, vocab, ctx: int, prompt: str, n: int, temp: float,
           gen: torch.Generator, device: str) -> str:
    ids = vocab.encode_mixed(prompt) if prompt else []
    if len(ids) < ctx:                            # left-pad with spaces
        ids = [vocab.stoi[" "]] * (ctx - len(ids)) + ids
    out = list(ids)
    for _ in range(n):
        x = torch.tensor(out[-ctx:], dtype=torch.long).unsqueeze(0)
        logits = net.predict_logits(onehot(x, vocab.V).to(device))[0] / max(temp, 1e-3)
        p = F.softmax(logits, -1)
        out.append(torch.multinomial(p.cpu(), 1, generator=gen).item())
    return vocab.decode(out[len(ids):])
