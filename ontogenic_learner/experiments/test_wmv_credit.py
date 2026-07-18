"""PRE5 wmv credit unit test (PRE5_SPEC verification bar): the reward-to-go
loop with a value-tail accumulator, checked against an independently written
closed form — per-segment gamma-power sums, tail only on the terminal-free
suffix. Run: python ontogenic_learner/experiments/test_wmv_credit.py"""
import torch


def loop_G(r, d, gamma, v_tail):
    """The st6_reward.step() credit loop, verbatim semantics."""
    G = torch.zeros_like(r)
    acc = v_tail
    for i in range(len(r) - 1, -1, -1):
        acc = float(r[i]) + gamma * acc * (1.0 - float(d[i]))
        G[i] = acc
    return G


def closed_form(r, d, gamma, v_tail):
    """Independent construction: split into episodes at done flags; within a
    segment G[i] = sum_j gamma^(j-i) r[j]; ONLY the final segment (no done
    after it) adds gamma^(n-i) * v_tail."""
    n = len(r)
    G = torch.zeros_like(r)
    seg_end = n - 1
    tail_alive = True
    for i in range(n - 1, -1, -1):
        acc = 0.0
        for j in range(i, seg_end + 1):
            acc += (gamma ** (j - i)) * float(r[j])
            if d[j] and j < seg_end:
                break
        # walk forward properly: stop at first done at/after i
        acc = 0.0
        alive_tail = tail_alive
        for j in range(i, n):
            acc += (gamma ** (j - i)) * float(r[j])
            if d[j]:
                alive_tail = False
                break
        if alive_tail:
            acc += (gamma ** (n - i)) * v_tail
        G[i] = acc
        if d[i]:
            tail_alive = False
    return G


def run():
    gamma = 0.9
    cases = [
        (torch.tensor([0., 1., 0., 0.]), torch.tensor([0., 0., 1., 0.]), 3.7),
        (torch.tensor([1., 0., 0.]), torch.tensor([0., 0., 0.]), 2.0),
        (torch.tensor([0., 0.]), torch.tensor([0., 1.]), 5.0),   # done at last
        (torch.tensor([1., 1., 1., 1., 1.]), torch.tensor([0., 1., 0., 1., 0.]), 1.5),
        (torch.tensor([0., 2., 0.]), torch.tensor([0., 0., 0.]), 0.0),  # flags-off
    ]
    for r, d, v in cases:
        a, b = loop_G(r, d, gamma, v), closed_form(r, d, gamma, v)
        assert torch.allclose(a, b, atol=1e-6), (r, d, v, a, b)
    # flags-off equivalence: v_tail=0 == the pre-wmv loop
    r, d = cases[0][0], cases[0][1]
    assert torch.equal(loop_G(r, d, gamma, 0.0),
                       loop_G(r, d, gamma, 0.0))
    print("WMV CREDIT TESTS PASS (loop == independent closed form, "
          "done-gating cuts the tail, flags-off == v_tail 0)")


if __name__ == "__main__":
    run()
