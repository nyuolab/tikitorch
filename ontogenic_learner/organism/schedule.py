"""The lifetime schedule: declarative segments keyed on life_step (restarts never skip
experience), block-granular domain interleaving inside mixed segments (block-scale drift is
what the gate is validated on), and a feed-injection queue for guest experience."""
import torch


class Schedule:
    def __init__(self, cfg, gen: torch.Generator):
        self.cfg = cfg
        self.gen = gen
        self.block_dom = None
        self.block_until = -1
        self.queue = []                          # pending guest blocks: (name, steps)

    def _segment(self, life_step: int) -> dict:
        for seg in self.cfg.schedule:
            lo, hi = self.cfg.seg_bounds(seg)
            if lo <= life_step < hi:
                return seg
        return self.cfg.schedule[-1]

    def domain(self, life_step: int) -> str:
        if self.queue:                           # guest experience preempts at block edge
            if self.block_dom is None or life_step >= self.block_until:
                name, steps = self.queue.pop(0)
                self.block_dom, self.block_until = name, life_step + steps
            return self.block_dom
        seg = self._segment(life_step)
        if self.block_dom is None or life_step >= self.block_until \
                or self.block_dom not in seg["mix"]:
            doms = sorted(seg["mix"].keys())
            w = torch.tensor([seg["mix"][d] for d in doms], dtype=torch.float)
            i = torch.multinomial(w, 1, generator=self.gen).item()
            self.block_dom = doms[i]
            self.block_until = life_step + seg.get("block_steps", 8000)
        return self.block_dom

    def inject(self, name: str, steps: int):
        self.queue.append((name, steps))

    def state(self):
        return {"block_dom": self.block_dom, "block_until": self.block_until,
                "queue": list(self.queue)}

    def restore(self, st):
        self.block_dom = st["block_dom"]
        self.block_until = st["block_until"]
        self.queue = [tuple(q) for q in st["queue"]]
