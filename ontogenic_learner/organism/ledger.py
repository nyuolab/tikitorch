"""Append-only audit ledger: one JSONL event per applied update (+ BIRTH/RESUME/PROBE/FEED/
INCIDENT/CHECKPOINT), monotonic seq numbers, buffered writes, daily rotation + gzip."""
import gzip
import json
import os
import time


class Ledger:
    def __init__(self, dir_: str, seq: int = 0):
        self.dir = dir_
        os.makedirs(dir_, exist_ok=True)
        self.seq = max(seq, self._recover_seq())
        self.buf = []
        self.last_flush = time.time()
        self.day = self._today()

    def _recover_seq(self) -> int:
        """Continue from the highest seq already on disk (crash-resume must never reuse
        seq numbers; gaps are allowed — they record buffered events lost to a crash)."""
        import glob
        best = 0
        for p in sorted(glob.glob(os.path.join(self.dir, "ledger-*.jsonl")))[-2:]:
            try:
                last = None
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            last = line
                if last:
                    best = max(best, json.loads(last).get("seq", 0))
            except Exception:
                pass
        return best

    def _today(self):
        return time.strftime("%Y%m%d", time.gmtime())

    def _path(self, day):
        return os.path.join(self.dir, f"ledger-{day}.jsonl")

    def append(self, event: str, **kv):
        self.seq += 1
        self.buf.append(json.dumps({"seq": self.seq, "ts": round(time.time(), 3),
                                    "event": event, **kv}))
        if len(self.buf) >= 100 or time.time() - self.last_flush > 5:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        day = self._today()
        if day != self.day:                      # rotate + gzip yesterday
            old = self._path(self.day)
            if os.path.exists(old):
                with open(old, "rb") as f, gzip.open(old + ".gz", "wb") as g:
                    g.write(f.read())
                os.remove(old)
            self.day = day
        with open(self._path(day), "a") as f:
            f.write("\n".join(self.buf) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.buf = []
        self.last_flush = time.time()
