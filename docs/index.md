# TikiTorch

A tropical-semiring deep-learning library — **multiplier-free, FPGA-native
training**, modeled closely on PyTorch. Every operation in the forward *and*
backward pass — and the optimizer — reduces to compare / add / bit-shift, so the
whole learning loop is algebraically closed and maps to FPGA fabric with **zero
DSP blocks**.

```bash
pip install -e .
```

```python
import torch
import tikitorch as tt
import tikitorch.optim as optim

model = tt.models.MorphSSM(d_model=128, n_layers=4, task_type="lm", vocab_size=128)
opt = optim.PureTropicalSGD(model.parameters())     # power-of-2 lr = bit-shift, no DSPs
loss = model.compute_loss(torch.randint(0, 128, (8, 64)), torch.randint(0, 128, (8, 64)))
loss.backward(); opt.step(); opt.zero_grad()
```

- **Full story, benchmarks & how it compares:** the
  [README](https://github.com/nyuolab/tikitorch#readme).
- **Verify it's multiplier-free / train on the FPGA kernel with no board:**
  `bash scripts/build_fpga_local.sh` then `python examples/fpga_quickstart.py`.
- **Prove closure on any model:** `tikitorch.audit.assert_closed(model)` /
  `find_runtime_multiplies` / `closure_report` (see **API reference → audit**).
- **Contributing & the closure invariant:**
  [CONTRIBUTING](https://github.com/nyuolab/tikitorch/blob/main/CONTRIBUTING.md).

Browse the **API reference** in the navigation — it is generated from the
in-source docstrings, so it always matches the code.
