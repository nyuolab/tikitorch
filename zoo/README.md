# Tracked model zoo

This directory contains small research checkpoints retained with the source tree. They
are reference artifacts, not a claim of state-of-the-art accuracy or a promise that all
historical bundles share one reconstruction path. The recorded closure manifests are
structural consistency snapshots, not proofs about arbitrary inputs or physical DSP
use.

All examples below use TikiTorch's restricted `weights_only=True` decoding. You do not
need `trusted=True` for any tracked checkpoint.

## Algorithmic reasoners

`algorithmic/` contains `running_max.tt`, `sort.tt`, `subset_sum.tt`, `knapsack.tt`,
and `min_coin_change.tt`. A reasoner constructor takes a `Task` object, and each bundle
also records experiment metadata, so load these files with the task-aware public
helper:

```python
from tikitorch.algorithmic import load_reasoner

model = load_reasoner("zoo/algorithmic/running_max.tt")
model.eval()
```

The complete per-file metrics and training configuration are in
[`algorithmic/zoo_manifest.json`](algorithmic/zoo_manifest.json). In particular, the
tracked models vary widely in measured quality; the manifest values should be reported
as written rather than treating membership in the zoo as an accuracy claim.

## Sequential-image MorphSSM checkpoints

The two image-sequence bundles predate the current constructor-only checkpoint
convention. Their safe tensor contents are loadable, but their saved `config` also
contains task, optimizer, recipe, and metric fields that are not `MorphSSM` constructor
arguments. Consequently, use the standard existing-instance form of `tt.load`:

```python
import json
from pathlib import Path

import tikitorch as tt

task = "smnist"  # or "scifar"
root = Path("zoo/image_seq")
metrics = json.loads((root / f"{task}_metrics.json").read_text())

model = tt.models.MorphSSM(
    d_model=metrics["d_model"],
    n_layers=metrics["n_layers"],
    dropout=0.0,
    tau=0.0,
    use_soft=False,
    out_proj_type="tropical",
    norm_type="tropical",
    head_type="tropical",
    input_type="tropical",
    task_type="classification",
    input_dim=metrics["input_dim"],
    num_classes=metrics["num_classes"],
)
tt.load(root / f"{task}_morphssm.tt", model=model)
model.eval()
```

| Task | Checkpoint | Input sequence | Recorded best hard-route accuracy |
| --- | --- | --- | ---: |
| Sequential MNIST | `image_seq/smnist_morphssm.tt` | 784 scalar pixels | 0.3429 |
| Sequential CIFAR-10 | `image_seq/scifar_morphssm.tt` | 1024 RGB triples | 0.3203 |

Those values come from the adjacent `*_metrics.json` sidecars. They are modest reference
runs and should not be presented as competitive vision results. The legacy sidecars use
the label `recipe: "closed"`, but record only `inference_closed`. The runner used a soft
relaxation and ordinary classification loss during fitting, so that label is not a
full-training-step or zero-DSP certificate.

Future checkpoints
written by [`scripts/run_closed_benchmarks.py`](../scripts/run_closed_benchmarks.py)
store only clean `MorphSSM` constructor arguments in the `.tt` bundle, leaving the
optimizer, recipe, effective batch, and measured accuracy in the JSON sidecar. Such new
bundles can be reconstructed directly with `tt.load(path)`. New sidecars call the recipe
`soft_to_hard`, explicitly record `training_zero_dsp: false`, and reserve
`inference_closed` for the checked hard inference path.
