# `tikitorch.serialization`

TikiTorch model-bundle serialization. PyTorch `state_dict` serialization remains
available independently.

## Safe loading modes

`tt.load` always decodes bundle contents with PyTorch's restricted
`weights_only=True` loader and validates the TikiTorch format version. Public models
from `tikitorch.models` can then be reconstructed from their saved constructor
configuration:

```python
import tikitorch as tt

model = tt.models.MorphSSM(
    d_model=32,
    n_layers=1,
    task_type="classification",
    input_dim=1,
    num_classes=4,
)
config = {
    "d_model": 32,
    "n_layers": 1,
    "task_type": "classification",
    "input_dim": 1,
    "num_classes": 4,
}
tt.save(model, "model.tt", config=config)
restored = tt.load("model.tt")
```

For a custom class, construct the expected model yourself and pass `model=`. This
restores tensors without allowing checkpoint metadata to choose Python code to import:

```python
restored = MyModel(...)
tt.load("custom.tt", model=restored)
```

Custom-class reconstruction from metadata is disabled by default. `trusted=True`
permits the checkpoint to select a module and constructor, which can execute Python
code. Use it only when you trust the artifact's origin and integrity; prefer `model=`
for downloaded or user-supplied checkpoints.

Bundle constructor configuration is deliberately restricted to primitive scalars,
lists, tuples, and mappings (with string constructor-key names). `tt.save` rejects
objects such as classes and `pathlib.Path` before writing, because PyTorch's restricted
loader cannot safely reconstruct them. Convert such values to a primitive spelling and
convert them back inside your own trusted model constructor if needed.

The stored closure manifest is a structural consistency check. It is not a graph,
runtime, training-step, or physical-hardware certificate.

## Bundled model zoo

The repository's algorithmic checkpoints use a task-aware loader, and its two existing
image-sequence checkpoints predate constructor-only bundle configuration. Follow the
[model-zoo loading guide](https://github.com/nyuolab/tikitorch/blob/main/zoo/README.md)
rather than assuming every historical artifact supports automatic reconstruction with
plain `tt.load(path)`.

::: tikitorch.serialization
