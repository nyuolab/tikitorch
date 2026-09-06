# `tikitorch.algorithmic`

The tracked algorithmic-reasoning checkpoints need their saved task definition as
well as ordinary constructor arguments. Load them with `load_reasoner`, not plain
`tt.load`:

```python
from tikitorch.algorithmic import load_reasoner

model = load_reasoner("zoo/algorithmic/running_max.tt")
```

See the [model-zoo guide](https://github.com/nyuolab/tikitorch/blob/main/zoo/README.md)
for every shipped checkpoint, its recorded metrics, and the legacy image-sequence
loading path.

::: tikitorch.algorithmic
