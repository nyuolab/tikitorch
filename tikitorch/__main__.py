"""``python -m tikitorch <ModelSpec>`` — print a closure/hardware profile.

Builds a named built-in tropical model (small, dependency-light config),
synthesizes a matching example input, traces a forward pass, and prints the
:func:`tikitorch.profile.format_report` output. The headline line reads
``multiplications: 0`` for a closed tropical model.

Examples::

    python -m tikitorch TropicalCNN
    python -m tikitorch MorphSSM
    python -m tikitorch --list
"""
import argparse
import sys

import torch

import tikitorch as tt
from tikitorch.profile import format_report, profile_model


def _build(spec: str):
    """Return ``(model, example_inputs)`` for a named built-in spec.

    Each entry constructs a deliberately small instance and a matching example
    input so a forward pass is cheap. Names are matched case-insensitively.
    """
    models = tt.models
    name = spec.lower()

    if name == "tropicalcnn":
        model = models.TropicalCNN(in_channels=1, num_classes=10)
        x = torch.rand(1, 1, 28, 28)
    elif name == "morphssm":
        model = models.MorphSSM(d_model=64, n_layers=2, task_type="lm",
                                vocab_size=128, use_soft=False)
        x = torch.randint(0, 128, (1, 16))
    elif name == "tropicalmlp":
        model = models.TropicalMLP(d_model=64, n_layers=2, input_dim=32,
                                   num_classes=10)
        x = torch.rand(2, 32)
    elif name == "morphmlp":
        model = models.MorphMLP(d_model=64, n_layers=2, input_dim=32,
                                num_classes=10)
        x = torch.rand(2, 32)
    elif name == "tropicalsequencemodel":
        model = models.TropicalSequenceModel(d_model=64, n_layers=2,
                                             task_type="lm", vocab_size=128)
        x = torch.randint(0, 128, (1, 16))
    elif name == "tropicaltransformer":
        model = models.TropicalTransformer(d_model=64, n_heads=2, n_layers=2,
                                           task_type="lm", vocab_size=128)
        x = torch.randint(0, 128, (1, 16))
    elif name in ("mote", "motemodel"):
        model = models.MoTEModel(d_model=64, n_experts=4, n_layers=2,
                                 task_type="lm", vocab_size=128)
        x = torch.randint(0, 128, (1, 16))
    else:
        return None, None
    return model, x


BUILTINS = (
    "TropicalCNN", "MorphSSM", "TropicalMLP", "MorphMLP",
    "TropicalSequenceModel", "TropicalTransformer", "MoTEModel",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tikitorch",
        description="Print a closure / hardware-cost profile for a built-in model.",
    )
    parser.add_argument("model", nargs="?", default=None,
                        help="built-in model name, e.g. TropicalCNN or MorphSSM")
    parser.add_argument("--list", action="store_true",
                        help="list the available built-in model names and exit")
    args = parser.parse_args(argv)

    if args.list or args.model is None:
        print("available built-in models:")
        for n in BUILTINS:
            print(f"    {n}")
        # No model given is not an error: we just showed the menu.
        return 0 if args.list else (0 if args.model is None else 1)

    model, x = _build(args.model)
    if model is None:
        print(f"unknown model spec: {args.model!r}", file=sys.stderr)
        print("available: " + ", ".join(BUILTINS), file=sys.stderr)
        return 2

    report = profile_model(model, example_inputs=x)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
