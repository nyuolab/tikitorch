"""tikitorch.hardware — HLS kernels and the FPGA/silicon build guide.

The actual artifacts are data files shipped alongside this package:

    hls/forge_hls_types.hpp           portability shim (ap_int under Vitis HLS,
                                      int16_t/int32_t fallback for plain g++)
    hls/forge_kernels/*.cpp           fused tropical training-step kernels
                                      (1/2/3-layer tropical LM; MorphSSM core
                                      small + PTB-XL ECG configs)
    hls/forge_kernels/*_tb.cpp        standalone g++ testbenches
    hls/forge_kernels/*.hpp           shared flat int32 buffer layouts
    hls/forge_kernels/synth*.tcl      Vitis HLS synthesis scripts
    README.md                         closure->LUT mapping + HLS->AFI->F2 guide
    BUILD_GUIDE.md                    condensed HLS -> AFI -> F2 walkthrough

A historical ``forge_train_step_morph_ecg`` build records the fused MorphSSM
core forward/reverse/update at 1,207 tok/s, DSP=0, and WNS=+0.009 ns. Its
5.8 mJ/token value is derived rather than measured full-card energy, and the
embedding, classification head, and loss remain host-owned. These are
configuration-specific core measurements, not an end-to-end system guarantee.

Use :func:`hls_dir` / :func:`readme_path` / :func:`build_guide_path` to locate
these on disk after installation (they are included as package data).
"""

import os

__all__ = ["hls_dir", "readme_path", "build_guide_path"]

_HERE = os.path.dirname(os.path.abspath(__file__))


def hls_dir() -> str:
    """Absolute path to the bundled HLS kernel directory."""
    return os.path.join(_HERE, "hls")


def readme_path() -> str:
    """Absolute path to the hardware README (closure->LUT mapping + build flow)."""
    return os.path.join(_HERE, "README.md")


def build_guide_path() -> str:
    """Absolute path to the condensed HLS -> AFI -> silicon build guide."""
    return os.path.join(_HERE, "BUILD_GUIDE.md")
