"""Fresh-window, fixed-point MorphLM pilot; CSIM is never called FPGA execution.

This research runner targets the existing mode-3 ABI, not arbitrary PyTorch
models. Each update consumes a different, nonoverlapping training window.
Validation uses frozen weights (mode 3, lr=0 still executes backward) and an
independent integer forward reference. Reported LUT-normalized NLL is NOT
ordinary softmax perplexity or proof of useful/large-scale language learning.

The native library must include fpga_lm_layout.cpp with matching kernel flags.
Its probe verifies the host ABI, NOT the identity of a physical loaded image.
PCI execution additionally requires an explicit image identifier from the
launcher's independently verified image gate and a checked-transport bridge.
Legacy PCI libraries without the error-status capability are rejected.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HLS = REPO / "tikitorch/hardware/hls/forge_kernels"
GENERATOR = REPO / "tikitorch/fpga/silicon/gen_buffer_init.py"
WEIGHTS = ("EMBED", "MH", "MP", "MB", "W_HEAD")
REGIONS = WEIGHTS + ("TAP_IDX", "PROJ_IDX", "AD_HEAD", "BS_STREAM", "TOKENS",
                     "TARGETS", "GRAD_LOGIT", "PRED", "CFG", "CE_LUT")
ENTRY = b"forge_train_step_morph_lm"
I32P = ctypes.POINTER(ctypes.c_int32)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _finite_json(text):
    value = json.loads(text)

    def check(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite numeric value in comparison record")
        if isinstance(item, dict):
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)

    check(value)
    return value


def layout(width):
    """Load the shared initializer without inheriting geometry from the shell."""
    if width not in (16, 32, 64, 128, 256, 512):
        raise ValueError("width must be a supported power of two from 16 to 512")
    prior = {key: os.environ.get(key) for key in ("ML_D", "ML_BATCH")}
    try:
        os.environ.update(ML_D=str(width), ML_BATCH="1")
        spec = importlib.util.spec_from_file_location("_morph_lm_initializer", GENERATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


def layout_values(shape):
    values = [1, shape.ML_D, shape.ML_V, shape.ML_L_SEQ, shape.ML_N_LAYERS,
              shape.ML_K, shape.ML_BATCH, shape.ML_BUFFER_WORDS,
              shape.ML_BUFFER_BYTES, shape.ML_LOG2_D]
    for name in REGIONS:
        values.extend((getattr(shape, "OFF_" + name), getattr(shape, "WORDS_" + name)))
    return values


def corpus_plan(path, steps, eval_windows, sequence=128):
    raw = Path(path).read_bytes()
    if not raw or any(byte >= 128 for byte in raw):
        raise ValueError("corpus must be nonempty strict ASCII bytes; no clipping or recoding")
    # Disjoint byte partitions. Even target bytes never cross split boundaries.
    ends = (0, len(raw) * 8 // 10, len(raw) * 9 // 10, len(raw))
    splits = {name: {"start": ends[i], "stop": ends[i + 1]}
              for i, name in enumerate(("train", "validation", "test"))}
    for split in splits.values():
        split["windows"] = (split["stop"] - split["start"]) // (sequence + 1)
    if splits["train"]["windows"] < steps:
        raise ValueError("not enough disjoint training windows; this pilot never repeats a window")
    if min(splits["validation"]["windows"], splits["test"]["windows"]) < eval_windows:
        raise ValueError("not enough disjoint validation/test windows")
    return raw, {"path": str(Path(path).resolve()), "sha256": digest(raw), "bytes": len(raw),
                 "encoding": "strict ASCII128, one byte per token", "splits": splits,
                 "window_stride": sequence + 1, "test_policy": "reserved, not evaluated"}


def window(raw, dataset, split, index, sequence=128):
    part = dataset["splits"][split]
    if index < 0 or index >= part["windows"]:
        raise ValueError("window outside its dataset partition")
    start = part["start"] + index * (sequence + 1)
    return list(raw[start:start + sequence]), list(raw[start + 1:start + sequence + 1])


def source_identity():
    files = [Path(__file__), REPO / "scripts/fpga_lm_layout.cpp", GENERATOR,
             REPO / "tikitorch/fpga/bridge.cpp", HLS / "forge_train_step_morph_lm.cpp",
             HLS / "forge_buffer_morph_lm.hpp", HLS / "forge_ce_lut.hpp",
             HLS.parent / "forge_hls_types.hpp"]
    return {path.relative_to(REPO).as_posix(): digest(path.read_bytes()) for path in files}


def build_csim(directory, width=16):
    """Build an explicit new library, never load an unverified prebuilt dylib."""
    shape = layout(width)
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("a C++17 compiler is required")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    library = target / ("libmorphlm.dylib" if sys.platform == "darwin" else "libmorphlm.so")
    flags = ["-O2", "-std=c++17", "-shared", "-fPIC", "-Wno-unknown-pragmas",
             "-DCSIM_ONLY", "-DTT_KERNEL_MORPH_LM", f"-DML_D={width}"]
    if width >= 512:
        flags += ["-DTT_RESIDENT_W", "-DTT_ONCHIP_GRAD", "-DTT_URAM_W"]
    command = [compiler, *flags, f"-I{HLS}", str(REPO / "tikitorch/fpga/bridge.cpp"),
               str(HLS / "forge_train_step_morph_lm.cpp"),
               str(REPO / "scripts/fpga_lm_layout.cpp"), "-o", str(library)]
    frozen_sources = source_identity()
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError("CSIM compilation failed: " + result.stderr[-4000:])
    if source_identity() != frozen_sources:
        raise RuntimeError("source changed during CSIM compilation; discard this build and retry")
    write_json(target / "build.json", {"command": command, "source_hashes": frozen_sources,
                                      "compiler_version": version,
                                      "library_sha256": digest(library.read_bytes()),
                                      "layout": layout_values(shape)})
    return library


class NativeBridge:
    """Explicit backend, bounds-checked transfers, and no fallback."""

    def __init__(self, path, backend, shape):
        path = Path(path).resolve(strict=True)
        self.identity = {"path": str(path), "sha256": digest(path.read_bytes())}
        self.shape = shape
        if backend == "csim":
            record = json.loads((path.parent / "build.json").read_text())
            if (record.get("library_sha256") != self.identity["sha256"]
                    or record.get("source_hashes") != source_identity()
                    or record.get("layout") != layout_values(shape)):
                raise RuntimeError("CSIM build/source/layout mismatch; build a new current-source library")
            self.identity["build_record"] = record
        self.lib = ctypes.CDLL(str(path))
        try:
            probe = self.lib.tt_morph_lm_layout
            probe.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            probe.restype = ctypes.c_int
            count = probe(None, 0)
            if count != len(layout_values(shape)):
                raise RuntimeError("native layout schema mismatch")
            values = (ctypes.c_int * count)()
            if probe(values, count) != count or list(values) != layout_values(shape):
                raise RuntimeError("native/header/Python layout mismatch")
            self.lib.tt_morph_lm_backend.restype = ctypes.c_int
            if self.lib.tt_morph_lm_backend() != {"csim": 1, "pci": 2}[backend]:
                raise RuntimeError("native backend capability mismatch; fallback is forbidden")
        except AttributeError as error:
            raise RuntimeError("library lacks required fpga_lm_layout.cpp ABI probe") from error
        self.last_error = None
        if backend == "pci":
            try:
                self.last_error = self.lib.tt_fpga_last_error
                self.last_error.argtypes = [ctypes.c_void_p]
                self.last_error.restype = ctypes.c_int
            except AttributeError as error:
                raise RuntimeError("PCI library lacks checked transport; fallback is forbidden") from error
        for name, args, result in (
            ("tt_fpga_open", [ctypes.c_char_p], ctypes.c_void_p),
            ("tt_fpga_alloc", [ctypes.c_void_p, ctypes.c_long], ctypes.c_void_p),
            ("tt_fpga_write_region", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long,
                                      I32P, ctypes.c_long], None),
            ("tt_fpga_read_region", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long,
                                     I32P, ctypes.c_long], None),
            ("tt_fpga_run", [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int], ctypes.c_int),
            ("tt_fpga_free", [ctypes.c_void_p, ctypes.c_void_p], None),
            ("tt_fpga_close", [ctypes.c_void_p], None),
        ):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.handle = self.lib.tt_fpga_open(None)
        if not self.handle:
            raise RuntimeError("native device open failed")
        try:
            self._check()
        except RuntimeError:
            self.lib.tt_fpga_close(self.handle)
            raise
        self.buffer = self.lib.tt_fpga_alloc(self.handle, shape.ML_BUFFER_WORDS)
        if not self.buffer:
            self.lib.tt_fpga_close(self.handle)
            raise RuntimeError("native allocation failed")
        try:
            self._check()
        except RuntimeError:
            self.lib.tt_fpga_free(self.handle, self.buffer)
            self.lib.tt_fpga_close(self.handle)
            raise

    def _check(self):
        if self.last_error is not None:
            status = self.last_error(self.handle)
            if status:
                raise RuntimeError(f"native checked transport failed with status {status}")

    def _bounds(self, offset, count):
        if offset < 0 or count < 0 or offset + count > self.shape.ML_BUFFER_WORDS:
            raise ValueError("native transfer exceeds verified buffer bounds")

    def write(self, offset, values):
        self._bounds(offset, len(values))
        if any(not -(1 << 31) <= int(value) < (1 << 31) for value in values):
            raise ValueError("native input is outside int32")
        data = (ctypes.c_int32 * len(values))(*values)
        self.lib.tt_fpga_write_region(self.handle, self.buffer, offset, data, len(values))
        self._check()

    def read(self, offset, count):
        self._bounds(offset, count)
        data = (ctypes.c_int32 * count)()
        self.lib.tt_fpga_read_region(self.handle, self.buffer, offset, data, count)
        self._check()
        return list(data)

    def step(self, lr, threshold):
        result = self.lib.tt_fpga_run(self.handle, ENTRY, self.buffer, 3, 1, lr, threshold, -1)
        self._check()
        if result != 0:
            raise RuntimeError(f"native mode-3 failed with status {result}")

    def close(self):
        self.lib.tt_fpga_free(self.handle, self.buffer)
        self.lib.tt_fpga_close(self.handle)


def integer_forward(weights, tokens, targets, shape, zshift=4, use_context=True):
    """Independent saturated integer forward and exact balanced-LUT gradient.

    NLL uses q_v=e_v/sum(e), before the kernel's power-of-two shift/truncation.
    That named diagnostic distribution is not its rounded surrogate derivative.
    """
    import numpy as np

    if len(tokens) != shape.ML_L_SEQ or len(targets) != len(tokens):
        raise ValueError("reference requires one complete window")
    if any(not 0 <= int(value) < shape.ML_V for value in tokens + targets):
        raise ValueError("token outside ASCII vocabulary")
    d, n, k, vocab = shape.ML_D, shape.ML_N_LAYERS, shape.ML_K, shape.ML_V
    arrays = {}
    dims = {"EMBED": (vocab, d), "MH": (n, d, k), "MP": (n, d, d),
            "MB": (n, d), "W_HEAD": (vocab, d)}
    for name in WEIGHTS:
        values = np.asarray(weights[name], dtype=np.int64)
        # Kernel consumes low16 bits, sign-extended, even though storage is int32.
        arrays[name] = (((values + 32768) % 65536) - 32768).reshape(dims[name])
    history = np.zeros((n, len(tokens), d), dtype=np.int64)
    lut = np.asarray([round(65536 * 2.0 ** (-i / 16.0)) for i in range(256)], dtype=np.int64)
    predictions, gradients, nll = [], [], 0.0
    saturated = 0
    zero_targets = 0

    def sat(values):
        nonlocal saturated
        saturated += int(np.count_nonzero((values < -32768) | (values > 32767)))
        return np.clip(values, -32768, 32767)

    for position, token in enumerate(tokens):
        residual = arrays["EMBED"][token].copy()
        for layer in range(n):
            centered = sat(residual - (int(residual.sum()) >> shape.ML_LOG2_D))
            history[layer, position] = centered
            candidates = [history[layer, position - (tap << min(layer, 6))]
                          + arrays["MH"][layer, :, tap]
                          for tap in range(k if use_context else 1)
                          if position >= (tap << min(layer, 6))]
            stacked = np.stack(candidates)
            conv = sat(stacked.max(axis=0) if layer % 2 == 0 else stacked.min(axis=0))
            projection = sat((arrays["MP"][layer] + conv).max(axis=1) + arrays["MB"][layer])
            residual = sat(residual + projection)
        logits = sat((arrays["W_HEAD"] + residual).max(axis=1))
        prediction = int(logits.argmax())
        predictions.append(prediction)
        energies = lut[np.minimum((logits[prediction] - logits) >> zshift, 255)]
        total = int(energies.sum())
        nll += math.log(total) - math.log(int(energies[targets[position]]))
        shift = 0
        while total >> shift > 65536:
            shift += 1
        gradient = energies >> shift
        zero_targets += int(gradient[targets[position]] == 0)
        gradient[targets[position]] -= gradient.sum()
        gradients.extend(gradient.tolist())
    return {"predictions": predictions, "gradients": gradients,
            "lut_normalized_nll": nll / len(tokens),
            "accuracy": sum(a == b for a, b in zip(predictions, targets)) / len(tokens),
            "saturation_events": saturated, "rounded_zero_target_mass": zero_targets}


def read_weights(bridge, shape):
    return {name: bridge.read(getattr(shape, "OFF_" + name), getattr(shape, "WORDS_" + name))
            for name in WEIGHTS}


def weight_bytes(weights):
    return b"".join(struct.pack(f"<{len(weights[name])}i", *weights[name]) for name in WEIGHTS)


def stage_window(bridge, shape, tokens, targets):
    if len(tokens) != shape.ML_L_SEQ or len(targets) != shape.ML_L_SEQ:
        raise ValueError("each update needs a complete window")
    if any(not 0 <= value < shape.ML_V for value in tokens + targets):
        raise ValueError("token outside ASCII vocabulary")
    for region, values in (("TOKENS", tokens), ("TARGETS", targets)):
        offset = getattr(shape, "OFF_" + region)
        bridge.write(offset, values)
        if bridge.read(offset, len(values)) != values:
            raise RuntimeError(f"{region} transport readback mismatch")


def checkpoint(directory, step, weights, shape, payload_directory=None):
    payload = weight_bytes(weights)
    root = Path(directory).resolve()
    target = Path(payload_directory).resolve() if payload_directory else root
    path = target / f"checkpoint_{step:06d}.i32"
    with path.open("xb") as stream:
        stream.write(payload)
    record = {"step": step, "file": path.name if target == root else str(path),
              "location_portable": target == root, "sha256": digest(payload),
              "bytes": len(payload), "format": "little-endian signed int32, low16 weights",
              "regions": [{"name": name, "words": getattr(shape, "WORDS_" + name)}
                          for name in WEIGHTS]}
    write_json(root / path.with_suffix(".json").name, record)
    return record


def compare_runs(csim_directory, pci_directory):
    """Require paired protocols AND complete checkpoint payload equality.

    This comparison does not establish a physical image's identity. Both native
    runs must independently satisfy their capability/image/transport gates.
    """
    roots = [Path(csim_directory), Path(pci_directory)]
    if any((root / name).exists() for root in roots for name in ("failure.json", "cleanup_failure.json")):
        raise ValueError("failed runs cannot pass the paired comparison")
    manifests = [_finite_json((root / "manifest.json").read_text()) for root in roots]
    summaries = [_finite_json((root / "summary.json").read_text()) for root in roots]
    if [record.get("backend") for record in manifests] != ["csim", "pci"]:
        raise ValueError("comparison requires CSIM then PCI runs, with explicit backend identities")
    if not all(record.get("complete") is True for record in summaries):
        raise ValueError("both compared runs must be complete")
    for manifest in manifests:
        if manifest.get("schema") != "tikitorch.fpga_lm_pretrain.v1":
            raise ValueError("unsupported paired run schema")
        for count in (manifest.get("steps"), manifest.get("evaluation", {}).get("every_updates"),
                      manifest.get("evaluation", {}).get("windows")):
            if type(count) is not int or count < 1:
                raise ValueError("paired steps and evaluation counts must be positive integers")
    for key in ("schema", "layout", "steps", "seed", "optimizer", "source_hashes", "evaluation"):
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f"paired protocol mismatch: {key}")
    shape = layout(manifests[0]["layout"][1])
    if manifests[0]["layout"] != layout_values(shape):
        raise ValueError("paired layout does not match the supported shared-header geometry")
    expected_regions = [{"name": name, "words": getattr(shape, "WORDS_" + name)} for name in WEIGHTS]
    expected_bytes = sum(region["words"] for region in expected_regions) * 4
    for key in ("sha256", "bytes", "encoding", "splits", "window_stride", "test_policy"):
        if manifests[0]["dataset"].get(key) != manifests[1]["dataset"].get(key):
            raise ValueError(f"paired dataset mismatch: {key}")
    checkpoints = []
    for index, root in enumerate(roots):
        if (summaries[index].get("steps") != manifests[index]["steps"]
                or summaries[index].get("backend") != manifests[index]["backend"]):
            raise ValueError("summary does not match declared backend/completed steps")
        metrics = [_finite_json(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        training = [row for row in metrics if row["kind"] == "train"]
        if ([row["step"] for row in training] != list(range(1, manifests[index]["steps"] + 1))
                or [row["window_index"] for row in training] != list(range(manifests[index]["steps"]))):
            raise ValueError("paired training trajectory coverage is incomplete")
        records = [_finite_json(path.read_text()) for path in sorted(root.glob("checkpoint_*.json"))]
        expected = [0] + [step for step in range(1, manifests[0]["steps"] + 1)
                          if step % manifests[0]["evaluation"]["every_updates"] == 0
                          or step == manifests[0]["steps"]]
        if [record["step"] for record in records] != expected:
            raise ValueError("paired checkpoint coverage is incomplete")
        validation = [row for row in metrics if row["kind"] == "validation"]
        if ([row["step"] for row in validation] != expected
                or not all(row.get("weights_unchanged") is True and
                           row.get("native_reference_predictions_and_loss_gradients_equal") is True
                           for row in validation)):
            raise ValueError("paired validation gate coverage is incomplete")
        if summaries[index].get("final_checkpoint") != records[-1]:
            raise ValueError("summary final checkpoint is not the compared checkpoint")
        if any(row.get("checkpoint") != record for row, record in zip(validation, records)):
            raise ValueError("validation checkpoint links do not match the compared checkpoints")
        for record in records:
            if (record.get("bytes") != expected_bytes or record.get("regions") != expected_regions
                    or record.get("format") != "little-endian signed int32, low16 weights"):
                raise ValueError("checkpoint schema is not a complete model weight payload")
            payload_path = Path(record["file"])
            if not payload_path.is_absolute():
                payload_path = root / payload_path
            payload = payload_path.read_bytes()
            if digest(payload) != record["sha256"] or len(payload) != record["bytes"]:
                raise ValueError("checkpoint payload does not match its recorded digest/size")
        checkpoints.append(records)
    for left, right in zip(*checkpoints):
        for key in ("step", "sha256", "bytes", "format", "regions"):
            if left[key] != right[key]:
                raise ValueError(f"checkpoint {left['step']} differs: {key}")
        paths = [Path(record["file"]) if Path(record["file"]).is_absolute() else root / record["file"]
                 for record, root in zip((left, right), roots)]
        if paths[0].read_bytes() != paths[1].read_bytes():
            raise ValueError(f"checkpoint {left['step']} payload bytes differ")
    return {"complete": True, "comparison": "paired CSIM/PCI full checkpoint payload digests",
            "checkpoints": [{"step": item["step"], "sha256": item["sha256"]}
                            for item in checkpoints[0]],
            "all_checkpoint_payloads_equal": True,
            "quality_or_speed_advantage_established": False}


def baselines(raw, dataset, count, sequence):
    """Train-fitted Laplace-smoothed unigram/bigram; evaluate validation only."""
    import numpy as np

    train = dataset["splits"]["train"]
    ids = np.frombuffer(raw[train["start"]:train["stop"]], dtype=np.uint8)
    uni = np.bincount(ids, minlength=128).astype(np.float64) + 1.0
    bigram = np.ones((128, 128), dtype=np.float64)
    np.add.at(bigram, (ids[:-1], ids[1:]), 1.0)
    uni /= uni.sum()
    bigram /= bigram.sum(axis=1, keepdims=True)
    sums = {"unigram_nll": 0.0, "bigram_nll": 0.0}
    for index in range(count):
        tokens, targets = window(raw, dataset, "validation", index, sequence)
        sums["unigram_nll"] -= float(np.log(uni[targets]).sum())
        sums["bigram_nll"] -= float(np.log(bigram[tokens, targets]).sum())
    return {key: value / (count * sequence) for key, value in sums.items()}


def run(config, bridge_factory=NativeBridge):
    if config.steps < 1 or config.eval_every < 1 or config.eval_windows < 1:
        raise ValueError("steps, eval-every and eval-windows must be positive")
    if not 0 < config.lr <= 32767 or not 0 <= config.threshold < (1 << 31):
        raise ValueError("lr must be 1..32767 and threshold 0..2147483647")
    if config.backend == "pci" and (not config.image_id or not config.image_id.startswith("agfi-")):
        raise ValueError("PCI requires --image-id from an independently verified loaded-image gate")
    if config.backend == "pci" and config.width != 512:
        raise ValueError("this PCI pilot supports only the independently gated D512 image")
    shape = layout(config.width)
    raw, dataset = corpus_plan(config.corpus, config.steps, config.eval_windows, shape.ML_L_SEQ)
    manifest = {"schema": "tikitorch.fpga_lm_pretrain.v1", "backend": config.backend,
                "image_id_declared": config.image_id, "layout": layout_values(shape),
                "dataset": dataset, "steps": config.steps, "seed": config.seed,
                "optimizer": {"kind": "integer thresholded SignSGD", "lr": config.lr,
                              "threshold": config.threshold, "loss": "balanced LUT surrogate",
                              "ce_zshift": 4}, "source_hashes": source_identity(),
                "evaluation": {"split": "validation", "windows": config.eval_windows,
                               "every_updates": config.eval_every,
                               "distribution": "LUT energies normalized before gradient truncation",
                               "native_mode": "3, one step, lr=0; backward still executes",
                               "context_control": "host reference keeps only current-position convolution tap",
                               "baselines": "Laplace smoothing 1; fitted on entire training partition"},
                "scope": "small fixed-point LM pilot; not LLM quality, GPU parity or physical DSP proof",
                "clock_boundaries": {
                    "native_kick_s": "mode-3 call including host polling, excludes transfers",
                    "update_including_io_s": "stage/readback inputs, mode-3, prediction readback",
                    "end_to_end_wall_s": "includes native setup, NumPy import, baseline fit, full initialization, "
                                         "training, frozen evaluation, checkpoints and metrics I/O; excludes "
                                         "compilation, corpus loading/hashing, directory creation and final summary write"},
                "platform": platform.platform(), "python": sys.version}
    if config.plan:
        return manifest
    if not config.library or not config.output:
        raise ValueError("execution requires explicit --library and new --output directory")
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_directory = getattr(config, "checkpoint_dir", None)
    if checkpoint_directory:
        checkpoint_directory = Path(checkpoint_directory)
        checkpoint_directory.mkdir(parents=True, exist_ok=False)
    bridge = None
    started = time.perf_counter()
    timed_update = 0.0
    completed = 0
    try:
        import numpy as np

        bridge = bridge_factory(config.library, config.backend, shape)
        manifest["library"] = bridge.identity
        manifest["numpy_version"] = np.__version__
        manifest["started_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["baselines"] = baselines(raw, dataset, config.eval_windows, shape.ML_L_SEQ)
        write_json(output / "manifest.json", manifest)
        initial = shape.build_init(config.seed)
        bridge.write(0, initial)
        if bridge.read(0, len(initial)) != initial:
            raise RuntimeError("complete initialization transport readback mismatch")
        weights = read_weights(bridge, shape)
        log = (output / "metrics.jsonl").open("x", encoding="utf-8")

        def emit(record):
            log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            log.flush()

        def evaluate(step, current):
            snapshot = checkpoint(output, step, current, shape, checkpoint_directory)
            scores = []
            for index in range(config.eval_windows):
                tokens, targets = window(raw, dataset, "validation", index, shape.ML_L_SEQ)
                reference = integer_forward(current, tokens, targets, shape)
                no_context = integer_forward(current, tokens, targets, shape, use_context=False)
                reference["context_reset_lut_normalized_nll"] = no_context["lut_normalized_nll"]
                reference["context_reset_accuracy"] = no_context["accuracy"]
                stage_window(bridge, shape, tokens, targets)
                bridge.step(0, config.threshold)
                predictions = bridge.read(shape.OFF_PRED, shape.ML_L_SEQ)
                gradients = bridge.read(shape.OFF_GRAD_LOGIT, shape.WORDS_GRAD_LOGIT)
                if predictions != reference["predictions"] or gradients != reference["gradients"]:
                    raise RuntimeError("frozen native/reference forward or LUT-gradient mismatch")
                scores.append({key: value for key, value in reference.items()
                               if key not in ("predictions", "gradients")})
            after = read_weights(bridge, shape)
            if weight_bytes(after) != weight_bytes(current):
                raise RuntimeError("frozen evaluation changed weights")
            record = {"kind": "validation", "step": step, "checkpoint": snapshot,
                      "tokens": config.eval_windows * shape.ML_L_SEQ,
                      "native_reference_predictions_and_loss_gradients_equal": True,
                      "weights_unchanged": True,
                      "metrics": {key: sum(row[key] for row in scores) / len(scores)
                                  for key in scores[0]}}
            emit(record)
            return record

        try:
            initial_eval = evaluate(0, weights)
            final_eval = initial_eval
            for index in range(config.steps):
                tokens, targets = window(raw, dataset, "train", index, shape.ML_L_SEQ)
                before = time.perf_counter()
                stage_window(bridge, shape, tokens, targets)
                kick = time.perf_counter()
                bridge.step(config.lr, config.threshold)
                kick_s = time.perf_counter() - kick
                predictions = bridge.read(shape.OFF_PRED, shape.ML_L_SEQ)
                update_s = time.perf_counter() - before
                timed_update += update_s
                completed = index + 1
                emit({"kind": "train", "step": completed, "window_index": index,
                      "tokens_seen": completed * shape.ML_L_SEQ,
                      "preupdate_accuracy": sum(a == b for a, b in zip(predictions, targets))
                                            / shape.ML_L_SEQ,
                      "native_kick_s": kick_s, "update_including_io_s": update_s})
                if completed % config.eval_every == 0 or completed == config.steps:
                    weights = read_weights(bridge, shape)
                    final_eval = evaluate(completed, weights)
            wall = time.perf_counter() - started
            summary = {"complete": True, "backend": config.backend, "steps": completed,
                       "distinct_training_windows": completed,
                       "training_token_exposures": completed * shape.ML_L_SEQ,
                       "update_including_io_s": timed_update, "end_to_end_wall_s": wall,
                       "tokens_per_update_wall_s": completed * shape.ML_L_SEQ / timed_update,
                       "tokens_per_end_to_end_s": completed * shape.ML_L_SEQ / wall,
                       "initial_validation": initial_eval["metrics"],
                       "final_validation": final_eval["metrics"],
                       "final_checkpoint": final_eval["checkpoint"],
                       "heldout_test_evaluated": False,
                       "quality_advantage_established": False}
        finally:
            log.close()
        to_close, bridge = bridge, None
        to_close.close()
        wall = time.perf_counter() - started
        summary["end_to_end_wall_s"] = wall
        summary["tokens_per_end_to_end_s"] = completed * shape.ML_L_SEQ / wall
        write_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        write_json(output / "failure.json", {"complete": False, "steps_completed": completed,
                                            "error": str(error), "type": type(error).__name__})
        raise
    finally:
        if bridge is not None:
            # Preserve the primary failure while recording a secondary cleanup
            # error; no successful summary has been published on this path.
            try:
                bridge.close()
            except Exception as error:
                write_json(output / "cleanup_failure.json", {"complete": False,
                           "error": str(error), "type": type(error).__name__})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=("csim", "pci"), default="csim")
    result.add_argument("--library", type=Path)
    result.add_argument("--image-id")
    result.add_argument("--width", type=int, default=16)
    result.add_argument("--corpus", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--checkpoint-dir", type=Path,
                        help="optional new payload directory; JSON records remain in output (nonportable paths)")
    result.add_argument("--steps", type=int, default=8)
    result.add_argument("--eval-every", type=int, default=4)
    result.add_argument("--eval-windows", type=int, default=2)
    result.add_argument("--lr", type=int, default=1)
    result.add_argument("--threshold", type=int, default=2048)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--plan", action="store_true", help="offline manifest only, no native execution")
    result.add_argument("--build-csim", type=Path, metavar="NEW_DIRECTORY",
                        help="compile explicit current-source CSIM library, then exit")
    result.add_argument("--compare-runs", nargs=2, type=Path, metavar=("CSIM_RUN", "PCI_RUN"),
                        help="offline paired protocol and full checkpoint payload comparison, then exit")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.compare_runs:
        if arguments.build_csim or arguments.plan:
            raise ValueError("--compare-runs cannot be combined with --build-csim or --plan")
        print(json.dumps(compare_runs(*arguments.compare_runs), indent=2, sort_keys=True))
    elif arguments.build_csim:
        if arguments.plan:
            raise ValueError("--plan and --build-csim cannot be combined")
        print(build_csim(arguments.build_csim, arguments.width))
    else:
        if not arguments.corpus:
            raise ValueError("--corpus is required")
        print(json.dumps(run(arguments), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
