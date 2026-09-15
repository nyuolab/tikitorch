"""Source-bound, sanitized CPU-only integer error-feedback SGD v1.

This is a NEW optimizer, not the SignSGD algorithm in the existing FPGA image.
Native arithmetic runs exclusively in an owned child process: a UBSan failure,
timeout, clipping event, or protocol error poisons that process and forbids a
successful checkpoint. There is no AWS, native fallback, or training study CLI.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import struct
import subprocess
import time

from scripts import fpga_lm_pretrain as pilot

REPO = Path(__file__).resolve().parents[2]
CPP = Path(__file__).with_name("fpga_lm_efsgd_v1.cpp")
SCHEMA = "tikitorch.fpga_lm_efsgd.v1"
MAGIC = 0x45465331
STATE_HEADER = 20
WEIGHTS = pilot.WEIGHTS


def source_identity():
    result = pilot.source_identity()
    for path in (Path(__file__).resolve(), CPP):
        result[path.relative_to(REPO).as_posix()] = pilot.digest(path.read_bytes())
    return result


def source_digest(sources=None):
    value = source_identity() if sources is None else sources
    return pilot.digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(label + " must be an exact integer in the declared range")
    return value


def scalar_update(weight, remainder, gradient, k):
    """Independent scalar oracle; truncation toward zero, no deadzone/clipping."""
    _integer(k, 0, 30, "k")
    _integer(weight, -32768, 32767, "weight")
    _integer(remainder, -(1 << k) + 1, (1 << k) - 1, "remainder")
    _integer(gradient, -(1 << 31), (1 << 31) - 1, "gradient")
    accumulator = remainder + gradient
    if not -(1 << 31) <= accumulator < (1 << 31):
        raise OverflowError("int32 gradient accumulation would overflow")
    update = (abs(accumulator) >> k) * (-1 if accumulator < 0 else 1)
    candidate = weight - update
    if not -32768 <= candidate <= 32767:
        raise OverflowError("weight update would clip")
    return candidate, accumulator - update * (1 << k), update


def _flags(width, sources):
    return ["-O2", "-std=c++17", "-Wno-unknown-pragmas", "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined", "-fno-omit-frame-pointer", "-DCSIM_ONLY",
            "-DTT_KERNEL_MORPH_LM", "-DTT_RESIDENT_W", "-DTT_ONCHIP_GRAD", "-DTT_URAM_W",
            "-DML_D=%d" % width, '-DEF_SOURCE_SHA256="%s"' % source_digest(sources)]


def build(directory, width=512):
    """Build one new exclusive, always-sanitized executable from exact sources."""
    _integer(width, 16, 512, "width")
    shape = pilot.layout(width)
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("C++17 compiler with undefined-behavior sanitizer required")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    executable = directory / "fpga_lm_efsgd_v1"
    sources = source_identity()
    command = [compiler, *_flags(width, sources), "-I" + str(pilot.HLS), str(CPP), "-o", str(executable)]
    try:
        version = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                                 check=True, timeout=10).stdout.strip()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=120)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        with (directory / "compiler.log").open("xb") as handle:
            handle.write(stdout + stderr)
        if process.returncode:
            raise RuntimeError("sanitized native compilation failed: " + stderr.decode(errors="replace")[-4000:])
        if source_identity() != sources:
            raise RuntimeError("source changed while compiling native EFSGD")
        pilot.write_json(directory / "build.json", {
            "schema": SCHEMA, "cpu_only": True, "existing_afi_compatible": False,
            "command": command, "compiler_version": version, "source_hashes": sources,
            "source_sha256": source_digest(sources), "layout": pilot.layout_values(shape),
            "executable_sha256": pilot.digest(executable.read_bytes()),
            "sanitizer": "undefined, nonrecovering", "compiler_timeout_policy": "owned process group",
            "native_process_timeout_policy": "inherits enclosing worker group; exact-child kill/reap"})
        return executable
    except BaseException as error:
        pilot.write_json(directory / "failure.json", {"error": str(error), "type": type(error).__name__})
        raise


def _pack(values):
    if any(type(value) is not int or not -(1 << 31) <= value < (1 << 31) for value in values):
        raise ValueError("native words must be exact signed int32 integers")
    return struct.pack("<%di" % len(values), *values)


def _unpack(payload):
    if not isinstance(payload, bytes) or len(payload) % 4:
        raise ValueError("native state must be complete little-endian int32 bytes")
    return list(struct.unpack("<%di" % (len(payload) // 4), payload))


def decode_state(payload, shape, sources=None):
    values = _unpack(payload)
    count = sum(getattr(shape, "WORDS_" + name) for name in WEIGHTS)
    if len(values) != STATE_HEADER + shape.WORDS_CE_LUT + 2 * count:
        raise ValueError("full weights/remainders state byte coverage differs")
    if (values[:5] != [MAGIC, 1, shape.ML_D, shape.ML_BUFFER_WORDS, count]
            or not 0 <= values[5] <= 30 or values[6] not in (0, 1) or not 0 <= values[7] <= 30
            or values[10:12] != [shape.WORDS_CE_LUT, 0]):
        raise ValueError("native state version/shape/configuration differs")
    identity = "".join("%08x" % (word & 0xFFFFFFFF) for word in values[12:20])
    if identity != source_digest(sources):
        raise ValueError("native state is bound to different source bytes")
    lut = values[STATE_HEADER:STATE_HEADER + shape.WORDS_CE_LUT]
    if lut != [round(65536 * 2.0 ** (-i / 16.0)) for i in range(256)]:
        raise ValueError("native state loss LUT differs from frozen recipe")
    offset = STATE_HEADER + shape.WORDS_CE_LUT
    sections = {}
    for section in ("weights", "remainders"):
        sections[section] = {}
        for name in WEIGHTS:
            length = getattr(shape, "WORDS_" + name)
            row = values[offset:offset + length]
            low, high = (-32768, 32767) if section == "weights" else (-(1 << values[5]) + 1, (1 << values[5]) - 1)
            if any(not low <= value <= high for value in row):
                raise ValueError("native " + section + " violates representable bounds")
            sections[section][name] = row
            offset += length
    return {"schema": SCHEMA, "k": values[5], "step": (values[8] & 0xFFFFFFFF) | ((values[9] & 0xFFFFFFFF) << 32),
            "config": {"loss_kind": values[6], "zshift": values[7], "lut": lut},
            "source_sha256": identity, **sections}


class EFSGD:
    """One owned CPU process; no library is ever loaded into the caller."""

    def __init__(self, executable, shape, initial_buffer, k, *, timeout_s=60, log_path):
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 900:
            raise ValueError("native operation timeout must be finite in (0,900]")
        self.shape, self.timeout_s = shape, float(timeout_s)
        self.executable = Path(executable).resolve(strict=True)
        self.sources = source_identity()
        self.build_path = self.executable.parent / "build.json"
        self.build_bytes = self.build_path.read_bytes()
        self.build_record = pilot._finite_json(self.build_bytes.decode())
        self.executable_hash = pilot.digest(self.executable.read_bytes())
        record = self.build_record
        expected_command = [record["command"][0], *_flags(shape.ML_D, self.sources), "-I" + str(pilot.HLS),
                            str(CPP), "-o", str(self.executable)]
        if (record.get("schema") != SCHEMA or record.get("source_hashes") != self.sources
                or record.get("source_sha256") != source_digest(self.sources)
                or record.get("layout") != pilot.layout_values(shape)
                or record.get("command") != expected_command
                or record.get("sanitizer") != "undefined, nonrecovering"
                or record.get("executable_sha256") != self.executable_hash
                or record.get("cpu_only") is not True or record.get("existing_afi_compatible") is not False
                or (self.executable.parent / "failure.json").exists()):
            raise ValueError("native EFSGD source/build/layout/sanitizer binding differs")
        self.identity = {"schema": SCHEMA, "executable_sha256": self.executable_hash,
                         "build_sha256": pilot.digest(self.build_bytes), "source_hashes": self.sources,
                         "layout": pilot.layout_values(shape), "cpu_only": True, "existing_afi_compatible": False}
        self.poisoned, self.closed = False, False
        self.log = Path(log_path).open("xb")
        environment = dict(os.environ)
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        try:
            self.process = subprocess.Popen([str(self.executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, bufsize=0, start_new_session=False, env=environment)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            if _unpack(self._exchange(7, b"")) != pilot.layout_values(shape):
                raise RuntimeError("compiled header/native layout probe differs")
            self.reset(initial_buffer, k)
        except BaseException:
            self._poison()
            self.close()
            raise

    def _check_identity(self):
        if (source_identity() != self.sources or self.build_path.read_bytes() != self.build_bytes
                or pilot.digest(self.executable.read_bytes()) != self.executable_hash):
            raise RuntimeError("EFSGD sources/build/executable changed during execution")

    def _poison(self):
        self.poisoned = True
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        if process is not None:
            process.wait(timeout=10)

    def _transfer(self, stream, size, deadline, data=None):
        result, written = bytearray(), 0
        with selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_READ if data is None else selectors.EVENT_WRITE)
            while (len(result) if data is None else written) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("bounded native EFSGD operation timed out")
                if data is None:
                    chunk = os.read(stream.fileno(), min(size - len(result), 1024 * 1024))
                    if not chunk:
                        raise RuntimeError("native EFSGD process ended before a complete response; inspect native log")
                    result.extend(chunk)
                else:
                    written += os.write(stream.fileno(), data[written:written + 1024 * 1024])
        return bytes(result)

    def _exchange(self, operation, payload):
        if self.poisoned or self.closed:
            raise RuntimeError("native EFSGD state is poisoned or closed; create a fresh process")
        try:
            self._check_identity()
            deadline = time.monotonic() + self.timeout_s
            packet = struct.pack("<III", MAGIC, operation, len(payload) // 4) + payload
            self._transfer(self.process.stdin, len(packet), deadline, packet)
            header = self._transfer(self.process.stdout, 12, deadline)
            magic, status, count = struct.unpack("<III", header)
            maximum = STATE_HEADER + self.shape.WORDS_CE_LUT + 2 * sum(
                getattr(self.shape, "WORDS_" + name) for name in WEIGHTS)
            if magic != MAGIC or status or count > max(maximum, self.shape.ML_BUFFER_WORDS):
                raise RuntimeError("native EFSGD failed closed: status=%d, count=%d" % (status, count))
            result = self._transfer(self.process.stdout, count * 4, deadline)
            self._check_identity()
            return result
        except BaseException:
            self._poison()
            raise

    def reset(self, initial_buffer, k):
        _integer(k, 0, 30, "k")
        if len(initial_buffer) != self.shape.ML_BUFFER_WORDS:
            raise ValueError("initial buffer must exactly match the complete compiled layout")
        if self._exchange(1, _pack([k] + list(initial_buffer))):
            self._poison()
            raise RuntimeError("reset returned unexpected payload")

    def _window(self, tokens, targets, training, routes):
        if (type(routes) is not bool or len(tokens) != self.shape.ML_L_SEQ
                or len(targets) != self.shape.ML_L_SEQ):
            raise ValueError("one complete token/target window and literal route flag required")
        for value in list(tokens) + list(targets):
            _integer(value, 0, self.shape.ML_V - 1, "ASCII token")
        values = _unpack(self._exchange(2 if training else 3, _pack([int(routes)] + list(tokens) + list(targets))))
        route_names = ("AD_HEAD", "PROJ_IDX", "TAP_IDX") if routes else ()
        counts = [("predictions", self.shape.WORDS_PRED), ("gradients", self.shape.WORDS_GRAD_LOGIT)]
        counts += [(name, getattr(self.shape, "WORDS_" + name)) for name in route_names]
        if len(values) != 2 + sum(count for _, count in counts):
            self._poison()
            raise RuntimeError("native result byte coverage differs")
        result = {"step": (values[0] & 0xFFFFFFFF) | ((values[1] & 0xFFFFFFFF) << 32)}
        offset = 2
        for name, count in counts:
            result[name] = values[offset:offset + count]
            offset += count
        result["native_routes"] = {name: result.pop(name) for name in route_names}
        return result

    def train(self, tokens, targets, *, routes=False):
        return self._window(tokens, targets, True, routes)

    def evaluate(self, tokens, targets, *, routes=True):
        return self._window(tokens, targets, False, routes)

    def export_state(self):
        payload = self._exchange(4, b"")
        try:
            decode_state(payload, self.shape, self.sources)
        except BaseException:
            self._poison()
            raise
        return payload

    def import_state(self, payload):
        decode_state(payload, self.shape, self.sources)
        if self._exchange(5, payload):
            self._poison()
            raise RuntimeError("import returned unexpected payload")
        if self.export_state() != payload:
            self._poison()
            raise RuntimeError("complete native optimizer-state readback differs after import")

    def save_checkpoint(self, directory, stem):
        if not isinstance(stem, str) or not stem or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in stem):
            raise ValueError("checkpoint stem must be a simple filename component")
        directory = Path(directory)
        payload = self.export_state()
        state = decode_state(payload, self.shape, self.sources)
        record = {"schema": SCHEMA, "file": stem + ".i32", "bytes": len(payload),
                  "sha256": hashlib.sha256(payload).hexdigest(), "step": state["step"], "k": state["k"],
                  "config": state["config"], "identity": self.identity,
                  "regions": [{"name": name, "weight_words": getattr(self.shape, "WORDS_" + name),
                               "remainder_words": getattr(self.shape, "WORDS_" + name)} for name in WEIGHTS],
                  "format": "v1 little-endian int32 header+source digest+LUT+full weights+full remainders"}
        if (directory / (stem + ".json")).exists():
            raise FileExistsError("checkpoint metadata already exists")
        with (directory / record["file"]).open("xb") as handle:
            handle.write(payload)
        pilot.write_json(directory / (stem + ".json"), record)
        return record

    def close(self):
        if self.closed:
            return
        self.closed = True
        process = getattr(self, "process", None)
        try:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                process.stdin.close()
                process.stdout.close()
        finally:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
