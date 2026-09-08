"""Device-resident route storage for training; portable bytes are an export.

Each field has its own byte-aligned uint8 tensor. This uses at most seven padding
bits per field, unlike the bit-contiguous portable PackedRouteTape format.
Packing uses bounded tensor intermediates, never an N-by-bit_width bit matrix.
The pure pack/unpack kernels can be compiled with torch.compile. They are portable
PyTorch kernels, not a claim of a hand-tuned CUDA implementation or speedup.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch

from .packed_tape import PackedField, PackedRouteTape, bits_for_cardinality


def _copy_metadata(value):
    """Copy JSON-like configuration without opaque calls in compiled forwards."""
    if isinstance(value, dict):
        return {k: _copy_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_metadata(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("fingerprint must contain JSON-like configuration values")


def pack_tensor(values: torch.Tensor, width: int) -> torch.Tensor:
    """Pack validated unsigned int64 codes LSB-first on their current device."""
    flat = values.reshape(-1)
    count = flat.numel()
    nbytes = (count * width + 7) // 8
    if not nbytes:
        return torch.empty(0, dtype=torch.uint8, device=values.device)
    base = torch.arange(nbytes, device=values.device) * 8
    result = torch.zeros(nbytes, dtype=torch.uint8, device=values.device)
    for bit in range(8):
        position = base + bit
        index = position // width
        code = flat[index.clamp_max(count - 1)]
        selected = ((code >> (position % width)) & 1).to(torch.uint8)
        result = result | (torch.where(index < count, selected, 0) << bit)
    return result


def unpack_tensor(payload: torch.Tensor, width: int, count: int) -> torch.Tensor:
    """Decode a field without a host transfer or a bit-expansion matrix."""
    result = torch.zeros(count, dtype=torch.long, device=payload.device)
    if not width or not count:
        return result
    start = torch.arange(count, device=payload.device) * width
    for bit in range(width):
        position = start + bit
        selected = (payload[position // 8].to(torch.long) >> (position % 8)) & 1
        result = result | (selected << bit)
    return result


class DeviceRouteTape:
    """A forward-owned route tape on CPU/CUDA, with optional bit packing.

    ``packed=False`` is the matched unpacked baseline. ``validate=False`` is
    reserved for trusted operator-produced codes with declared cardinalities;
    it avoids data-dependent GPU synchronization. External callers should keep
    validation enabled. Tapes own their storage and never retain input aliases.
    """

    def __init__(self, *, device: torch.device | str = "cpu", packed: bool = True,
                 fingerprint: Mapping[str, Any] | None = None,
                 validate: bool = True):
        self.device = torch.device(device)
        self.packed = packed
        self.validate = validate
        self._fingerprint = _copy_metadata(dict(fingerprint)) if fingerprint is not None else None
        self._fields: dict[str, PackedField] = {}
        self._storage: dict[str, torch.Tensor] = {}
        self._bit_length = 0

    @property
    def fingerprint(self):
        return _copy_metadata(self._fingerprint)

    @property
    def fields(self):
        return tuple(self._fields.values())

    @property
    def bit_length(self):
        return self._bit_length

    @property
    def nbytes(self):
        """Actual retained tensor bytes, including per-field byte padding."""
        return sum(t.numel() * t.element_size() for t in self._storage.values())

    @property
    def payload_nbytes(self):
        return self.nbytes

    def __len__(self):
        return len(self._fields)

    def __contains__(self, name):
        return name in self._fields

    def append(self, name: str, values: Any, *, bit_width: int | None = None,
               cardinality: int | None = None) -> PackedField:
        if not isinstance(name, str) or not name:
            raise ValueError("field name must be a nonempty string")
        if name in self._fields:
            raise ValueError(f"duplicate route field {name!r}")
        tensor = torch.as_tensor(values, device=self.device)
        if tensor.is_floating_point() or tensor.is_complex():
            raise TypeError("route values must be integer or boolean")
        tensor = tensor.detach().to(torch.long).contiguous()
        required = bits_for_cardinality(cardinality) if cardinality is not None else None
        if bit_width is None:
            if required is None:
                raise ValueError("device routes require cardinality or bit_width")
            bit_width = required
        if isinstance(bit_width, bool) or not isinstance(bit_width, int) or not 0 <= bit_width <= 63:
            raise ValueError("bit_width must be an integer in [0, 63]")
        if required is not None and bit_width < required:
            raise ValueError("bit_width is below the cardinality coding bound")
        if self.validate and tensor.numel():
            maximum, minimum = int(tensor.max()), int(tensor.min())
            if minimum < 0 or maximum >= (1 << bit_width):
                raise ValueError("route does not fit its unsigned bit width")
            if cardinality is not None and maximum >= cardinality:
                raise ValueError("route is outside its declared cardinality")
        field = PackedField(name, tuple(tensor.shape), bit_width,
                            self._bit_length, cardinality)
        self._storage[name] = pack_tensor(tensor, bit_width) if self.packed else tensor.clone()
        self._fields[name] = field
        self._bit_length += field.bit_length
        return field

    add = append

    def append_signed(self, name, values, *, lo: int, hi: int):
        if isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(lo, int) or not isinstance(hi, int) or hi < lo:
            raise ValueError("invalid signed range")
        tensor = torch.as_tensor(values, device=self.device)
        if tensor.is_floating_point() or tensor.is_complex():
            raise TypeError("route values must be integer or boolean")
        tensor = tensor.to(torch.long)
        if not -(1 << 63) <= lo <= hi < (1 << 63):
            raise ValueError("signed range must fit int64")
        if self.validate and tensor.numel() and (int(tensor.min()) < lo or int(tensor.max()) > hi):
            raise ValueError("signed route is outside its declared range")
        field = self.append(name, tensor - lo, cardinality=hi - lo + 1)
        field = PackedField(field.name, field.shape, field.bit_width,
                            field.bit_offset, field.cardinality, value_offset=lo)
        self._fields[name] = field
        return field

    def append_mask(self, name, values):
        return self.append(name, values, cardinality=2)

    def read(self, name: str, *, device=None):
        field = self._fields[name]
        storage = self._storage[name]
        decoded = unpack_tensor(storage, field.bit_width, field.count) if self.packed else storage.clone()
        decoded = decoded.reshape(field.shape)
        if field.value_offset:
            decoded = decoded + field.value_offset
        return decoded if device is None else decoded.to(device)

    unpack = read

    def to(self, device):
        target = type(self)(device=device, packed=self.packed,
                            fingerprint=self._fingerprint, validate=self.validate)
        target._fields = dict(self._fields)
        target._bit_length = self._bit_length
        target._storage = {name: t.to(device).clone() for name, t in self._storage.items()}
        return target

    def clone(self):
        return self.to(self.device)

    def to_bytes(self, *, fingerprint=None):
        binding = self._fingerprint if fingerprint is None else dict(fingerprint)
        if self._fingerprint is not None and binding != self._fingerprint:
            raise ValueError("route tape fingerprint mismatch: refusing rebinding")
        portable = PackedRouteTape()
        for field in self.fields:
            portable.append(field.name, self.read(field.name) - field.value_offset,
                            bit_width=field.bit_width, cardinality=field.cardinality)
            portable._fields[field.name] = replace(portable._fields[field.name],
                                                    value_offset=field.value_offset)
        return portable.to_bytes(fingerprint=binding)

    serialize = to_bytes

    @classmethod
    def from_bytes(cls, blob, *, device="cpu", packed=True, expected_fingerprint=None):
        portable = PackedRouteTape.from_bytes(blob, expected_fingerprint=expected_fingerprint)
        tape = cls(device=device, packed=packed, fingerprint=portable.fingerprint)
        for field in portable.fields:
            tape.append(field.name, portable.read(field.name) - field.value_offset,
                        bit_width=field.bit_width, cardinality=field.cardinality)
            tape._fields[field.name] = replace(tape._fields[field.name],
                                               value_offset=field.value_offset)
        return tape

    def state_dict(self):
        return {"record": self.to_bytes()}

    @classmethod
    def from_state_dict(cls, state, **kwargs):
        return cls.from_bytes(state["record"], **kwargs)
