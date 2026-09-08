"""Bit-packed route fields and a bounded chronological tape buffer.

The binary format is intentionally small and self-describing.  Route values are
unsigned integers packed least-significant bit first, without byte alignment
between fields.  A JSON header records field names, shapes, widths, offsets and
optional route cardinalities; the payload contains no activation values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import json
import struct
from typing import Any, Iterator, Mapping

import numpy as np
import torch


_MAGIC = b"RST1"


def bits_for_cardinality(cardinality: int) -> int:
    """Worst-case fixed-width bits needed to distinguish ``cardinality`` codes."""

    if isinstance(cardinality, bool) or not isinstance(cardinality, int) or cardinality <= 0:
        raise ValueError("cardinality must be a positive integer")
    return (cardinality - 1).bit_length()


route_code_lower_bound = bits_for_cardinality


@dataclass(frozen=True)
class PackedField:
    """Metadata for one field in a :class:`PackedRouteTape`."""

    name: str
    shape: tuple[int, ...]
    bit_width: int
    bit_offset: int
    cardinality: int | None = None
    # ontogenic_v2 (format v2): decoded value = stored_unsigned + value_offset.
    # Zero = v1 semantics; nonzero enables signed records (e.g. the dyadic
    # norm shifts in [-8, 8]).
    value_offset: int = 0

    @property
    def count(self) -> int:
        count = 1
        for size in self.shape:
            count *= size
        return count

    @property
    def bit_length(self) -> int:
        return self.count * self.bit_width


def _write_unsigned(payload: bytearray, bit_offset: int, bit_width: int, value: int) -> None:
    remaining = bit_width
    position = bit_offset
    while remaining:
        byte_index = position // 8
        intra_byte = position % 8
        take = min(remaining, 8 - intra_byte)
        mask = (1 << take) - 1
        payload[byte_index] |= (value & mask) << intra_byte
        value >>= take
        position += take
        remaining -= take


def _read_unsigned(payload: bytes | bytearray, bit_offset: int, bit_width: int) -> int:
    remaining = bit_width
    position = bit_offset
    value = 0
    output_shift = 0
    while remaining:
        byte_index = position // 8
        intra_byte = position % 8
        take = min(remaining, 8 - intra_byte)
        mask = (1 << take) - 1
        value |= ((payload[byte_index] >> intra_byte) & mask) << output_shift
        position += take
        output_shift += take
        remaining -= take
    return value


class PackedRouteTape:
    """A collection of named unsigned route fields packed at bit granularity."""

    def __init__(self) -> None:
        self._payload = bytearray()
        self._fields: dict[str, PackedField] = {}
        self._bit_length = 0
        self._fingerprint: dict[str, Any] | None = None

    @property
    def fingerprint(self) -> dict[str, Any] | None:
        """A defensive copy of the optional configuration binding."""
        return deepcopy(self._fingerprint)

    @property
    def bit_length(self) -> int:
        return self._bit_length

    @property
    def payload_nbytes(self) -> int:
        return len(self._payload)

    @property
    def nbytes(self) -> int:
        """Packed payload bytes, excluding the self-describing serialized header."""

        return self.payload_nbytes

    @property
    def fields(self) -> tuple[PackedField, ...]:
        return tuple(self._fields.values())

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def append(
        self,
        name: str,
        values: Any,
        *,
        bit_width: int | None = None,
        cardinality: int | None = None,
    ) -> PackedField:
        """Append a tensor-like unsigned field and return its immutable metadata.

        Supplying ``cardinality`` uses the theorem's fixed-width bound
        ``ceil(log2(cardinality))``.  Cardinality one therefore consumes zero
        payload bits and can encode only zero.
        """

        if not isinstance(name, str) or not name:
            raise ValueError("field name must be a nonempty string")
        if name in self._fields:
            raise ValueError(f"duplicate route field {name!r}")
        tensor = torch.as_tensor(values, dtype=torch.long, device="cpu").contiguous()
        if tensor.numel() and int(tensor.min()) < 0:
            raise ValueError("packed route fields must be unsigned")
        maximum = int(tensor.max()) if tensor.numel() else 0

        if cardinality is not None:
            required = bits_for_cardinality(cardinality)
            if maximum >= cardinality:
                raise ValueError(f"field value {maximum} is outside cardinality {cardinality}")
            if bit_width is None:
                bit_width = required
            elif bit_width < required:
                raise ValueError(
                    f"bit_width={bit_width} cannot encode cardinality={cardinality}"
                )
        if bit_width is None:
            bit_width = max(1, maximum.bit_length())
        if isinstance(bit_width, bool) or not isinstance(bit_width, int) or bit_width < 0:
            raise ValueError("bit_width must be a nonnegative integer")
        if bit_width == 0 and maximum != 0:
            raise ValueError("a zero-width field can encode only zero")
        if bit_width and maximum >= (1 << bit_width):
            raise ValueError(f"field value {maximum} does not fit in {bit_width} bits")

        shape = tuple(int(size) for size in tensor.shape)
        field = PackedField(name, shape, bit_width, self._bit_length, cardinality)
        new_bit_length = self._bit_length + field.bit_length
        required_bytes = (new_bit_length + 7) // 8
        self._payload.extend(b"\x00" * (required_bytes - len(self._payload)))
        if bit_width:
            # Vectorized LSB-first packing (ontogenic_v2 Exp-1 item 5):
            # element bits laid out LSB-first per value == the sequential
            # _write_unsigned stream. Golden-equality vs the loop is
            # test-enforced; the loop is retained for the fuzz oracle.
            vals = tensor.reshape(-1).numpy().astype(np.uint64)
            bits = ((vals[:, None] >> np.arange(bit_width, dtype=np.uint64))
                    & np.uint64(1)).astype(np.uint8).reshape(-1)
            head = self._bit_length % 8
            padded = np.concatenate(
                [np.zeros(head, dtype=np.uint8), bits])
            packed = np.packbits(padded, bitorder="little")
            start_byte = self._bit_length // 8
            if head:
                self._payload[start_byte] |= int(packed[0])
                chunk = packed[1:]
                start_byte += 1
            else:
                chunk = packed
            self._payload[start_byte:start_byte + len(chunk)] = \
                chunk.tobytes()
        self._fields[name] = field
        self._bit_length = new_bit_length
        return field

    add = append

    def append_signed(self, name: str, values: Any, *, lo: int,
                      hi: int) -> PackedField:
        """Append a signed integer field with declared range ``[lo, hi]``
        (inclusive). Stored as ``value - lo`` with cardinality
        ``hi - lo + 1``; :meth:`read` restores the offset. (ontogenic_v2
        Exp-1 item 2 — e.g. the 17-code dyadic norm shifts.)"""

        if not isinstance(lo, int) or not isinstance(hi, int) or hi < lo:
            raise ValueError(f"invalid signed range [{lo}, {hi}]")
        tensor = torch.as_tensor(values, dtype=torch.long,
                                 device="cpu").contiguous()
        if tensor.numel():
            mn, mx = int(tensor.min()), int(tensor.max())
            if mn < lo or mx > hi:
                raise ValueError(
                    f"signed field values [{mn}, {mx}] outside "
                    f"declared range [{lo}, {hi}]")
        field = self.append(name, tensor - lo, cardinality=hi - lo + 1)
        shifted = PackedField(field.name, field.shape, field.bit_width,
                              field.bit_offset, field.cardinality,
                              value_offset=lo)
        self._fields[name] = shifted
        return shifted

    def append_mask(self, name: str, values: Any) -> PackedField:
        """Append a boolean/0-1 field using one bit per element."""

        return self.append(name, torch.as_tensor(values, dtype=torch.bool), cardinality=2)

    def read(self, name: str, *, device: torch.device | str | None = None) -> torch.Tensor:
        """Decode a field as a ``torch.long`` tensor."""

        try:
            field = self._fields[name]
        except KeyError as exc:
            raise KeyError(f"unknown route field {name!r}") from exc
        if field.bit_width == 0:
            out = torch.zeros(field.count, dtype=torch.long
                              ).reshape(field.shape)
            if device is not None:
                out = out.to(device)
            if field.value_offset:
                out = out + field.value_offset
            return out
        # Vectorized unpack (ontogenic_v2 Exp-1 item 5), byte-exact vs the
        # retained _read_unsigned loop (fuzz-oracle-tested).
        w = field.bit_width
        lo_byte = field.bit_offset // 8
        hi_byte = (field.bit_offset + field.count * w + 7) // 8
        raw = np.frombuffer(bytes(self._payload[lo_byte:hi_byte]),
                            dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder="little")
        start = field.bit_offset - lo_byte * 8
        bits = bits[start:start + field.count * w].reshape(field.count, w)
        values = (bits.astype(np.uint64)
                  << np.arange(w, dtype=np.uint64)).sum(axis=1)
        out = torch.from_numpy(values.astype(np.int64)).reshape(field.shape)
        if device is not None:
            out = out.to(device)
        if field.value_offset:
            out = out + field.value_offset
        return out

    unpack = read

    def payload(self) -> bytes:
        """Return the header-free packed bits."""

        return bytes(self._payload)

    def to_bytes(self, *, fingerprint: Mapping[str, Any] | None = None) -> bytes:
        """Serialize the field table and payload into a portable binary record.

        ``fingerprint`` (ontogenic_v2, format v2) binds the tape to a model
        configuration (d_model, layers, K, dilations, k, vocab, tau,
        tie-rule, dtype); :meth:`from_bytes` refuses replay against a
        mismatched model when an expected fingerprint is supplied. Records
        using neither fingerprints nor signed fields serialize as version 1,
        byte-identical to the pre-v2 writer.
        """

        if fingerprint is None:
            fingerprint = self._fingerprint
        elif self._fingerprint is not None and dict(fingerprint) != self._fingerprint:
            raise ValueError("PackedRouteTape fingerprint mismatch: refusing rebinding")

        def field_dict(field: PackedField) -> dict[str, Any]:
            d = asdict(field)
            if not field.value_offset:
                d.pop("value_offset", None)
            return d

        needs_v2 = fingerprint is not None or any(
            field.value_offset for field in self.fields)
        header_obj: dict[str, Any] = {
            "version": 2 if needs_v2 else 1,
            "bit_length": self._bit_length,
            "fields": [field_dict(field) for field in self.fields],
        }
        if fingerprint is not None:
            header_obj["fingerprint"] = dict(fingerprint)
        header = json.dumps(header_obj, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
        return _MAGIC + struct.pack("<I", len(header)) + header + bytes(self._payload)

    serialize = to_bytes

    @classmethod
    def from_bytes(cls, blob: bytes | bytearray | memoryview, *,
                   expected_fingerprint: Mapping[str, Any] | None = None,
                   ) -> "PackedRouteTape":
        """Decode and validate a record produced by :meth:`to_bytes`.

        When ``expected_fingerprint`` is supplied the record must carry an
        exactly equal fingerprint (format v2) — a tape can never be silently
        replayed against a mismatched model.
        """

        raw = bytes(blob)
        if len(raw) < 8 or raw[:4] != _MAGIC:
            raise ValueError("not a PackedRouteTape record")
        header_length = struct.unpack("<I", raw[4:8])[0]
        header_end = 8 + header_length
        if header_end > len(raw):
            raise ValueError("truncated PackedRouteTape header")
        try:
            header = json.loads(raw[8:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid PackedRouteTape header") from exc
        if header.get("version") not in (1, 2):
            raise ValueError("unsupported PackedRouteTape version")
        if expected_fingerprint is not None:
            got = header.get("fingerprint")
            if got != dict(expected_fingerprint):
                raise ValueError(
                    "PackedRouteTape fingerprint mismatch: refusing replay "
                    f"(expected {dict(expected_fingerprint)!r}, got {got!r})")

        tape = cls()
        if header.get("fingerprint") is not None:
            if not isinstance(header["fingerprint"], dict):
                raise ValueError("PackedRouteTape fingerprint must be an object")
            tape._fingerprint = deepcopy(header["fingerprint"])
        tape._bit_length = int(header["bit_length"])
        if tape._bit_length < 0:
            raise ValueError("negative packed bit length")
        expected_payload_bytes = (tape._bit_length + 7) // 8
        payload = raw[header_end:]
        if len(payload) != expected_payload_bytes:
            raise ValueError("PackedRouteTape payload length does not match its header")
        tape._payload = bytearray(payload)

        previous_end = 0
        for item in header.get("fields", []):
            field = PackedField(
                name=str(item["name"]),
                shape=tuple(int(size) for size in item["shape"]),
                bit_width=int(item["bit_width"]),
                bit_offset=int(item["bit_offset"]),
                cardinality=(None if item.get("cardinality") is None else int(item["cardinality"])),
                value_offset=int(item.get("value_offset", 0)),
            )
            if not field.name or field.name in tape._fields:
                raise ValueError("invalid or duplicate PackedRouteTape field name")
            if field.bit_width < 0 or field.bit_offset != previous_end:
                raise ValueError("PackedRouteTape fields must be contiguous and ordered")
            if any(size < 0 for size in field.shape):
                raise ValueError("PackedRouteTape field has a negative dimension")
            if field.cardinality is not None:
                required = bits_for_cardinality(field.cardinality)
                if field.bit_width < required:
                    raise ValueError("PackedRouteTape field width is below its coding bound")
            previous_end = field.bit_offset + field.bit_length
            if previous_end > tape._bit_length:
                raise ValueError("PackedRouteTape field extends beyond the payload")
            tape._fields[field.name] = field
        if previous_end != tape._bit_length:
            raise ValueError("PackedRouteTape field table does not cover the payload")

        # Validate cardinality after structural validation (in the STORED
        # domain: read() applies value_offset, so subtract it back).
        for field in tape.fields:
            if field.cardinality is not None:
                decoded = tape.read(field.name)
                if decoded.numel():
                    stored_max = int(decoded.max()) - field.value_offset
                    stored_min = int(decoded.min()) - field.value_offset
                    if stored_max >= field.cardinality or stored_min < 0:
                        raise ValueError(
                            "decoded route lies outside its declared cardinality")
        return tape

    deserialize = from_bytes

    def clone(self) -> "PackedRouteTape":
        return type(self).from_bytes(self.to_bytes())

    def state_dict(self) -> dict[str, bytes]:
        return {"record": self.to_bytes()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PackedRouteTape":
        return cls.from_bytes(state["record"])


def pack_field(values: Any, *, bit_width: int | None = None, cardinality: int | None = None) -> bytes:
    """Pack one anonymous field and return a self-describing record."""

    tape = PackedRouteTape()
    tape.append("field", values, bit_width=bit_width, cardinality=cardinality)
    return tape.to_bytes()


def unpack_field(blob: bytes | bytearray | memoryview) -> torch.Tensor:
    """Inverse of :func:`pack_field`."""

    return PackedRouteTape.from_bytes(blob).read("field")


class RouteTapeRingBuffer:
    """Fixed-slot FIFO ring for serialized route tapes.

    Appending after the buffer reaches ``capacity`` overwrites the oldest tape.
    Records are serialized on insertion, so later mutation of the source tape
    cannot mutate the retained differentiation certificate.  ``slot_bytes`` can
    additionally impose a fixed maximum serialized size per slot.
    """

    def __init__(self, capacity: int, *, slot_bytes: int | None = None) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if slot_bytes is not None and (
            isinstance(slot_bytes, bool) or not isinstance(slot_bytes, int) or slot_bytes <= 0
        ):
            raise ValueError("slot_bytes must be a positive integer when supplied")
        self.capacity = capacity
        self.slot_bytes = slot_bytes
        self._slots: list[bytes | None] = [None] * capacity
        self._start = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _physical_index(self, chronological_index: int) -> int:
        if chronological_index < 0:
            chronological_index += self._size
        if not 0 <= chronological_index < self._size:
            raise IndexError("route tape index out of range")
        return (self._start + chronological_index) % self.capacity

    def append(self, tape: PackedRouteTape | bytes | bytearray | memoryview) -> None:
        record = tape.to_bytes() if isinstance(tape, PackedRouteTape) else bytes(tape)
        # Validate before changing ring state.
        PackedRouteTape.from_bytes(record)
        if self.slot_bytes is not None and len(record) > self.slot_bytes:
            raise ValueError(
                f"serialized tape needs {len(record)} bytes, exceeding slot_bytes={self.slot_bytes}"
            )
        if self._size < self.capacity:
            index = (self._start + self._size) % self.capacity
            self._size += 1
        else:
            index = self._start
            self._start = (self._start + 1) % self.capacity
        self._slots[index] = record

    push = append

    def __getitem__(self, index: int) -> PackedRouteTape:
        physical = self._physical_index(index)
        record = self._slots[physical]
        assert record is not None
        return PackedRouteTape.from_bytes(record)

    def __iter__(self) -> Iterator[PackedRouteTape]:
        for index in range(self._size):
            yield self[index]

    @property
    def oldest(self) -> PackedRouteTape:
        return self[0]

    @property
    def newest(self) -> PackedRouteTape:
        return self[-1]

    @property
    def retained_serialized_bytes(self) -> int:
        return sum(len(self._slots[self._physical_index(i)] or b"") for i in range(self._size))

    @property
    def allocated_bytes(self) -> int | None:
        """Fixed payload allocation when ``slot_bytes`` was specified."""

        return None if self.slot_bytes is None else self.capacity * self.slot_bytes

    def pop_oldest(self) -> PackedRouteTape:
        if not self._size:
            raise IndexError("pop from an empty route tape ring")
        index = self._start
        record = self._slots[index]
        assert record is not None
        self._slots[index] = None
        self._start = (self._start + 1) % self.capacity
        self._size -= 1
        return PackedRouteTape.from_bytes(record)

    def clear(self) -> None:
        self._slots = [None] * self.capacity
        self._start = 0
        self._size = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "slot_bytes": self.slot_bytes,
            "records": [tape.to_bytes() for tape in self],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RouteTapeRingBuffer":
        ring = cls(int(state["capacity"]), slot_bytes=state.get("slot_bytes"))
        records = list(state.get("records", []))
        if len(records) > ring.capacity:
            raise ValueError("ring state contains more records than its capacity")
        for record in records:
            ring.append(record)
        return ring


FixedCapacityRingBuffer = RouteTapeRingBuffer
PackedTapeRingBuffer = RouteTapeRingBuffer


__all__ = [
    "FixedCapacityRingBuffer",
    "PackedField",
    "PackedRouteTape",
    "PackedTapeRingBuffer",
    "RouteTapeRingBuffer",
    "bits_for_cardinality",
    "pack_field",
    "route_code_lower_bound",
    "unpack_field",
]
