import pytest
import torch

from tikitorch.autodiff import DeviceRouteTape, PackedRouteTape
from tikitorch.autodiff.device_tape import pack_tensor, unpack_tensor


@pytest.mark.parametrize("width", [0, 1, 3, 5, 8, 9, 16, 31, 63])
@pytest.mark.parametrize("count", [0, 1, 7, 32])
def test_byte_exact_with_portable_format(width, count):
    vals = torch.arange(count) % max(1, min(1 << width, 37))
    if count and width == 63:
        vals[-1] = (1 << 63) - 1
    tape = DeviceRouteTape()
    tape.append("x", vals, bit_width=width)
    portable = PackedRouteTape()
    portable.append("x", vals, bit_width=width)
    assert torch.equal(tape.read("x"), vals)
    assert tape._storage["x"].numpy().tobytes() == portable.payload()
    assert tape.to_bytes() == portable.to_bytes()


@pytest.mark.parametrize("packed", [True, False])
def test_owned_storage_signed_fields_and_fingerprint_lifecycle(packed):
    binding = {"dtype": "float32", "shape": [2, 3]}
    tape = DeviceRouteTape(packed=packed, fingerprint=binding)
    values = torch.tensor([-8, 0, 8])
    tape.append_signed("shift", values, lo=-8, hi=8)
    tape.append_mask("gate", torch.tensor([True, False]))
    values.zero_()
    assert tape.read("shift").tolist() == [-8, 0, 8]
    original = tape.to_bytes()
    variants = [tape.clone(), DeviceRouteTape.from_state_dict(tape.state_dict()),
                PackedRouteTape.from_bytes(original).clone(),
                PackedRouteTape.from_state_dict(PackedRouteTape.from_bytes(original).state_dict())]
    for other in variants:
        assert other.fingerprint == binding
        assert other.to_bytes() == original
        other.fingerprint["shape"][0] = 99
        assert other.fingerprint == binding
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            other.to_bytes(fingerprint={"bad": True})


def test_device_tape_validation():
    for values, width in [(torch.tensor([-1]), 3), (torch.tensor([8]), 3)]:
        with pytest.raises(ValueError):
            DeviceRouteTape().append("x", values, bit_width=width)
    with pytest.raises(TypeError):
        DeviceRouteTape().append("x", torch.tensor([1.2]), cardinality=3)
    with pytest.raises(ValueError):
        DeviceRouteTape().append("x", torch.tensor([3]), cardinality=3)
    with pytest.raises(ValueError):
        DeviceRouteTape().append_signed("x", torch.tensor([50], dtype=torch.uint8), lo=100, hi=600)
    tape = DeviceRouteTape()
    tape.append_signed("x", torch.tensor([0], dtype=torch.uint8), lo=-256, hi=0)
    assert tape.read("x").item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_packing_stays_on_device():
    values = torch.arange(1000, device="cuda") % 17
    tape = DeviceRouteTape(device="cuda")
    tape.append("x", values, cardinality=17)
    assert tape._storage["x"].is_cuda and tape.read("x").is_cuda
    assert torch.equal(tape.read("x"), values)
    assert torch.equal(tape.to("cpu").read("x"), values.cpu())


def test_compiled_pack_unpack_fullgraph():
    # AOT eager validates graph capture without depending on a C++/CUDA toolchain.
    encode = torch.compile(pack_tensor, backend="aot_eager", fullgraph=True)
    decode = torch.compile(unpack_tensor, backend="aot_eager", fullgraph=True)
    values = torch.arange(49) % 17
    assert torch.equal(decode(encode(values, 5), 5, 49), values)
