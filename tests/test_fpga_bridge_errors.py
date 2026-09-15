"""Exercise real bridge.cpp against a CPU-only, fault-injecting AWS PCI stub."""
import ctypes as C
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "tests/fixtures/pci_error_stub"
BUFFER_BASE = 0x01000000


def configure_library(path):
    lib = C.CDLL(str(path))
    declarations = {
        "tt_fpga_open": ([C.c_char_p], C.c_void_p),
        "tt_fpga_close": ([C.c_void_p], None),
        "tt_fpga_last_error": ([C.c_void_p], C.c_int),
        "tt_fpga_board_present": ([C.c_void_p], C.c_int),
        "tt_fpga_alloc": ([C.c_void_p, C.c_long], C.c_void_p),
        "tt_fpga_free": ([C.c_void_p, C.c_void_p], None),
        "tt_fpga_write_region": ([C.c_void_p, C.c_void_p, C.c_long, C.POINTER(C.c_int32), C.c_long], None),
        "tt_fpga_read_region": ([C.c_void_p, C.c_void_p, C.c_long, C.POINTER(C.c_int32), C.c_long], None),
        "tt_fpga_load_weights": ([C.c_void_p, C.c_void_p, C.POINTER(C.c_int16), C.c_long], None),
        "tt_fpga_save_weights": ([C.c_void_p, C.c_void_p, C.POINTER(C.c_int16), C.c_long], None),
        "tt_fpga_run": ([C.c_void_p, C.c_char_p, C.c_void_p, *([C.c_int] * 5)], C.c_int),
        "tt_fpga_kick": ([C.c_void_p, *([C.c_int] * 4)], None),
        "tt_fpga_done": ([C.c_void_p, C.c_int], C.c_int),
        "stub_reset": ([], None),
        "stub_fault": ([C.c_int, C.c_int, C.c_uint64, C.c_int, C.c_int], None),
        "stub_operations": ([], C.c_int),
        "stub_live_handles": ([], C.c_int),
        "stub_complete": ([C.c_int], None),
        "stub_reset_asserted": ([C.c_int], None),
        "stub_store": ([C.c_int, C.c_uint64, C.c_uint32], None),
        "stub_load": ([C.c_int, C.c_uint64], C.c_uint32),
    }
    for name, (arguments, result) in declarations.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    return lib


@pytest.fixture(scope="module")
def compiled_bridges(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for CPU-only native bridge tests")
    directory = tmp_path_factory.mktemp("pci-bridge")
    libraries = {}
    for backend, define in (("pci", "TT_FPGA_PCI"), ("local", "TT_KERNEL_1LAYER")):
        path = directory / f"bridge_{backend}.so"
        subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O0",
                        f"-D{define}", f"-I{STUB}", str(ROOT / "tikitorch/fpga/bridge.cpp"),
                        str(STUB / "stub.cpp"), "-o", str(path)],
                       check=True, capture_output=True, text=True)
        libraries[backend] = configure_library(path)
    return libraries


@pytest.fixture
def pci(compiled_bridges, monkeypatch):
    monkeypatch.delenv("TT_CLKGEN_BRINGUP", raising=False)
    monkeypatch.setenv("TT_KICK_SPIN", "3")
    lib = compiled_bridges["pci"]
    lib.stub_reset()
    return lib


@pytest.fixture
def context(pci):
    handle = pci.tt_fpga_open(b"0")
    assert handle
    buffer = pci.tt_fpga_alloc(handle, 8)
    assert buffer
    yield pci, handle, buffer
    pci.tt_fpga_free(handle, buffer)
    pci.tt_fpga_close(handle)
    assert pci.stub_live_handles() == 0


def run(lib, handle, buffer):
    return lib.tt_fpga_run(handle, b"forge_train_step", buffer, 0, 0, 0, 0, 0)


def test_pci_successful_transfers_and_run(context):
    lib, handle, buffer = context
    values = (C.c_int32 * 3)(-7, 42, 99)
    lib.tt_fpga_write_region(handle, buffer, 2, values, 3)
    assert list((C.c_int32 * 8).from_address(buffer))[2:5] == list(values)
    lib.stub_store(4, BUFFER_BASE + 12, 17)
    read = (C.c_int32 * 3)()
    lib.tt_fpga_read_region(handle, buffer, 2, read, 3)
    assert list(read) == [-7, 17, 99]
    assert run(lib, handle, buffer) == 0
    assert lib.tt_fpga_last_error(handle) == 0


def test_pci_weight_round_trip_and_zero_length_boundary(context):
    lib, handle, buffer = context
    weights = (C.c_int16 * 3)(-32768, 17, 32767)
    output = (C.c_int16 * 3)()
    lib.tt_fpga_load_weights(handle, buffer, weights, 3)
    lib.tt_fpga_save_weights(handle, buffer, output, 3)
    assert list(output) == list(weights)
    calls = lib.stub_operations()
    lib.tt_fpga_write_region(handle, buffer, 8, None, 0)
    lib.tt_fpga_read_region(handle, buffer, 8, None, 0)
    assert lib.tt_fpga_last_error(handle) == 0
    assert lib.stub_operations() == calls


@pytest.mark.parametrize("operation", ["write", "read"])
def test_failed_transfers_do_not_publish_partial_host_state(context, operation):
    lib, handle, buffer = context
    shadow = (C.c_int32 * 8).from_address(buffer)
    shadow[:] = [123] * 8
    values = (C.c_int32 * 3)(71, 72, 73)
    lib.stub_store(4, BUFFER_BASE, 999)
    lib.stub_fault(1 if operation == "write" else 2, 4, BUFFER_BASE + 4, 1, -77)
    getattr(lib, f"tt_fpga_{operation}_region")(handle, buffer, 0, values, 3)
    assert list(shadow) == [123] * 8
    assert list(values) == [71, 72, 73]
    assert lib.tt_fpga_last_error(handle) == -77
    calls = lib.stub_operations()
    lib.tt_fpga_write_region(handle, buffer, 0, values, 3)
    lib.tt_fpga_read_region(handle, buffer, 0, values, 3)
    lib.tt_fpga_kick(handle, 0, 0, 0, 0)
    assert lib.tt_fpga_done(handle, 0) == 0
    assert run(lib, handle, buffer) == -77
    assert lib.stub_operations() == calls


@pytest.mark.parametrize("address", [0x24, 0x2C, 0x34, 0x3C, 0x00])
def test_register_write_failure_cannot_start_or_succeed(context, address):
    lib, handle, buffer = context
    lib.stub_fault(1, 0, address, 1, 77)
    assert run(lib, handle, buffer) == 77
    assert lib.tt_fpga_last_error(handle) == 77
    assert lib.stub_load(0, 0) == 0


@pytest.mark.parametrize("address", [0x00, 0x10])
def test_poll_and_return_read_failures_are_errors(context, address):
    lib, handle, buffer = context
    lib.stub_fault(2, 0, address, 1, -88)
    assert run(lib, handle, buffer) == -88
    assert lib.tt_fpga_last_error(handle) == -88


def test_nonblocking_helpers_check_failures(context):
    lib, handle, _ = context
    lib.stub_fault(1, 0, 0x2C, 1, -66)
    lib.tt_fpga_kick(handle, 0, 0, 0, 0)
    assert lib.tt_fpga_last_error(handle) == -66
    assert lib.stub_load(0, 0) == 0


def test_nonblocking_poll_failure_is_not_reported_as_done(context):
    lib, handle, _ = context
    lib.stub_fault(2, 0, 0x10000, 1, -66)
    assert lib.tt_fpga_done(handle, 1) == 0
    assert lib.tt_fpga_last_error(handle) == -66


def test_timeout_is_sticky(context):
    lib, handle, buffer = context
    lib.stub_complete(0)
    assert run(lib, handle, buffer) == -200
    assert lib.tt_fpga_last_error(handle) == -200
    calls = lib.stub_operations()
    assert run(lib, handle, buffer) == -200
    assert lib.stub_operations() == calls


@pytest.mark.parametrize("offset,count", [(-1, 1), (0, -1), (8, 1), (7, 2), (2**62, 1)])
def test_invalid_region_never_accesses_pci(context, offset, count):
    lib, handle, buffer = context
    calls = lib.stub_operations()
    lib.tt_fpga_write_region(handle, buffer, offset, (C.c_int32 * 1)(9), count)
    assert lib.tt_fpga_last_error(handle) != 0
    assert lib.stub_operations() == calls


def test_unknown_buffer_is_rejected_without_dereferencing(context):
    lib, handle, _ = context
    calls = lib.stub_operations()
    lib.tt_fpga_read_region(handle, 1, 0, (C.c_int32 * 1)(), 1)
    assert lib.tt_fpga_last_error(handle) == -202
    assert lib.stub_operations() == calls


def test_null_handle_is_safe(pci):
    values = (C.c_int32 * 1)(42)
    pci.tt_fpga_write_region(None, None, 0, values, 1)
    pci.tt_fpga_read_region(None, None, 0, values, 1)
    assert run(pci, None, None) != 0
    assert pci.tt_fpga_last_error(None) != 0
    assert pci.tt_fpga_done(None, 0) == 0
    assert list(values) == [42]


@pytest.mark.parametrize("address", [0x18, 0x1C])
def test_open_rejects_failed_buffer_address_setup(pci, address):
    pci.stub_fault(1, 0, address, 1, -44)
    assert not pci.tt_fpga_open(b"0")
    assert pci.stub_live_handles() == 0


@pytest.mark.parametrize("bar", [0, 4])
def test_open_rejects_attach_errors_and_cleans_up(pci, bar):
    pci.stub_fault(3, bar, 0, 1, -55)
    assert not pci.tt_fpga_open(b"0")
    assert pci.stub_live_handles() == 0


def test_clock_bringup_detach_failure_rejects_open(pci, monkeypatch, capfd):
    monkeypatch.setenv("TT_CLKGEN_BRINGUP", "1")
    pci.stub_fault(4, 14, 0, 1, -55)
    assert not pci.tt_fpga_open(b"0")
    assert "PCI detach failed: -55" in capfd.readouterr().err


def test_close_reports_detach_failure_and_attempts_both_bars(pci, capfd):
    handle = pci.tt_fpga_open(b"0")
    assert handle
    pci.stub_fault(4, 0, 0, 1, -55)
    calls = pci.stub_operations()
    pci.tt_fpga_close(handle)
    assert pci.stub_operations() == calls + 2
    assert "PCI detach failed: -55" in capfd.readouterr().err


@pytest.mark.parametrize("kind,address", [(2, 0x58020), (1, 0x58014), (2, 0x58014)])
def test_clock_bringup_read_or_write_failure_rejects_open(pci, monkeypatch, kind, address):
    monkeypatch.setenv("TT_CLKGEN_BRINGUP", "1")
    pci.stub_fault(kind, 14, address, 1, -44)
    assert not pci.tt_fpga_open(b"0")
    assert pci.stub_live_handles() == 0


def test_clock_bringup_checks_reset_readback(pci, monkeypatch):
    monkeypatch.setenv("TT_CLKGEN_BRINGUP", "1")
    pci.stub_reset_asserted(1)
    assert not pci.tt_fpga_open(b"0")
    assert pci.stub_live_handles() == 0


def test_weight_export_does_not_copy_failed_read(context):
    lib, handle, buffer = context
    weights = (C.c_int16 * 3)(11, 12, 13)
    lib.stub_fault(2, 4, BUFFER_BASE + 4, 1, -99)
    lib.tt_fpga_save_weights(handle, buffer, weights, 3)
    assert list(weights) == [11, 12, 13]
    assert lib.tt_fpga_last_error(handle) == -99


def test_local_bridge_behavior_is_preserved(compiled_bridges):
    lib = compiled_bridges["local"]
    handle = lib.tt_fpga_open(None)
    buffer = lib.tt_fpga_alloc(handle, 8)
    try:
        weights = (C.c_int16 * 2)(-8, 16)
        lib.tt_fpga_load_weights(handle, buffer, weights, 2)
        output = (C.c_int16 * 2)()
        lib.tt_fpga_save_weights(handle, buffer, output, 2)
        assert list(output) == [-8, 16]
        assert lib.tt_fpga_run(handle, b"forge_train_step", buffer, 1, 2, 3, 4, 0) == 0
        assert (C.c_int32 * 8).from_address(buffer)[0] == 10
        assert lib.tt_fpga_last_error(handle) == 0
    finally:
        lib.tt_fpga_free(handle, buffer)
        lib.tt_fpga_close(handle)
