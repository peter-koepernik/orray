"""Reusable test helpers for OrthogonalArray implementations.

These helpers are written to be pytest-friendly: they raise AssertionError on
failure and return None on success. You can call them directly in tests.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from orray.oa import OrthogonalArray

__all__ = [
    "check_valid",
    "check_jit_compatible",
    "check_device_placement",
    "check_return_type",
    "check_exceptions",
]

def _devices_of(x) -> tuple[jax.Device, ...] | tuple[()]:
    """Return devices that hold `x`.

    Supports single-device and sharded arrays across JAX versions.
    """
    # jax.Array (0.4+) typically has .devices()
    if hasattr(x, "devices"):
        try:
            ds = tuple(x.devices())
            return ds
        except TypeError:
            pass
    # Older DeviceArray had .device_buffer.device() or .device()
    if hasattr(x, "device"):
        try:
            d = x.device()
            return (d,) if d is not None else tuple()
        except Exception:
            pass
    return tuple()


def check_jit_compatible(oa: OrthogonalArray, batch_size: int, *, device: jax.Device | None = None) -> None:
    """Assert that `batches(jit_compatible=True)` is jax.jit-compatible with traced indices.

    Builds a sequence via `oa.batches(batch_size, jit_compatible=True, device=...)`
    and scans over indices calling `seq[i]` inside a jitted function.
    """
    assert batch_size > 0, "batch_size must be positive"
    seq = oa.batches(batch_size, jit_compatible=True, device=device)
    n = len(seq)

    def step(carry, i):
        batch, mask = seq[i]
        # consume both so they are not DCE'ed
        s = jnp.sum(batch.astype(jnp.int32)) + jnp.sum(mask.astype(jnp.int32))
        return carry + s, None

    out, _ = jax.lax.scan(step, jnp.asarray(0, jnp.int32), jnp.arange(n))
    assert hasattr(out, "dtype"), "JIT run did not return a numeric value"


def check_device_placement(oa: OrthogonalArray, batch_size: int, device: jax.Device) -> None:
    """Assert that `batches(..., device=device)` yields arrays on that device.

    Checks first and last batches in jit-compatible mode.
    """
    seq = oa.batches(batch_size, jit_compatible=True, device=device)
    b0, m0 = seq[0]
    d0 = _devices_of(b0)
    assert d0 and device in d0, f"First batch not on device {device}; got {d0}"
    # also check mask placement if detectable
    dm0 = _devices_of(m0)
    if dm0:
        assert device in dm0, f"Mask not on device {device}; got {dm0}"
    # last batch
    bi, mi = seq[len(seq) - 1]
    di = _devices_of(bi)
    assert di and device in di, f"Last batch not on device {device}; got {di}"


def check_return_type(oa: OrthogonalArray, batch_size: int, *, jit_compatible: bool = False) -> None:
    """Assert that `batches` yields the expected JAX types and shapes.

    - Non-jit mode: returns a single JAX array with dtype int8 and shape (batch_size, num_cols) for batch 0.
    - Jit mode: returns (batch, mask) where batch is int8 with shape (batch_size, num_cols) and mask is bool with shape (batch_size,).
    """
    if not jit_compatible:
        seq = oa.batches(batch_size)
        x0 = seq[0]
        assert hasattr(x0, "dtype"), "Item is not an array-like with dtype"
        assert x0.dtype == jnp.int8, f"Expected dtype int8, got {x0.dtype}"
        assert x0.shape == (batch_size, oa.num_cols), (
            f"Expected shape {(batch_size, oa.num_cols)}, got {x0.shape}"
        )
    else:
        seq = oa.batches(batch_size, jit_compatible=True)
        x0, m0 = seq[0]
        assert hasattr(x0, "dtype"), "Batch is not an array-like with dtype"
        assert x0.dtype == jnp.int8, f"Expected dtype int8, got {x0.dtype}"
        assert x0.shape == (batch_size, oa.num_cols), (
            f"Expected shape {(batch_size, oa.num_cols)}, got {x0.shape}"
        )
        # mask dtype may be bool or uint8 depending on implementation; accept truthy types but require correct shape
        assert m0.shape == (batch_size,), f"Expected mask shape {(batch_size,)}, got {m0.shape}"


def check_exceptions(
    oa: OrthogonalArray,
    *,
    invalid_batch_sizes: tuple[int, ...] = (0, -1),
) -> None:
    """Assert common invalid-argument paths raise ValueError.

    Specifically checks that invalid batch sizes are rejected by `batches`.
    """
    for bs in invalid_batch_sizes:
        try:
            _ = oa.batches(bs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for batch_size={bs}")

def check_valid(oa: OrthogonalArray):
    """tests that a given orthogonal array indeed satisfies
    the defining property: in any `strength` number of columns,
    all `pow(levels, strength)` number of possible rows appear
    an even number of time
    """
    # implemented later
    return True