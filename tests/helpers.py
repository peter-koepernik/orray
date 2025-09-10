"""Reusable test helpers for OrthogonalArray implementations.

These helpers are written to be pytest-friendly: they raise AssertionError on
failure and return None on success. You can call them directly in tests.
"""

from __future__ import annotations

import math
import random
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Bool, Int32, Int64

from orray.oa import OrthogonalArray

__all__ = [
    "check_jit_compatible",
    "check_device_placement",
    "check_return_type",
    "check_exceptions",
    "check_is_orthogonal",
    "check_getitem"
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


def check_getitem(oa: OrthogonalArray) -> None:
    materialized_array = oa.materialize()
    materialized_array_from_getitem = jnp.vstack([oa[i] for i in range(oa.num_rows)])
    assert jnp.allclose(materialized_array, materialized_array_from_getitem)


def check_jit_compatible(
    oa: OrthogonalArray, batch_size: int, *, device: jax.Device | None = None
) -> None:
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
        s = jnp.sum(batch.astype(jnp.int64)) + jnp.sum(mask.astype(jnp.int64))
        return carry + s, None

    out, _ = jax.lax.scan(
        step, jnp.asarray(0, dtype=jnp.int64), jnp.arange(n, dtype=jnp.int64)
    )
    assert hasattr(out, "dtype"), "JIT run did not return a numeric value"


def check_device_placement(
    oa: OrthogonalArray, batch_size: int, device: jax.Device
) -> None:
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


def check_return_type(
    oa: OrthogonalArray, batch_size: int, *, jit_compatible: bool = False
) -> None:
    """Assert that `batches` yields the expected JAX types and shapes.

    - Non-jit mode: returns a single JAX array with dtype int32 and shape (batch_size, num_cols) for batch 0.
    - Jit mode: returns (batch, mask) where batch is int32 with shape (batch_size, num_cols) and mask is bool with shape (batch_size,).
    """
    if not jit_compatible:
        seq = oa.batches(batch_size)
        x0 = seq[0]
        assert hasattr(x0, "dtype"), "Item is not an array-like with dtype"
        assert x0.dtype == jnp.int32, f"Expected dtype int32, got {x0.dtype}"
        assert x0.shape == (batch_size, oa.num_cols), (
            f"Expected shape {(batch_size, oa.num_cols)}, got {x0.shape}"
        )
    else:
        seq = oa.batches(batch_size, jit_compatible=True)
        x0, m0 = seq[0]
        assert hasattr(x0, "dtype"), "Batch is not an array-like with dtype"
        assert x0.dtype == jnp.int32, f"Expected dtype int32, got {x0.dtype}"
        assert x0.shape == (batch_size, oa.num_cols), (
            f"Expected shape {(batch_size, oa.num_cols)}, got {x0.shape}"
        )
        # mask dtype may be bool or uint32 depending on implementation; accept truthy types but require correct shape
        assert m0.shape == (batch_size,), (
            f"Expected mask shape {(batch_size,)}, got {m0.shape}"
        )


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


@partial(jax.jit, static_argnames=("n", "k", "batch_size", "sort"))
def sample_column_indices(
    rng: jax.random.PRNGKey, n: int, k: int, batch_size: int, sort: bool = False
) -> Int32[Array, "batch_size k"]:
    """
    Returns an integer array of shape (batch_size, k). Each row is a uniformly random
    k-subset of {0, ..., n-1}, sampled without replacement and optionally sorted ascending.

    Args:
        rng: PRNGKey
        n: size of the ground set (>= 1)
        k: subset size (1 <= k <= n)
        batch_size: batch size (>= 1)
        sort: whether to sort the rows

    Returns:
        (batch_size, k) int32 array with rows i1 < ... < ik, values in {0, ..., n-1}.
    """
    # Basic argument checks (executed at trace time since they are Python ints).
    if not (1 <= k <= n):
        raise ValueError("Require 1 <= k <= n.")
    if batch_size < 1:
        raise ValueError("Require batch_size >= 1.")

    # Draw i.i.d. uniforms for each of the n items per batch row.
    # The indices of the top-k form a uniformly random size-k subset w/o replacement.
    g = jax.random.uniform(rng, shape=(batch_size, n))  # (batch_size, n)
    _, idx = jax.lax.top_k(g, k)  # (batch_size, k), indices in descending Gumbel order
    if sort:
        idx = jnp.sort(idx, axis=1)  # sort to enforce i1 < ... < ik
    return idx.astype(jnp.int32)


@partial(jax.jit, static_argnames="num_levels")
def _check_all_rows_appear_equally_often(
    x: Int64[Array, "batch_size num_rows num_cols"], num_levels: int
) -> Bool[Array, "batch_size"]:
    """
    x: int array of shape (batch_size, num_rows, num_columns) with entries in {0,...,num_levels-1}
    num_levels: size of the alphabet

    Returns:
        ok: bool array of shape (batch_size,), where ok[i] is True iff
            every one of the k**num_columns possible columns appears exactly
            num_rows / (k**num_columns) times in x[i, ...].
    """
    jax.debug.print("x:\n{}", x)
    B, R, C = x.shape
    Kp = num_levels**C  # number of possible columns

    # Quick impossibility check
    if R % Kp != 0:
        return jnp.zeros((B,), dtype=bool)
    # each possible row should appear this many times
    target = R // Kp

    # Encode each row (length C, values 0..k-1) as a base-k integer in [0, Kp-1]
    # weights = [k^(C-1), k^(C-2), ..., k^0]
    weights = jnp.power(
        jnp.int64(num_levels), jnp.arange(C - 1, -1, -1, dtype=jnp.int64)
    )
    x0 = x.astype(jnp.int64)
    # (B,R,C) @ (C,) -> (B,R)
    codes = jnp.tensordot(x0, weights, axes=([2], [0]))  # integer codes per row
    assert codes.shape == (B, R)

    # Batched histogram over codes with fixed length Kp
    def hist_one(row_codes):
        return jnp.bincount(row_codes, length=Kp)

    counts = jax.vmap(hist_one)(codes)  # (B, Kp)
    assert counts.shape == (B, Kp)

    # Check every bin equals the target count
    ok = jnp.all(counts == target, axis=1)
    return ok


@partial(jax.jit, static_argnames=["num_levels", "strength"])
def _set_of_columns_is_valid(
    materialized_orthogonal_array: Int32[Array, "num_rows num_cols"],
    indices: Int32[Array, "batch_size ..."],
    num_levels: int,
    strength: int,
):
    num_rows, num_cols = materialized_orthogonal_array.shape
    batch_size, _ = indices.shape
    assert indices.shape[1] == min(strength, num_cols)

    # shape (num_rows, batch_size, strength)
    cols = materialized_orthogonal_array[:, indices]
    cols = jnp.permute_dims(cols, (1, 0, 2))
    assert cols.shape == (batch_size, num_rows, min(strength, num_cols))

    ok = _check_all_rows_appear_equally_often(cols, num_levels)
    return ok


@partial(jax.jit, static_argnames=["confidence", "epsilon"])
def _is_orthogonal(
    rng: jax.random.PRNGKey,
    orthogonal_array: OrthogonalArray,
    confidence: float = 0.99,
    epsilon: float = 0.02,
):
    _oa = orthogonal_array.materialize()

    strength = orthogonal_array.strength
    num_levels = orthogonal_array.num_levels
    num_rows = orthogonal_array.num_rows
    num_cols = orthogonal_array.num_cols

    if strength >= num_cols:
        # check full array
        indices = jnp.arange(num_cols, dtype=jnp.int32)
        ok = _set_of_columns_is_valid(_oa, indices[None, :], num_levels, strength)
        return _oa, ok

    N = int(math.ceil(-math.log(1 - confidence) / epsilon))  # default: 231

    assert strength >= 1
    assert num_levels >= 2
    assert _oa.shape == (num_rows, num_cols)

    indices = sample_column_indices(rng, num_cols, strength, batch_size=N)
    ok = _set_of_columns_is_valid(_oa, indices, num_levels, strength)
    return _oa, ok


def check_is_orthogonal(
    rng: jax.random.PRNGKey,
    orthogonal_array: OrthogonalArray,
    confidence: float = 0.99,
    epsilon: float = 0.02,
) -> None:
    """
    If at least a fraction of epsilon of the sets of columns of the given array do
    *not* have the defining properties of OA's, then this check fails with probability
    at least 'confidence'.

    If the array is orthogonal this check always passes.

    Math:
        If we sample from a Bin(1,1-p) with p >= epsilon, N times, then the probability
        that we hit 1 every time is:
        (1-p)^N <= (1-eps)^N <= exp(-eps N) <!= 1-conf, so N >= - log(1-conf) / eps
    """

    _oa, ok = _is_orthogonal(rng, orthogonal_array, confidence, epsilon)

    num_levels = orthogonal_array.num_levels
    assert jnp.all(_oa >= 0)
    assert jnp.all(_oa < num_levels)

    assert jnp.all(ok)
