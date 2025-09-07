import pytest
import jax
import jax.numpy as jnp

from orray.oa import MaterializedOrthogonalArray
from .helpers import (
	check_jit_compatible,
	check_device_placement,
	check_return_type,
	check_exceptions,
)


@pytest.fixture
def small_oa() -> MaterializedOrthogonalArray:
	# 4x3 array with int8 dtype
	arr = (jnp.arange(12, dtype=jnp.int8) % 3).reshape(4, 3)
	return MaterializedOrthogonalArray(num_levels=3, strength=1, orthogonal_array=arr)


def test_return_types_small_oa(small_oa: MaterializedOrthogonalArray):
	check_return_type(small_oa, batch_size=2)
	check_return_type(small_oa, batch_size=2, jit_compatible=True)


def test_jit_compat_small_oa(small_oa: MaterializedOrthogonalArray):
	check_jit_compatible(small_oa, batch_size=2)


def test_device_placement_small_oa(small_oa: MaterializedOrthogonalArray):
	dev = jax.devices()[0]
	check_device_placement(small_oa, batch_size=2, device=dev)


def test_batches_invalid_sizes(small_oa: MaterializedOrthogonalArray):
	# generic invalids
	check_exceptions(small_oa)
	# too large also errors
	with pytest.raises(ValueError):
		_ = small_oa.batches(small_oa.num_rows + 1)


def test_batches_length_and_last_batch_shape(small_oa: MaterializedOrthogonalArray):
	bs = 3
	seq = small_oa.batches(bs)
	expected_len = (small_oa.num_rows + bs - 1) // bs
	assert len(seq) == expected_len
	# check last batch shape truncation in non-jit mode
	last = seq[len(seq) - 1]
	expected_last = small_oa.num_rows % bs or bs
	assert last.shape == (expected_last, small_oa.num_cols)

	# jit mode should always be full batch_size with a mask
	jseq = small_oa.batches(bs, jit_compatible=True)
	jb, jm = jseq[len(jseq) - 1]
	assert jb.shape == (bs, small_oa.num_cols)
	assert jm.shape == (bs,)
