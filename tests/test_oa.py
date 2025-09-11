import jax
import pytest

from orray.oa import OrthogonalArray

from .helpers import (
    check_is_orthogonal,
    check_device_placement,
    check_exceptions,
    check_jit_compatible,
    check_return_type,
	check_getitem,
	check_randomise
)


def test_randomise(oa: OrthogonalArray):
	check_randomise(oa)

def test_is_orthogonal(oa: OrthogonalArray):
	rng = jax.random.PRNGKey(0)
	check_is_orthogonal(rng, oa)


def test_return_type(oa: OrthogonalArray):
    check_return_type(oa, batch_size=2)
    check_return_type(oa, batch_size=2, jit_compatible=True)


def test_jit_compatible(oa: OrthogonalArray):
	batch_size = min(len(oa), 128)
	check_jit_compatible(oa, batch_size=batch_size)


def test_device_placement(oa: OrthogonalArray):
    dev = jax.devices()[0]
    check_device_placement(oa, batch_size=2, device=dev)



def test_batches_invalid_sizes(oa: OrthogonalArray):
    # generic invalids
    check_exceptions(oa)
    # too large also errors
    with pytest.raises(ValueError):
        _ = oa.batches(oa.num_rows + 1)

def test_getitem(oa: OrthogonalArray):
	check_getitem(oa)

def test_batches_length_and_last_batch_shape(oa: OrthogonalArray):
    batch_size = 3
    seq = oa.batches(batch_size)
    expected_len = (oa.num_rows + batch_size - 1) // batch_size
    assert len(seq) == expected_len
    # check last batch shape truncation in non-jit mode
    last = seq[len(seq) - 1]
    expected_last = oa.num_rows % batch_size or batch_size
    assert last.shape == (expected_last, oa.num_cols)

    # jit mode should always be full batch_size with a mask
    jseq = oa.batches(batch_size, jit_compatible=True)
    jb, jm = jseq[len(jseq) - 1]
    assert jb.shape == (batch_size, oa.num_cols)
    assert jm.shape == (batch_size,)
