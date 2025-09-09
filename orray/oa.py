import abc
import math
import jax
import jax.numpy as jnp
import equinox as eqx
import operator
from functools import partial, reduce
from typing import Sequence, overload, Literal, Callable, Collection
from jaxtyping import Int8, Bool, Array, Int, UInt8


class OrthogonalArray(eqx.Module, abc.ABC, Sequence[Int8[Array, " num_cols"]]):
    """
    An abstract base class for all orthogonal array implementations.

    This class defines the public API and shared functionality, while delegating
    the core data generation logic to its subclasses.
    """

    num_rows: int = eqx.field(static=True)
    num_cols: int = eqx.field(static=True)
    num_levels: int = eqx.field(static=True)
    strength: int = eqx.field(static=True)

    def __check_init__(self):
        """Initializes the static properties of the array."""
        if self.num_rows <= 0 or self.num_cols <= 0:
            raise ValueError("Dimensions must be positive.")
        if self.num_levels < 2:
            raise ValueError("Number of levels must be at least 2.")
        if self.strength < 1:
            raise ValueError("Strength must be at least 1.")

    @property
    def runs(self) -> int:
        """Alias for self.num_rows."""
        return self.num_rows

    @property
    def factors(self) -> int:
        """Alias for self.num_cols."""
        return self.num_cols

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_cols)

    def __len__(self):
        return self.num_rows

    def __getitem__(self, i) -> Int8[Array, " num_cols"]:
        return self._get_batch(batch_idx=i, batch_size=1)[0, ...]

    @abc.abstractmethod
    def _get_batch(
        self, batch_idx: Int8[Array, ""], batch_size: int, device: jax.Device | None = None
    ) -> Int8[Array, "batch_size num_cols"]:
        """Get a batch of rows from the orthogonal array.

        Args:
            batch_idx: The index of the batch to retrieve. Guaranteed to be in range [0, num_batches).
            batch_size: The number of rows to return. Guaranteed to be in [1, num_rows].
            device: Optional JAX device where the batch should be created. If provided,
                the batch must be computed directly on this device to minimize data transfer.

        Returns:
            A JAX int8 array with shape (batch_size, num_cols) containing the requested
            batch of orthogonal array rows.

        Notes:
            - For batch_idx < num_batches - 1: Returns the corresponding rows of the
              orthogonal array
            - For batch_idx == num_batches - 1: Returns the remaining rows. If fewer
              than batch_size rows remain, pads to shape (batch_size, num_cols) with
              arbitrary values.
            - When device is specified, the batch must be created directly on that
              device rather than computed elsewhere and transferred.
            - Must be `jax.jit` compatible (with static `batch_size` and `device`, and traced `batch_idx`)
        """
        raise NotImplementedError

    def materialize(
        self, device: jax.Device | None = None
    ) -> Int8[Array, "num_rows num_cols"]:
        """Materializes the entire orthogonal array into a single jax array.
        Only use for small arrays!

        Args:
            device: Optional target device on which to create/return the array.

        Raises:
            MemoryError: If the orthogonal array is too large to fit into memory.
        """
        try:
            full_array = jnp.empty(self.shape, dtype=jnp.int8, device=device)
        except MemoryError as e:
            num_bytes_per_entry = 1  # int8 = 1 byte
            total_num_bytes = self.num_rows * self.num_cols * num_bytes_per_entry
            total_num_mib = total_num_bytes / (1024**2)
            raise MemoryError(
                f"Failed to allocate memory for orthogonal array with shape {self.shape}, which would require {total_num_mib:.2f} MiB:\n{e}"
            )

        batch_size = min(self.num_rows, 8192)
        num_batches = math.ceil(self.num_rows / batch_size)

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, self.num_rows)
            generated_batch = self._get_batch(i, batch_size, device=device)
            full_array = full_array.at[start:end, :].set(generated_batch[: end - start, :])
        return full_array

    @overload
    def batches(
        self,
        batch_size: int,
        jit_compatible: Literal[False] = ...,
        *,
        device: jax.Device | None = ...,
    ) -> Sequence[Int8[Array, "batch_size num_cols"]]: ...

    @overload
    def batches(
        self,
        batch_size: int,
        jit_compatible: Literal[True],
        *,
        device: jax.Device | None = ...,
    ) -> Sequence[
        tuple[Int8[Array, "batch_size num_cols"], Bool[Array, "batch_size"]]
    ]: ...

    def batches(
        self,
        batch_size: int,
        jit_compatible: bool = False,
        *,
        device: jax.Device | None = None,
    ):
        """Returns a Sequence over batches of rows (runs) of the orthogonal arrays.

        If `jit_compatible` is False (default), each item is a batch array of shape
        (<= batch_size, num_cols); the last batch is truncated to the remaining rows.

        If `jit_compatible` is True, each item is a tuple (batch, mask) where `batch`
        has shape (batch_size, num_cols) and `mask` has shape (batch_size,) marking
        which rows are valid (always all True except potentially on the last batch);
        this enables JIT-friendly static shapes.

        Args:
            batch_size: Number of rows per batch (must be marked static in a JIT context!)
                jit_compatible: Whether to return (batch, mask) with static shapes.
                Must be in [1, num_rows]
            device: Optional target device on which to generate batches.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive but is {batch_size}.")
        if batch_size > self.num_rows:
            raise ValueError(
                "batch_size must be in [1, num_rows] = [1, {self.num_rows}] but is {batch_size}."
            )

        class _BatchSequence:
            __slots__ = ("_parent", "_batch_size", "_num_batches", "_jit", "_device")

            def __init__(
                self,
                p: "OrthogonalArray",
                bs: int,
                jit_flag: bool,
                dev: jax.Device | None,
            ):
                self._parent = p
                self._batch_size = bs
                self._num_batches = math.ceil(p.num_rows / bs)
                self._jit = jit_flag
                self._device = dev

            def __len__(self):
                return self._num_batches

            def __getitem__(self, i: int):
                if self._jit:
                    i = jnp.mod(i, self._batch_size)
                    valid_count = self._parent.num_rows - i * self._batch_size
                    mask = jnp.arange(self._batch_size) < valid_count
                    batch = self._parent._get_batch(
                        i, self._batch_size, device=self._device
                    )
                    return batch, mask
                else:
                    if i < -self._num_batches or i >= self._num_batches:
                        raise IndexError(
                            f"Index {i} out of bounds [{-self._num_batches},{self._num_batches})"
                        )
                    i = i % self._num_batches
                    batch = self._parent._get_batch(
                        i, self._batch_size, device=self._device
                    )
                    if i == len(self) - 1:
                        last_size = self._parent.num_rows % self._batch_size
                        if last_size > 0:
                            return batch[:last_size]
                    return batch

        return _BatchSequence(self, batch_size, jit_compatible, device)


class MaterializedOrthogonalArray(OrthogonalArray):
    """An OrthogonalArray class that just stores the full
    orthogonal array in memory. Only works for small arrays.
    """

    _oa: Int8[Array, "num_rows num_cols"]

    def __init__(
        self,
        num_levels: int,
        strength: int,
        orthogonal_array: Int8[Array, "num_rows num_cols"],
    ):
        self.num_rows, self.num_cols = orthogonal_array.shape
        self.num_levels = num_levels
        self.strength = strength

        # Ensure orthogonal_array has jnp.int8 type
        orthogonal_array = jnp.asarray(orthogonal_array, dtype=jnp.int8)

        # this ensures that the final batch is correctly padded
        # regardless of the batch_size (which is <= num_rows)
        self._oa = jnp.concatenate(
            [orthogonal_array, jnp.zeros_like(orthogonal_array)], axis=0
        )

    def _get_batch(
        self, batch_idx: Int8[Array, ""], batch_size: int, device: jax.Device | None = None
    ) -> Int8[Array, "batch_size num_cols"]:
        start = batch_idx * batch_size

        # don't truncate at num_rows since self._oa is already padded
        # result = self._oa[start:start + batch_size, :]
        result = jax.lax.dynamic_slice_in_dim(self._oa, start, batch_size, axis=0)

        # Ensure result is on the specified device
        if device is not None:
            result = jax.device_put(result, device)

        return result


class LinearOrthogonalArray(OrthogonalArray):
    """
    Constructs orthogonal arrays by taking all possible linear combinations of
    the rows of a `generator_matrix` modulo `mod`, where the arities of the linear
    combinations of the rows are passed in `arities`. Each batch of rows obtained in
    this way is then passed to the `post_linear_combination_processor`, if given.
    Currently the only constructions that use this postprocessing are the kerdock
    and DG constructions, where it's the gray map.

    If `binary_oa_even_to_odd_strength` is True, then every row is duplicated, once
    with an additional 0, and once inverted and with an additional 1

    """

    _generator_matrix: Int8[Array, "k num_cols"]
    _arities: tuple[tuple[int, int], ...]
    _mod: int
    _even_to_odd: bool
    _post_linear_combination_processor: (
        Callable[[Int8[Array, "num_rows num_cols1"]], Int8[Array, "num_rows num_cols2"]]
        | None
    )

    def __init__(
        self,
        generator_matrix: Int8[Array, "k num_cols"],
        arities: Collection[tuple[int, int]],
        mod: int,
        num_levels: int,
        strength: int,  # for `even_to_odd`, this is the strength of the final construction, so odd
        binary_oa_even_to_odd_strength: bool = False,
        post_linear_combination_processor: Callable[
            [Int8[Array, "num_rows num_cols1"]], Int8[Array, "num_rows num_cols2"]
        ]
        | None = None,
    ):
        self.num_levels = num_levels
        self.strength = strength

        self._generator_matrix = generator_matrix
        self._arities = tuple(tuple(a) for a in arities)  # make hashable
        self._mod = mod
        self._even_to_odd = binary_oa_even_to_odd_strength
        if self._even_to_odd and num_levels != 2:
            raise ValueError(
                "`even_to_odd` only allowed for binary OA's, but num_levels = {num_levels} != 2."
            )
        if self._even_to_odd and strength % 2 == 0:
            raise ValueError(
                "`even_to_odd` is set and implies an odd strength, but `strength` is given as {strength}."
            )
        self._post_linear_combination_processor = post_linear_combination_processor

        # final shape of OA = (num_rows, num_cols)
        self.num_rows = 1
        for n_cols, q in arities:
            self.num_rows *= q**n_cols
        if self._even_to_odd:
            self.num_rows *= 2

        if self._post_linear_combination_processor is None:
            self.num_cols = self._generator_matrix.shape[1]
        else:
            # infer how many rows it will turn into
            _, self.num_cols = jax.eval_shape(
                self._post_linear_combination_processor, self._generator_matrix
            ).shape

        if self._even_to_odd:
            self.num_cols += 1

    def _get_batch_even_batch_size(
        self, batch_idx: Int8[Array, ""], batch_size: int, device: jax.Device | None = None
    ) -> Int8[Array, "batch_size num_cols"]:
        if device is not None:
            _generator_matrix = jax.device_put(self._generator_matrix, device)

        batch_start_row = batch_idx * batch_size

        valid = batch_start_row + jnp.arange(batch_size, device=device) < self.rows

        if self._even_to_odd:
            # batch_size = int(batch_end_row // 2 - batch_start_row // 2)
            _batch_size = batch_size // 2
            unmodified_rows = get_row_batch_of_trivial_mixed_level_oa(
                i0=batch_start_row // 2,
                arities=self._arities,
                batch_size=_batch_size,
                device=device,
            )
            rows = jnp.repeat(unmodified_rows, repeats=2, axis=0)
        else:
            rows = get_row_batch_of_trivial_mixed_level_oa(
                i0=batch_start_row,
                arities=self._arities,
                batch_size=batch_size,
                device=device,
            )
        OA = jnp.mod(rows @ _generator_matrix, self._mod)
        if self._post_linear_combination_processor is not None:
            OA = self._post_linear_combination_processor(OA)

        if self._even_to_odd:
            OA = jnp.concatenate(
                (OA, jnp.zeros((batch_size, 1), dtype=jnp.int8)), axis=1
            )
            OA = OA.at[1::2, :].set(1 - OA[1::2, :])

        return OA, valid

    def _get_batch(
        self, batch_idx: Int8[Array, ""], batch_size: int, device: jax.Device | None = None
    ) -> Int8[Array, "batch_size num_cols"]:
        if device is not None:
            _generator_matrix = jax.device_put(self._generator_matrix, device)

        batch_start_row = batch_idx * batch_size

        if self._even_to_odd:
            _batch_size = batch_size // 2 + 1
            unmodified_rows = get_row_batch_of_trivial_mixed_level_oa(
                i0=batch_start_row // 2,
                arities=self._arities,
                batch_size=_batch_size,
                device=device,
            )
            rows = jnp.repeat(unmodified_rows, repeats=2, axis=0)
            rows = jax.lax.dynamic_slice_in_dim(
                rows, batch_start_row % 2, batch_size, axis=0
            )
        else:
            rows = get_row_batch_of_trivial_mixed_level_oa(
                i0=batch_start_row,
                arities=self._arities,
                batch_size=batch_size,
                device=device,
            )
        OA = jnp.mod(rows @ self._generator_matrix, self._mod)
        if self._post_linear_combination_processor is not None:
            OA = self._post_linear_combination_processor(OA)

        if self._even_to_odd:
            OA = jnp.concatenate(
                (OA, jnp.zeros((batch_size, 1), dtype=jnp.int8, device=device)), axis=1
            )
            idx = jnp.arange(batch_size, device=device)
            flip_mask = jnp.bitwise_and(idx + batch_start_row, 1).astype(bool)
            OA = jnp.where(flip_mask[:, None], 1 - OA, OA)
        return OA


# @partial(jax.jit, static_argnames=("arities", "batch_size"))
def get_row_batch_of_trivial_mixed_level_oa(
    i0: Int8[Array, ""],
    arities: tuple[tuple[int, int], ...],
    batch_size: int,
    device: jax.Device | None = None,
) -> Int8[Array, "batch_size num_cols"]:
    n_cols = sum(n for (n, _) in arities)
    n_rows = reduce(operator.mul, [pow(q, n) for (n, q) in arities])
    result = jnp.zeros((batch_size, n_cols), dtype=jnp.int8, device=device)

    indices = i0 + jnp.arange(batch_size, device=device)

    j = jnp.asarray(0)  # col index
    period = n_rows
    for n, q in arities:
        if q == 2:
            ints = i0 + jnp.arange(batch_size, device=device)
            bits = jnp.arange(n, device=device)
            _update = jnp.bitwise_and(jnp.right_shift(ints[:, None], bits[None, :]), 1)
            update = _update.astype(jnp.int8)
            result = jax.lax.dynamic_update_slice_in_dim(result, update, j, axis=1)
            j += n
            continue

        for _ in range(n):
            period = period // q
            update = jnp.astype(indices // period, jnp.int8)[..., None]
            result = jax.lax.dynamic_update_slice_in_dim(result, update, j, axis=1)
            indices = indices % period
            j += 1

    return result
