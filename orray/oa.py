import abc
import math
import operator
from functools import partial, reduce, cached_property
from typing import Callable, Collection, Literal, Sequence, overload, Optional
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Int, UInt8


class OrthogonalArray(eqx.Module, Sequence[UInt8[Array, " num_cols"]]):
    """
    An abstract base class for all orthogonal array implementations.

    This class defines the public API and shared functionality, while delegating
    the core data generation logic to its subclasses.
    """

    num_rows: int = eqx.field(static=True)
    num_cols: int = eqx.field(static=True)
    num_levels: int = eqx.field(static=True)
    strength: int = eqx.field(static=True)
    device: jax.Device = eqx.field(static=True)

    """
    an orthogonal array remains valid if an arbitrary (e.g. random)
    value is added to each column (modulo num_levels).
    """
    randomisation_rng: Optional[jax.random.PRNGKey]
    _row_offset: UInt8[Array, " 1 num_cols"]

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
    def shape(self) -> tuple[int, ...]:
        return (self.num_rows, self.num_cols)

    @property
    def dtype(self) -> DTypeLike:
        return jnp.uint8

    @abc.abstractmethod
    def randomise(self, rng: jax.random.PRNGKey) -> "OrthogonalArray":
        """Return a new module with given randomisation_rng."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_device(self, device: jax.Device) -> "OrthogonalArray":
        """Return a new module placed on the given device."""
        raise NotImplementedError

    def __len__(self):
        return self.num_rows

    def __getitem__(self, i):
        """Return a row or a contiguous slice of rows.

        - x[i] returns the i-th row as shape (num_cols,).
        - x[a:b] returns rows [a, b) as shape (b-a, num_cols)

        Only contiguous slices (step=None or step==1) are supported.
        """
        # Single row
        if isinstance(i, int):
            i = i % self.num_rows
            return self.get_batch(start=i, batch_size=1)[0, ...]

        # Slice of rows
        if isinstance(i, slice):
            start, stop, step = i.indices(self.num_rows)
            if step != 1:
                raise NotImplementedError(
                    "Only contiguous slices with step=1 are supported."
                )
            if stop <= start:
                return jnp.zeros((0, self.num_cols), dtype=jnp.uint8)

            return self.get_batch(start=start, batch_size=stop - start)

        raise TypeError(f"Invalid index type: {type(i)!r}. Expected int or slice.")

    @partial(jax.jit, static_argnames="batch_size")
    def get_batch(
        self, start: Int[Array, ""], batch_size: int
    ) -> UInt8[Array, "batch_size num_cols"]:
        return self._get_batch(start, batch_size)

    @abc.abstractmethod
    def _get_batch(
        self, start: Int[Array, ""], batch_size: int
    ) -> UInt8[Array, "batch_size num_cols"]:
        """Get a batch of rows from the orthogonal array.

        Args:
            start: The index of first row of the batch to retrieve. Guaranteed to be in range [0, num_rows).
            batch_size: The number of rows to return. Guaranteed to be in [1, num_rows].
            device: Optional JAX device where the batch should be created. If provided,
                the batch must be computed directly on this device to minimize data transfer.

        Returns:
            A JAX uint8 array with shape (batch_size, num_cols) containing the requested
            batch of orthogonal array rows.

        Notes:
            - For start < num_rows - batch_size - 1: Returns the corresponding rows of the
              orthogonal array
            - For start >= num_rows - batch_size - 1: Returns the remaining rows, padded
              to shape (batch_size, num_cols) with arbitrary values.
            - When device is specified, the batch must be created directly on that
              device rather than computed elsewhere and transferred.
            - Must be `jax.jit` compatible (with static `batch_size` and `device`, and traced `start`)
        """
        raise NotImplementedError

    @jax.jit
    def materialize(self) -> UInt8[Array, "num_rows num_cols"]:
        """Materializes the entire orthogonal array into a single jax array.
        Only use for small arrays!

        Raises:
            MemoryError: If the orthogonal array is too large to fit into memory.
        """
        try:
            full_array = jnp.empty(self.shape, dtype=jnp.uint8, device=self.device)
        except MemoryError as e:
            num_bytes_per_entry = 1  # uint8 = 1 bytes
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
            # use non-jitted version because we jit over the top
            generated_batch = self._get_batch(start, batch_size)
            full_array = full_array.at[start:end, :].set(
                generated_batch[: end - start, :]
            )
        return full_array

    @overload
    def batches(
        self,
        batch_size: int,
        jit_compatible: Literal[False] = ...,
    ) -> Sequence[UInt8[Array, "batch_size num_cols"]]: ...

    @overload
    def batches(
        self,
        batch_size: int,
        jit_compatible: Literal[True],
    ) -> Sequence[
        tuple[UInt8[Array, "batch_size num_cols"], Bool[Array, "batch_size"]]
    ]: ...

    def batches(
        self,
        batch_size: int,
        jit_compatible: bool = False,
    ):
        """Returns a Sequence over batches of rows (runs) of the orthogonal arrays.

        If `jit_compatible` is False (default), each item is a batch array of shape
        (<= batch_size, num_cols); the last batch is truncated to the remaining rows.

        If `jit_compatible` is True, each item is a tuple (batch, mask) where `batch`
        has shape (batch_size, num_cols) and `mask` has shape (batch_size,) marking
        which rows are valid (always all True except potentially on the last batch);
        this enables the user to consume the batches in a JIT context (e.g. `jax.lax.scan`)

        Args:
            batch_size: Number of rows per batch (must be static in a JIT context)
            jit_compatible: Whether to return (batch, mask) with static shapes.
                Must be in [1, num_rows]
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
            ):
                self._parent = p
                self._batch_size = bs
                self._num_batches = math.ceil(p.num_rows / bs)
                self._jit = jit_flag

            def __len__(self):
                return self._num_batches

            def __getitem__(self, i: int):
                if self._jit:
                    i = jnp.mod(i, self._batch_size)
                    start = i * self._batch_size
                    mask = start + jnp.arange(self._batch_size) < self._parent.num_rows
                    # use unjitted version because we assume the user will jit over the top
                    batch = self._parent._get_batch(start, self._batch_size)
                    return batch, mask
                else:
                    if i < -self._num_batches or i >= self._num_batches:
                        raise IndexError(
                            f"Index {i} out of bounds [{-self._num_batches},{self._num_batches})"
                        )
                    i = i % self._num_batches
                    start = i * self._batch_size
                    # use jitted version
                    batch = self._parent.get_batch(start, self._batch_size)
                    if i == len(self) - 1:
                        last_size = self._parent.num_rows % self._batch_size
                        if last_size > 0:
                            return batch[:last_size]
                    return batch

        return _BatchSequence(self, batch_size, jit_compatible)


class MaterializedOrthogonalArray(OrthogonalArray):
    """An OrthogonalArray class that just stores the full
    orthogonal array in memory. Only works for small arrays.
    """

    _oa: UInt8[Array, "num_rows num_cols"]

    def __init__(
        self,
        num_levels: int,
        strength: int,
        orthogonal_array: UInt8[Array, "num_rows num_cols"],
        device: jax.Device | None = None,
        randomisation_rng: Optional[jax.random.PRNGKey] = None,
    ):
        self.num_rows, self.num_cols = orthogonal_array.shape
        self.num_levels = num_levels
        self.strength = strength

        # Ensure orthogonal_array has jnp.uint8 type
        orthogonal_array = jnp.astype(orthogonal_array, jnp.uint8)

        # this ensures that the final batch is correctly padded
        # regardless of the batch_size (which is <= num_rows)
        self._oa = jnp.concatenate(
            [orthogonal_array, jnp.zeros_like(orthogonal_array)], axis=0
        )

        # default device from data; optionally place to requested device during init
        self.device = self._oa.device
        if device is not None:
            self._oa = jax.device_put(self._oa, device)
            self.device = device

        self.randomisation_rng = jax.device_put(randomisation_rng, self.device)
        self._row_offset = _get_row_offset(
            self.randomisation_rng, self.num_cols, self.num_levels, self.device
        )

    def to_device(self, device: jax.Device) -> "MaterializedOrthogonalArray":
        # Slice off padding to pass the exact OA shape expected by __init__.
        base_oa = jax.device_put(self._oa[0 : self.num_rows, :], device)
        return MaterializedOrthogonalArray(
            num_levels=self.num_levels,
            strength=self.strength,
            orthogonal_array=base_oa,
            device=device,
            randomisation_rng=self.randomisation_rng,
        )

    def randomise(self, rng: jax.random.PRNGKey) -> "MaterializedOrthogonalArray":
        # Slice off padding to pass the exact OA shape expected by __init__.
        base_oa = self._oa[0 : self.num_rows, :]
        return MaterializedOrthogonalArray(
            num_levels=self.num_levels,
            strength=self.strength,
            orthogonal_array=base_oa,
            device=self.device,
            randomisation_rng=rng,
        )

    def _get_batch(
        self, start: Int[Array, ""], batch_size: int
    ) -> UInt8[Array, "batch_size num_cols"]:
        # don't truncate at num_rows since self._oa is already padded
        # result = self._oa[start:start + batch_size, :]
        result = jax.lax.dynamic_slice_in_dim(self._oa, start, batch_size, axis=0)
        result = jnp.mod(result + self._row_offset, self.num_levels)

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

    _generator_matrix: UInt8[Array, "k num_cols"]
    _arities: tuple[tuple[int, int], ...] = eqx.field(static=True)
    _mod: int = eqx.field(static=True)
    _even_to_odd: bool = eqx.field(static=True)
    _post_linear_combination_processor: (
        Callable[
            [UInt8[Array, "num_rows num_cols1"]], UInt8[Array, "num_rows num_cols2"]
        ]
        | None
    ) = eqx.field(static=True)

    def __init__(
        self,
        generator_matrix: UInt8[Array, "k num_cols"],
        arities: Collection[tuple[int, int]],
        mod: int,
        num_levels: int,
        strength: int,  # for `even_to_odd`, this is the strength of the final construction, so odd
        binary_oa_even_to_odd_strength: bool = False,
        post_linear_combination_processor: Callable[
            [UInt8[Array, "num_rows num_cols1"]], UInt8[Array, "num_rows num_cols2"]
        ]
        | None = None,
        device: jax.Device | None = None,
        randomisation_rng: Optional[jax.random.PRNGKey] = None,
    ):
        self.num_levels = num_levels
        self.strength = strength

        # ensure dtype is uint8
        self._generator_matrix = jnp.astype(generator_matrix, jnp.uint8)

        self.device = self._generator_matrix.device
        if device is not None:
            self._generator_matrix = jax.device_put(self._generator_matrix, device)
            self.device = device

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

        self.randomisation_rng = jax.device_put(randomisation_rng, self.device)
        self._row_offset = _get_row_offset(
            self.randomisation_rng, self.num_cols, self.num_levels, self.device
        )

    def to_device(self, device: jax.Device) -> "LinearOrthogonalArray":
        return LinearOrthogonalArray(
            generator_matrix=jax.device_put(self._generator_matrix, device),
            arities=self._arities,
            mod=self._mod,
            num_levels=self.num_levels,
            strength=self.strength,
            binary_oa_even_to_odd_strength=self._even_to_odd,
            post_linear_combination_processor=self._post_linear_combination_processor,
            device=device,
            randomisation_rng=self.randomisation_rng,
        )

    def randomise(self, rng: jax.random.PRNGKey) -> "LinearOrthogonalArray":
        return LinearOrthogonalArray(
            generator_matrix=self._generator_matrix,
            arities=self._arities,
            mod=self._mod,
            num_levels=self.num_levels,
            strength=self.strength,
            binary_oa_even_to_odd_strength=self._even_to_odd,
            post_linear_combination_processor=self._post_linear_combination_processor,
            device=self.device,
            randomisation_rng=rng,
        )

    def _get_batch(
        self, start: Int[Array, ""], batch_size: int
    ) -> UInt8[Array, "batch_size num_cols"]:
        if self._even_to_odd:
            _batch_size = batch_size // 2 + 1
            unmodified_rows = get_row_batch_of_trivial_mixed_level_oa(
                i0=start // 2,
                arities=self._arities,
                batch_size=_batch_size,
                device=self.device,
            )
            rows = jnp.repeat(unmodified_rows, repeats=2, axis=0)
            rows = jax.lax.dynamic_slice_in_dim(rows, start % 2, batch_size, axis=0)
        else:
            rows = get_row_batch_of_trivial_mixed_level_oa(
                i0=start,
                arities=self._arities,
                batch_size=batch_size,
                device=self.device,
            )
        batch = jnp.mod(rows @ self._generator_matrix, self._mod)
        if self._post_linear_combination_processor is not None:
            batch = self._post_linear_combination_processor(batch)

        if self._even_to_odd:
            batch = jnp.concatenate(
                (
                    batch,
                    jnp.zeros((batch_size, 1), dtype=jnp.uint8, device=self.device),
                ),
                axis=1,
            )
            idx = jnp.arange(batch_size, device=self.device)
            flip_mask = jnp.bitwise_and(idx + start, 1).astype(bool)
            batch = jnp.where(flip_mask[:, None], 1 - batch, batch)

        batch = jnp.mod(batch + self._row_offset, self.num_levels)
        return batch


###################################################
#                     Helpers                     #
###################################################


def get_row_batch_of_trivial_mixed_level_oa(
    i0: Int[Array, ""],
    arities: tuple[tuple[int, int], ...],
    batch_size: int,
    device: jax.Device | None = None,
) -> UInt8[Array, "batch_size num_cols"]:
    n_cols = sum(n for (n, _) in arities)
    n_rows = reduce(operator.mul, [pow(q, n) for (n, q) in arities])
    result = jnp.zeros((batch_size, n_cols), dtype=jnp.uint8, device=device)

    indices = i0 + jnp.arange(batch_size, device=device)

    j = jnp.asarray(0)  # col index
    period = n_rows
    for n, q in arities:
        if q == 2:
            ints = i0 + jnp.arange(batch_size, device=device)
            bits = jnp.arange(n, device=device)
            _update = jnp.bitwise_and(jnp.right_shift(ints[:, None], bits[None, :]), 1)
            update = _update.astype(jnp.uint8)
            result = jax.lax.dynamic_update_slice_in_dim(result, update, j, axis=1)
            j += n
            continue

        for _ in range(n):
            period = period // q
            update = jnp.astype(indices // period, jnp.uint8)[..., None]
            result = jax.lax.dynamic_update_slice_in_dim(result, update, j, axis=1)
            indices = indices % period
            j += 1

    return result


def _get_row_offset(
    rng: Optional[jax.random.PRNGKey],
    num_cols: int,
    num_levels: int,
    device: jax.Device | None = None,
) -> UInt8[Array, " 1 num_cols"]:
    if rng is None:
        return jnp.zeros((1, num_cols), dtype=jnp.uint8, device=device)
    else:
        return jax.random.randint(
            rng,
            shape=(1, num_cols),
            minval=0,
            maxval=num_levels,
            dtype=jnp.uint8,
        )
