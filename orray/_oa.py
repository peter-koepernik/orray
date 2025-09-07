import itertools
import operator
from collections.abc import Callable, Collection, Sequence

import equinox as eqx
from functools import partial, reduce
import galois
import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import Int, UInt8, Array
from typing import Any, TypeAlias

Device: TypeAlias = Any

### functionality for batch-wise generation

@partial(jax.jit, static_argnames=("arities", "batch_size"))
def get_row_batch_of_trivial_mixed_level_oa(
    i0: Int[Array, ""],
    arities: tuple[tuple[int, int], ...],
    batch_size: int,
) -> tuple[UInt8[Array, "batch_size factors"], UInt8[Array, "batch_size"]]:  # noqa: F821
    n_cols = sum(n for (n, _) in arities)
    n_rows = reduce(operator.mul, [pow(q, n) for (n, q) in arities])
    result = jnp.zeros((batch_size, n_cols), dtype=jnp.uint8)
    
    # Calculate valid mask
    indices = i0 + jnp.arange(batch_size)
    valid = (indices < n_rows).astype(jnp.uint8)

    j = jnp.asarray(0)  # col index
    period = n_rows
    for n, q in arities:
        if q == 2:
            ints = i0 + jnp.arange(batch_size)
            bits = jnp.arange(n)
            _update = jnp.bitwise_and(jnp.right_shift(ints[:, None], bits), 1)
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
    return result, valid


class AbstractOABatchSequence(Sequence):
    """A generator that iterates over batches of rows of an orthogonal array"""

    runs: int
    factors: int
    levels: int
    batch_size: int
    strength: int

    def __len__(self) -> int:
        """returns number of batches"""
        return self.runs // self.batch_size + int(self.runs % self.batch_size > 0)

    def get_num_runs(self) -> int:
        return self.runs

    def shape(self) -> tuple[int, int]:
        return self.runs, self.factors

class OABatchSequence(AbstractOABatchSequence):
    """
    Constructs orthogonal arrays by taking linear combinations of rows of a given
    generator matrix.
    """

    generator_matrix: np.ndarray
    arities: Collection[tuple[int, int]]
    mod: int
    even_to_odd: bool
    post_linear_combination_processor: Callable[[np.ndarray], np.ndarray] | None
    device: Device | jax.sharding.Sharding | None

    def __init__(
        self,
        generator_matrix: np.ndarray,
        arities: Collection[tuple[int, int]],
        mod: int,
        levels: int,
        strength: int,
        batch_size: int = 0,
        binary_oa_even_to_odd_strength: bool = False,
        post_linear_combination_processor: Callable[[np.ndarray], np.ndarray]
        | None = None,
        device: Device | jax.sharding.Sharding | None = None,
    ):
        """
        The generator functions by iterating through all possible linear combinations of
        the rows of `generator_matrix` modulo `mod`, where the arities of the linear
        combinations of the rows are passed in `arities`. each batch of rows obtained in
        this way is then passed to the `post_linear_combination_processor`, if given.
        currently the only constructions that use this postprocessing are the kerdock
        and DG constructions, where this is the gray map.

        If `binary_oa_even_to_odd_strength` is true, then every row is duplicated, once
        with an additional 0, and once with an additional 1 and inverted

        The `OA` field gives the trivial option of just storing the full array and
        iterating over its rows;
        """
        self.generator_matrix = generator_matrix
        self.levels = levels
        self.arities = arities
        self.mod = mod
        self.strength = strength
        self.even_to_odd = binary_oa_even_to_odd_strength
        self.post_linear_combination_processor = post_linear_combination_processor
        self.device = device

        self.runs = 1
        for n_cols, q in arities:
            self.runs *= q**n_cols
        if self.even_to_odd:
            self.runs *= 2
        if self.post_linear_combination_processor is None:
            self.factors = self.generator_matrix.shape[1]
        else:
            # infer how many rows it will turn into
            factors = jax.eval_shape(
                self.post_linear_combination_processor, self.generator_matrix
            )
            self.factors = factors.shape[1]
        if self.even_to_odd:
            self.factors += 1
        self.batch_size = batch_size if batch_size != 0 else self.runs

    def __getitem__(self, i):
        n_batches = len(self)
        if i < -n_batches or i >= n_batches:
            raise IndexError(
                f"batch index {i} is out of bounds for sequence of length {n_batches}"
            )
        i = i % n_batches
        batch_start_row = np.asarray(i * self.batch_size)
        batch_end_row = min(self.runs, (i + 1) * self.batch_size)
        batch_n_rows = int(batch_end_row - batch_start_row)

        # TODO: fix the even_to_odd construction.
        # TODO: reprofile the construciton costs.
        if self.even_to_odd:
            batch_size = int(batch_end_row // 2 - batch_start_row // 2)
            unmodified_rows, _ = get_row_batch_of_trivial_mixed_level_oa(
                i0=batch_start_row // 2, arities=self.arities, batch_size=batch_size
            )
            rows = jnp.repeat(unmodified_rows, repeats=2, axis=0)
            rows = rows[batch_start_row % 2 : batch_n_rows, :]
        else:
            rows, _ = get_row_batch_of_trivial_mixed_level_oa(
                i0=batch_start_row, arities=self.arities, batch_size=batch_n_rows
            )
        OA = jnp.mod(rows @ self.generator_matrix, self.mod)
        if self.post_linear_combination_processor is not None:
            OA = self.post_linear_combination_processor(OA)

        if self.even_to_odd:
            OA = jnp.concatenate(
                (OA, jnp.zeros((batch_n_rows, 1), dtype=np.uint8)), axis=1
            )
            first_flip_row = 1 - (batch_start_row % 2)
            OA = OA.at[first_flip_row::2, :].set(1 - OA[first_flip_row::2, :])
        return OA



class OABatchSequenceJax(AbstractOABatchSequence):
    """
    Constructs orthogonal arrays by taking linear combinations of rows of a given
    generator matrix.
    """

    generator_matrix: np.ndarray
    arities: tuple[tuple[int, int], ...]
    mod: int
    even_to_odd: bool
    post_linear_combination_processor: Callable[[np.ndarray], np.ndarray] | None
    device: Device | jax.sharding.Sharding | None

    def __init__(
        self,
        generator_matrix: Int[Array, "rows cols"],
        arities: Collection[tuple[int, int]],
        mod: int,
        levels: int,
        strength: int,
        batch_size: int = 0,
        binary_oa_even_to_odd_strength: bool = False,
        post_linear_combination_processor: Callable[[Int[Array, "runs factors1"]], Int[Array, "runs factors2"]]
        | None = None,
        device: Device | jax.sharding.Sharding | None = None,
    ):
        """
        The generator functions by iterating through all possible linear combinations of
        the rows of `generator_matrix` modulo `mod`, where the arities of the linear
        combinations of the rows are passed in `arities`. each batch of rows obtained in
        this way is then passed to the `post_linear_combination_processor`, if given.
        currently the only constructions that use this postprocessing are the kerdock
        and DG constructions, where this is the gray map.

        If `binary_oa_even_to_odd_strength` is true, then every row is duplicated, once
        with an additional 0, and once with an additional 1 and inverted

        The `OA` field gives the trivial option of just storing the full array and
        iterating over its rows;
        """
        self.generator_matrix = generator_matrix
        self.levels = levels
        self.arities = tuple(tuple(a) for a in arities)  # make hashable
        self.mod = mod
        self.strength = strength
        self.even_to_odd = binary_oa_even_to_odd_strength
        self.post_linear_combination_processor = post_linear_combination_processor
        self.device = device

        # final shape of OA = (runs, factors)
        self.runs = 1  # number of rows in OA
        for n_cols, q in arities:
            self.runs *= q**n_cols
        if self.even_to_odd:
            self.runs *= 2
        if self.post_linear_combination_processor is None:
            self.factors = self.generator_matrix.shape[1]
        else:
            # infer how many rows it will turn into
            factors = jax.eval_shape(
                self.post_linear_combination_processor, self.generator_matrix
            )
            self.factors = factors.shape[1]
        if self.even_to_odd:
            self.factors += 1
        self.batch_size = batch_size if batch_size != 0 else self.runs
        if self.batch_size % 2 == 1:
            raise ValueError(f"batch_size must be even but is {self.batch_size}")

    def __getitem__(self, i):
        batch_start_row = i * self.batch_size

        valid = (batch_start_row + jnp.arange(self.batch_size) < self.runs).astype(jnp.uint8)

        # TODO: fix the even_to_odd construction.
        # TODO: reprofile the construciton costs.
        if self.even_to_odd:
            #batch_size = int(batch_end_row // 2 - batch_start_row // 2)
            batch_size = self.batch_size // 2
            unmodified_rows, _ = get_row_batch_of_trivial_mixed_level_oa(
                i0=batch_start_row // 2, arities=self.arities, batch_size=batch_size
            )
            rows = jnp.repeat(unmodified_rows, repeats=2, axis=0)
        else:
            rows, _ = get_row_batch_of_trivial_mixed_level_oa(
                i0=batch_start_row, arities=self.arities, batch_size=self.batch_size
            )
        OA = jnp.mod(rows @ self.generator_matrix, self.mod)
        if self.post_linear_combination_processor is not None:
            OA = self.post_linear_combination_processor(OA)

        if self.even_to_odd:
            OA = jnp.concatenate(
                (OA, jnp.zeros((self.batch_size, 1), dtype=jnp.uint8)), axis=1
            )
            OA = OA.at[1::2, :].set(1 - OA[1::2, :])
        return OA, valid

def make_batch_sequence(jit_compatible: bool = False, **args):
    if jit_compatible:
        return OABatchSequenceJax(**args)
    else:
        return OABatchSequence(**args)

@partial(jax.jit, static_argnames="levels")
def randomise_oa(
    orthogonal_array: UInt8[Array, "runs factors"],
    levels: int,
    rng: jax.random.PRNGKey,
    #random_generator: np.random.Generator | None = None,
) -> UInt8[np.ndarray, "runs factors"]:
    """Make columns uniform random while preserving orthogonality of 'orthogonal_array'.

    Samples one uniform random number in {0,...,q-1} for each column, and adds these
    numbers to the elements of the respective columns modulo q. This preserves the
    orthogonality of the array, but guarantees that every individual row is, marginally,
    a sequence of independent uniform random numbers in {0,...,q-1}.

    Args:
        orthogonal_array: the orthogonal array to randomise
        levels: the number of levels in the orthogonal array
        random_generator: the random generator

    Returns:
        modified orthogonal array whose columns are uniform randomly distributed.
    """
    _, factors = orthogonal_array.shape
    random_numbers = jax.random.randint(rng, shape=(1, factors), minval=0, maxval=levels, dtype=jnp.uint8)

    randomised_oa = jnp.mod(orthogonal_array + random_numbers, levels)
    return randomised_oa


def get_num_rows(
    factors: int,
    levels: int,
    strength: int,
    heuristic: bool = False,
) -> int:
    oa = generate_oa(factors, levels, strength, heuristic=heuristic, verbose=False)
    return oa.get_num_runs()

### master method for generating orthogonal arrays
def generate_oa(
    factors: int,
    levels: int,
    strength: int,
    batch_size: int = 0,
    *,
    heuristic: bool = False,
    jit_compatible: bool = False,
    verbose: bool = False,
) -> AbstractOABatchSequence:
    """constructs an orthogonal array with the given strength and number of levels and
    columns, with (best-effort) minimal number of rows.

    The `heuristic` flag can only be set if num_levels=2 and strength is odd and at
    least 5. The idea is that e.g. anbinary strength 5 array is actually almost of
    strength 6, which means that the `binary_oa_even_to_odd_strength` method will turn
    it into an array of almost strenth 7 and only twice as many rows. If the flag is
    set, then a binary OA of `strength`-2 is constructed, passed to
    `binary_oa_even_to_odd_strength`, and returned.

    if `batch_size` != 0, returns an iterator over batches of rows instead of the entire
    array; it has __len__ implemented which returns the total number of rows.

    if `randomise_columns` is true then all columns have a uniform random integer in
    {0,...,num_levels-1} added to them (modulo num_levels), which preserves
    orthogonality but ensures that each row is (marginally) distributed as a random
    sequence of independent uniform {0,...,num_levels-1} variables. Optionally you can
    also pass a seed in `randomise_columns_seed` (default 0).
    """
    assert strength >= 1
    assert factors >= 1
    assert levels >= 2

    if strength == 1:
        oa_generator = generate_oa_strength1(levels, batch_size=batch_size, jit_compatible=jit_compatible)
    elif factors <= strength:
        oa_generator = generate_trivial_oa(factors, levels, batch_size=batch_size, jit_compatible=jit_compatible)
    # now since strength < num_cols, the final array must have at least num_levels^strength rows
    elif factors <= levels:
        assert galois.is_prime(levels)
        # has exactly num_levels^strength rows, so is now optimal
        oa_generator = generate_oa_vandermonde(
            levels, 1, strength, batch_size=batch_size, jit_compatible=jit_compatible
        )
    elif strength == 2:
        q = levels
        assert galois.is_prime(q)
        # need smallest m with (q^m-1)/(q-1) >= num_cols. since LHS > q^(m-1), a lower
        # bound on m is obtained by choosing the smallest m such that q^(m-1) >= num_cols
        m = 1 + int(np.ceil(np.log(factors) / np.log(q)))
        assert (q**m - 1) // (q - 1) <= factors
        while (q**m - 1) // (q - 1) < factors:
            m += 1
        oa_generator = generate_oa_strength2(m, q, batch_size=batch_size, jit_compatible=jit_compatible)
    # now num_cols > max(strength, num_levels), strength >= 3
    elif levels == 2:
        oa_generator = _generate_binary_oa(
            factors, strength, batch_size, heuristic=heuristic, jit_compatible=jit_compatible, verbose=verbose
        )
    elif strength == 3:
        oa_generator = _generate_oa_strength3(factors, levels, batch_size, jit_compatible=jit_compatible)
    elif strength == 4 and levels == 3:
        # cap set construction OA(3^(2m), 3^m, 3, 4)
        m = int(np.ceil(np.log(factors) / np.log(3)))
        oa_generator = generate_oa_q3_strength4(m, batch_size=batch_size, jit_compatible=jit_compatible)
    else:
        # strength >= 4, levels >= 3 and at least one of them ">". Only remaining option
        # is vandermonde construction OA(q^(ms), q^m, q, s)
        m = int(np.ceil(np.log(factors) / np.log(levels)))
        assert levels**m >= factors and levels ** (m - 1) < factors
        oa_generator = generate_oa_vandermonde(
            levels, m, strength, batch_size=batch_size, jit_compatible=jit_compatible
        )
    num_rows = len(oa_generator)

    # check if trivial array is better (for example if num_cols=4 and strength=5 binary array)
    if levels**factors <= num_rows:
        oa_generator = generate_trivial_oa(factors, levels, batch_size=batch_size, jit_compatible=jit_compatible)
    assert oa_generator.factors >= factors, (
        f"bug in `generate_oa` method, generated oa has {oa_generator.factors} factors (columns), but {factors} were requested"
    )
    return oa_generator


### construction methods


def _generate_binary_oa(
    factors: int,
    strength: int,
    batch_size: int = 0,
    *,
    heuristic: bool = False,
    jit_compatible: bool = False,
    verbose: bool = False,
):
    # smallest m such that 2^m >= num_cols
    m = int(np.ceil(np.log(factors) / np.log(2)))
    assert 2**m >= factors
    assert m == 1 or 2 ** (m - 1) < factors

    if heuristic:
        if strength < 5:
            raise ValueError(
                "can only use heuristic construction for strength at least 5, but "
                f"'strength'={strength}"
            )
        if strength % 2 == 0:
            strength -= 2
        else:
            strength -= 3
        # strength -= 8

    m_is_even = m % 2 == 0
    match strength:
        # Kerdock: OA(2^(2m), 2^m, 2, 5), m>=4 even
        case 5 if m >= 4 and m_is_even:
            if verbose:
                print("OA(Selection): Kerdock")
            return generate_oa_kerdock(m, batch_size, even_to_odd=heuristic, jit_compatible=jit_compatible)
        # Delsarte-Goethals: OA(2^(3m-1), 2^m, 2, 7), m>=4 even
        # Strength 6 is an exceptional case where the strength 7 DG construction is more
        # efficient (has fewer rows) than the 'weaker' strength 6 BR construction,
        # respectively 2^(m-1) vs 2^(3m) rows!
        case 6 | 7 if m >= 4 and m_is_even:
            if verbose:
                print("OA(Selection): Delsarte-Goethals")
            return generate_oa_delsarte_goethals(m, batch_size, even_to_odd=heuristic, jit_compatible=jit_compatible)
        # Bose: OA(2^(2m+1), 2^m, 2, 5)
        # Bose: OA(2^(3m+1), 2^m, 2, 7)
        case _:
            # if strength is even, this has 2^m-1 columns, if strength is odd 2^m >= factors
            if strength % 2 == 0 and 2**m - 1 < factors:
                m += 1
                assert 2**m - 1 >= factors
            if verbose:
                print("OA(Selection): Bose-Ray")
            return generate_oa_bose_ray(m, strength, batch_size=batch_size, jit_compatible=jit_compatible)


def generate_oa_vandermonde(q: int, m: int, s: int, batch_size: int = 0, jit_compatible: bool = False):
    """Returns an OA(q^(ms), q^m, q, s) for a prime number q

    Runtime: O(n log n) where n = output-size"""
    assert q >= 2
    assert m >= 1
    assert s >= 1
    assert galois.is_prime(q)
    # construct the vandermonde matrix whose columns are [1 x x^2 ... x^(s-1)] and x
    # loops through all elements of F_q (including zero)
    galois_field = galois.GF(q**m)

    def get_vector_from_galois_element(_x: ...):
        r"""
        Given an element x of GF(q^m), calculates (1,x,x^2,...,x^(s-1)) \in GF(q^m)^t,
        and returns it as a vector in F_q^(m*t)
        """
        repeated_x = itertools.chain([galois_field(1)], itertools.repeat(_x, s - 1))
        galois_vector = itertools.accumulate(repeated_x, operator.mul)
        return np.concatenate([y.vector() for y in galois_vector])

    columns = [np.asarray([1] + (m * s - 1) * [0], dtype=np.uint8)] + [
        get_vector_from_galois_element(x) for x in galois_field.units
    ]
    # now M is a (m*s, q^m) matrix over F_q whose columns are s-wise linearly
    # independent over F_q
    M = np.asarray(np.column_stack(columns), dtype=np.uint8)
    generator = make_batch_sequence(
        jit_compatible=jit_compatible,
        generator_matrix=M,
        arities=[(m * s, q)],
        mod=q,
        levels=q,
        strength=s,
        batch_size=batch_size,
    )
    return generator


def generate_oa_bose_ray(m: int, s: int, batch_size: int = 0, jit_compatible: bool = False):
    """
    If `s=2u` is even: generates an OA(2^(mu), 2^m-1, 2, s), so N = (k+1)^(s/2)
    If `s=2u+1` is odd: generates an OA(2^(mu+1), 2^m, 2, s), so N = 2 k^floor(s/2)
    Note that if applicable to the concrete number of columns that's needed, and s=5 or
    s=7, then respectively kerdock or delsarte-goethals is better!

    Construction time is O(n log n) where n is the output size

    The construction is based on the construction of bose and ray-chauduri of a set of
    2^m-1 (2u)-wise linearly independent vectors in F_2^(mt), see sections 3 and 4 of

    ```bibtex
    @article{bose1960,
        title = {On a class of error correcting binary group codes},
        author = {R.C. Bose and D.K. Ray-Chaudhuri},
        journal = {Information and Control},
        volume = {3},
        number = {1},
        pages = {68-79},
        year = {1960},
        issn = {0019-9958},
        doi = {https://doi.org/10.1016/S0019-9958(60)90287-4},
    }
    ```
    """
    assert s >= 2
    t = s // 2
    # (mt) x max(mt,2^m-1), matrix with 2t-wise linearly independent columns
    vecs = generate_2t_wise_linearly_independent_vectors(m, t)
    generator = make_batch_sequence(
        jit_compatible=jit_compatible,
        generator_matrix=vecs,
        arities=[(m * t, 2)],
        mod=2,
        levels=2,
        strength=s,
        batch_size=batch_size,
        binary_oa_even_to_odd_strength=(s % 2 == 1),
    )
    return generator


def generate_2t_wise_linearly_independent_vectors(
    m: int, t: int
) -> Int[np.ndarray, "m*t ..."]:
    """Returns a set of 2^m-1 vectors in F_2^{mt} such that any set of 2t of
    them are linearly independent (i.e. a 2t-Sidon set in F_2^{mt}). Returns the result
    as a 0-1 matrix of shape (mt) x 2^m-1."""
    galois_field = galois.GF(2**m)
    assert m >= 1
    assert t >= 1

    def get_mt_vector(_m: ...) -> Int[np.ndarray, " mt"]:
        """Given a vector alpha in F_2^m, returns the vector (alpha, alpha^3, ...,
        alpha^(2t-1)) in F_2^(mt), where multiplication in F_2^m is defined through the
        identification of F_2^m with the Galois group GF(2^m)."""
        repeated_m_vector = itertools.chain([_m], itertools.repeat(_m**2, t - 1))
        mt_sequence = itertools.accumulate(repeated_m_vector, operator.mul)
        return np.concatenate([mt.vector() for mt in mt_sequence])

    mt_vector_generator = (get_mt_vector(m) for m in galois_field.units)
    return np.asarray(np.column_stack(list(mt_vector_generator)), dtype=np.uint8)


def binary_oa_even_to_odd_strength(OA: np.ndarray):
    """Takes an orthogonal array OA(N, k, 2, 2u) and outputs an orthogonal array
    OA(2N, k+1, 2, 2u+1), i.e. increments the strength of (and adds a column to) a
    binary array with even strength at the cost of doubling the number of rows. This is
    sharp in the sense that an OA of former type exists iff one of latter type exists.

    If OA is the input, the output is, in block matrix form,
    [    OA | 0]
    [not OA | 1]

    For this construction, see Theorem 2.24 in
    ```bibtex
    @book{hedayat99,
        title = {Orthogonal Arrays: Theory and Applications},
        author = {Hedayat, A. Samad and Sloane, Neil James Alexander and Stufken, John},
        isbn = {0387987665},
        publisher = {Springer},
        address = {New York},
        year = {1999}
    }
    ```
    """
    N, k = OA.shape
    OA = np.hstack((OA, np.zeros((N, 1)))).astype(bool)
    OA = np.vstack((OA, np.logical_not(OA)))
    return OA.astype(np.uint8)


def generate_oa_delsarte_goethals(
    m: int, batch_size: int = 0, *, even_to_odd: bool = False, jit_compatible: bool = False
):
    """Generates an OA(2^(3m-1), 2^m, 2, 7), i.e. a binary 8^m/2 x 2^m array of
    strength 7, where m >= 4 is an even integer.  Based on the linear construction of
    the Delsarte-Goethals code (reference below). If `even_to_odd` is true, applies the
    `even_to_odd` construction; the idea is that in practice, kerdock almost has
    strength 6, so after that construction it almost has strength 7.

    When defined, it satisfies N = .5 k^3, which is a quarter as many columns given k
    than the best linear array, which has N = 2k^3

    Construction time O(n log n) = O(m*2^(4m)) where n = 2^(4m-1) is the output size

    Original construction of the Delsarte-Goethals code is in
    ```bibtex
    @article{delsarte-goethals,
        title = {Alternating bilinear forms over GF(q)},
        author = {P. Delsarte and J.M. Goethals},
        journal = {Journal of Combinatorial Theory, Series A},
        volume = {19},
        number = {1},
        pages = {26-50},
        year = {1975},
        doi = {https://doi.org/10.1016/0097-3165(75)90090-4},
    }t
    ```

    The construction here is derived from section 6 of
    ```bibtex
    @article{HKCSS94,
        title={The Z/sub 4/-linearity of Kerdock, Preparata, Goethals, and related codes},
        author={Hammons, A.R. and Kumar, P.V. and Calderbank, A.R. and Sloane, N.J.A. and Sole, P.},
        journal={IEEE Transactions on Information Theory},
        year={1994},
        volume={40},
        number={2},
        pages={301-319},
        doi={10.1109/18.312154}
    }
    ```
    """
    assert m % 2 == 0
    assert m >= 4
    m -= 1  # make m consistent with the literature: >= 3 and odd
    n = 2**m - 1
    xi_table = calculate_xi_table(m, 3 * (n - 1))
    # the generator of the DG code is made up of three vertically stacked blocks of
    # respective lengths 1, m, m
    first_block = np.ones((1, n + 1), dtype=np.uint8)
    second_block = xi_table[: n + 1, :].T
    powers = list(range(3, 3 * n, 3))  # e.g. third power of xi has index 4 in xi_table
    indices = [0, 1] + [1 + i for i in powers]
    third_block = 2 * xi_table[indices, :].T
    # (2m+1) x 2^m, the generator of the Delsarte-Goethals code
    G = np.vstack((first_block, second_block, third_block), casting="no")
    generator = make_batch_sequence(
        jit_compatible=jit_compatible,
        generator_matrix=G,
        arities=[(m + 1, 4), (m, 2)],
        mod=4,
        levels=2,
        strength=7,
        batch_size=batch_size,
        post_linear_combination_processor=gray_map,
        binary_oa_even_to_odd_strength=even_to_odd,
    )
    return generator


def generate_oa_kerdock(m: int, batch_size: int = 0, *, even_to_odd: bool = False, jit_compatible: bool = False):
    """Generates an OA(4^m, 2^m, 2, 5), i.e. a (non-linear) binary 4^m x 2^m array of
    strength 5, where m >= 4 is an even integer. Based on the linear construction of the
    Kerdock code (reference below). If `even_to_odd` is true, applies the `even_to_odd`
    construction; the idea is that in practice, kerdock almost has strength 6, so after
    that construction it almost has strength 7.

    When defined, it satisfies N = k^2, which is half as many columns given k as the
    best linear array, which has N = 2k^2.

    Construction time O(n log n) = O(m*2^(3m)) where n = 2^(3m) is the output size

    Original construction of the Kerdock code is in
    ```bibtex
    @article{kerdock,
        title = {A class of low-rate nonlinear binary codes},
        author = {A.M. Kerdock},
        journal = {Information and Control},
        volume = {20},
        number = {2},
        pages = {182-187},
        year = {1972},
        issn = {0019-9958},
        doi = {https://doi.org/10.1016/S0019-9958(72)90376-2},
    }
    ```

    The construction here is derived from section 4 of
    ```bibtex
    @article{HKCSS94,
        title={The Z/sub 4/-linearity of Kerdock, Preparata, Goethals, and related codes},
        author={Hammons, A.R. and Kumar, P.V. and Calderbank, A.R. and Sloane, N.J.A. and Sole, P.},
        journal={IEEE Transactions on Information Theory},
        year={1994},
        volume={40},
        number={2},
        pages={301-319},
        doi={10.1109/18.312154}
    }
    ```
    """
    assert m % 2 == 0
    assert m >= 4
    m -= 1  # make m consistent with the literature: >= 3 and odd
    xi_table = calculate_xi_table(m)
    ones_column = np.ones((2**m, 1), dtype=np.uint8)
    # (m+1) x 2^m, generator of the kerdock code
    G = np.hstack((ones_column, xi_table), casting="no").T
    generator = make_batch_sequence(
        jit_compatible=jit_compatible,
        generator_matrix=G,
        arities=[(m + 1, 4)],
        mod=4,
        levels=2,
        strength=5,
        batch_size=batch_size,
        post_linear_combination_processor=gray_map,
        binary_oa_even_to_odd_strength=even_to_odd,
    )
    return generator


def get_h_polynomial(m: int):
    """
    Returns a monic primitive basic irreducible polynomial of odd degree m >= 3 in Z_4.
    For example, [1, 2, 1, 3] stands for X^3 + 2X^2 + X + 3.

    Polynomials currently constructed manually, previously (and still commented out),
    polynomials were taken from table 1 of

    ```bibtex
    @article{BHK92,
        title={4-phase sequences with near-optimum correlation properties},
        author={Boztas, S. and Hammons, R. and Kumar, P.Y.},
        journal={IEEE Transactions on Information Theory},
        year={1992},
        volume={38},
        number={3},
        pages={1101-1113},
        doi={10.1109/18.135649}
    }
    ```
    """
    assert m >= 3
    assert m % 2 == 1
    primitive = galois.primitive_poly(2, m)
    h = get_h_polynomial_from_primitive_F2(primitive.coefficients())
    return h


def get_h_polynomial_from_primitive_F2(primitive: np.ndarray[int]):
    """Takes a primitive polynomial over F_2 of some odd degree m and turns it into a
    monic primitive basic irreducible polynomial over Z_4, using Graeffe's method.
    See e.g. section III.A of

    ```bibtex
    @article{HKCSS94,
        title={The Z/sub 4/-linearity of Kerdock, Preparata, Goethals, and related codes},
        author={Hammons, A.R. and Kumar, P.V. and Calderbank, A.R. and Sloane, N.J.A. and Sole, P.},
        journal={IEEE Transactions on Information Theory},
        year={1994},
        volume={40},
        number={2},
        pages={301-319},
        doi={10.1109/18.312154}
    }
    ```
    """
    primitive = np.array(primitive, dtype=int)
    assert np.all(primitive >= 0) and np.all(primitive <= 1)
    assert len(primitive.shape) == 1
    m = len(primitive) - 1
    assert m % 2 == 1
    # split primitive in even and odd coefficients
    e = np.copy(primitive)
    e[::2] = 0
    d = -np.copy(primitive)
    d[1::2] = 0
    # square e and d
    e_sq = np.polymul(e, e)
    d_sq = np.polymul(d, d)
    assert len(d_sq) == 2 * m + 1
    e_sq = np.pad(e_sq, (2 * m + 1 - len(e_sq), 0), constant_values=0)
    # now h(x^2) = (+ or -) e(x)^2 - d(x)^2
    h_of_x_sq = np.mod(e_sq - d_sq, 4).astype(np.uint8)
    assert np.all(h_of_x_sq[1::2] == 0)
    h = h_of_x_sq[::2]
    if h[0] == 3:  # i.e. h[0] = -1 in Z_4
        h = np.mod(-h, 4)
    assert h[0] == 1
    assert h[-1] == 3
    return h


def calculate_xi_table(
    m: int, max_power_of_xi: int = -1, include_zero_row: bool = True
):
    r"""
    Let h be the monic primitive basic irreducible polynomial of degree m that is
    returned by `get_h_polynomial`. Then Z_4[X] / h(X) is a ring with 4^m elements
    called the Galois ring R = GR(4^m). (Different choices of h lead to isomorphic
    descriptions of the ring). Now let xi \in Z_4[X] be a root of h and such that
    xi^n = 1 (it always exists), where n = 2^m-1. Then it holds that R = Z_4[xi],
    and in fact every element c of R has a unique representation of the form

        c = sum_{r=0}^(m-1) b_r xi^r

    In particular, arbitrarily large powers of xi can be expressed as Z_4-linear
    combinations of 1, xi, ..., xi^(m-1), and the rows of the table returned by this
    function are these coefficients. That is, each row is a sequence b0,b1,...,b_{m-1}.
    The first row is just the zero-row, the 2nd row is for 1 = xi^0, the third for xi^1
    etc, up to the maximum power specified (in particular, rows 2 to m+1 are just the
    m x m identity matrix).

    The computation of the table is done through a shift register with feedback
    polynomial h(X).

    Runtime is O(n) where n = max_power_of_xi * m is the output size

    For reference, see section III.A in

    ```bibtex
    @article{HKCSS94,
        title={The Z/sub 4/-linearity of Kerdock, Preparata, Goethals, and related codes},
        author={Hammons, A.R. and Kumar, P.V. and Calderbank, A.R. and Sloane, N.J.A. and Sole, P.},
        journal={IEEE Transactions on Information Theory},
        year={1994},
        volume={40},
        number={2},
        pages={301-319},
        doi={10.1109/18.312154}
    }
    ```
    """
    assert m >= 3
    assert m % 2 == 1
    n = 2**m - 1
    if max_power_of_xi == -1:
        max_power_of_xi = n - 1
    assert max_power_of_xi >= 1

    h = get_h_polynomial(m)
    feedback = np.mod(
        -h[1:][::-1], 4
    )  # delete leading one, reverse order, take minus modulo 4
    assert feedback[0] == 1
    offset = int(include_zero_row)
    table = np.zeros((offset + 1 + max_power_of_xi, m), dtype=np.uint8)
    # row 1
    table[offset, 0] = 1
    for i in range(offset + 1, offset + 1 + max_power_of_xi):
        table[i, 1:] = table[i - 1, :-1]  # shift right
        table[i, :] = np.mod(table[i, :] + table[i - 1, -1] * feedback, 4)
    return table


@eqx.filter_jit
def gray_map(
    orthogonal_array: UInt8[np.ndarray, "runs factors"],
) -> UInt8[np.ndarray, "runs factors"]:
    """maps a 4-ary integer array of some shape (n1, n2) to a 2-ary integer array of
    shape (n1, 2*n2) through an element-wise application of the *Gray map*:

    The gray map is a *non-linear* map from Z_4 to Z_2^2 that is defined by
     0 -> 0 0
     1 -> 0 1
     2 -> 1 1
     3 -> 1 0

    1st coordinate: x -> (x >= 2)
    2nd coordinate: x -> (x >= 1 and x <= 2)
    """
    orthogonal_array_1 = orthogonal_array >= 2
    orthogonal_array_2 = jnp.logical_and(orthogonal_array >= 1, orthogonal_array <= 2)
    _orthogonal_array = jnp.hstack((orthogonal_array_1, orthogonal_array_2))
    return _orthogonal_array.astype(jnp.uint8)


def generate_trivial_oa(n_cols: int, q: int, batch_size: int = 0, jit_compatible: bool = False) -> OABatchSequence:
    """returns the trivial OA with q^n_cols columns"""
    return make_batch_sequence(
        jit_compatible=jit_compatible,
        generator_matrix=np.eye(n_cols, dtype=np.uint8),
        arities=[(n_cols, q)],
        mod=q,
        levels=q,
        strength=n_cols,
        batch_size=batch_size,
    )


def generate_trivial_mixed_level_oa(
    arities: Collection[tuple[int, int]],
) -> np.ndarray[int]:
    """args should be a list of (n_cols, q) meaning that many columns should be q-ary"""
    n_total_rows = 1
    n_total_cols = 0
    for n_cols, q in arities:
        assert isinstance(n_cols, int)
        assert isinstance(q, int)
        assert n_cols >= 1
        assert q >= 1
        n_total_cols += n_cols
        n_total_rows *= q**n_cols
    OA = np.zeros((n_total_rows, n_total_cols), dtype=np.uint8)

    def fill(M: np.ndarray, args):
        if len(args) == 0:
            return

        n_cols, q = args[0]
        if n_cols == 1:
            next_args = args[1:]
        else:
            next_args = [(n_cols - 1, q)] + args[1:]

        n_total_rows, n_total_cols = M.shape
        assert n_total_rows % q == 0
        skip = n_total_rows // q

        left = 0
        right = skip
        for i in range(q):
            M[left:right, 0] = i
            fill(M[left:right, 1:], next_args)
            left += skip
            right += skip

    fill(OA, arities)
    return OA


def generate_oa_from_s_wise_linearly_independent_vectors(
    vecs: np.ndarray, q: int, s: int, batch_size: int = 0, jit_compatible: bool = False
) -> OABatchSequence:
    """
    Takes a matrix of shape say (d, n) such that its columns are s-wise linearly
    independent over F_q, and returns an orthogonal array of strength s and shape
    (q^d, n). q must be a prime number.

    Runtime: q^d * d * n = O(n log n) where n = output-size
    """
    assert galois.is_prime(q)
    d, n = vecs.shape
    assert q >= 2
    return make_batch_sequence(
        jit_compatible=jit_compatible,
        generator_matrix=vecs.astype(np.uint8),
        arities=[(d, q)],
        mod=q,
        levels=q,
        strength=s,
        batch_size=batch_size,
    )


def generate_oa_strength1(levels: int, batch_size: int = 0, jit_compatible: bool = False) -> OABatchSequence:
    return make_batch_sequence(
        jit_compatible=jit_compatible,
        generator_matrix=np.ones((1, levels), dtype=np.uint8),
        arities=((1, levels),),
        mod=levels,
        levels=levels,
        strength=1,
        batch_size=batch_size,
    )


def generate_oa_strength2(m: int, q: int, batch_size: int = 0, jit_compatible: bool = False) -> OABatchSequence:
    """Generates an OA(q^m, k, q, 2) where k = (q^m-1)/(q-1), which is provably optimal
    given the number of columns and the strength.

    N = 1 + (q-1)*k = O(k)

    Construction time: O(output-size)
    """
    assert galois.is_prime(q)
    assert m >= 1
    N = q**m
    k = (N - 1) // (q - 1)
    # the columns will be all m-element vectors over F_q whose first non-zero entry is 1
    two_wise_linearly_independent_vectors = np.zeros((m, k), dtype=np.uint8)

    def fill(mat):
        d = mat.shape[0]
        assert mat.shape[1] == (q**d - 1) // (q - 1)
        mat[0, : q ** (d - 1)] = 1

        if d > 1:
            mat[1:, : q ** (d - 1)] = generate_trivial_oa(d - 1, q).T
            fill(mat[1:, q ** (d - 1) :])

    fill(two_wise_linearly_independent_vectors)
    assert two_wise_linearly_independent_vectors[-1, -1] == 1
    oa_generator = generate_oa_from_s_wise_linearly_independent_vectors(
        two_wise_linearly_independent_vectors, q, 2, jit_compatible=jit_compatible
    )
    assert oa_generator.shape == (N, k)
    return oa_generator


def generate_oa_strength3(
    factors: int, levels: int, batch_size: int = 0, jit_compatible: bool = False
) -> OABatchSequence:
    if levels > 3:
        # AG construction, OA(q^(3m+1), q^(2m), 3, q)
        m = int(np.ceil(0.5 * np.log(factors) / np.log(levels)))
        assert levels ** (2 * m) >= factors and levels ** (2 * (m - 1)) < factors
        return _generate_oa_strength3(m, levels, batch_size, jit_compatible=jit_compatible)

    # strength 3, and 3 levels -> cap set constructions
    # base 3: OA(3^(3m+1),  9^m, 3, 3)
    # base 4: OA(3^(4m+1), 20^m, 3, 3)
    # base 5: OA(3^(5m+1), 45^m, 3, 3)
    m_base3 = int(np.ceil(np.log(factors) / np.log(9)))
    m_base4 = int(np.ceil(np.log(factors) / np.log(20)))
    m_base5 = int(np.ceil(np.log(factors) / np.log(45)))
    if 3 * m_base3 < 4 * m_base4 and 3 * m_base3 < 5 * m_base5:
        # base 3 has smallest number of rows
        return _generate_oa_strength3(m_base3, 3, batch_size=batch_size, jit_compatible=jit_compatible)
    elif 4 * m_base4 < 5 * m_base5:
        # base 4 has smallest number of rows
        return generate_oa_q3_strength3_base4(m_base4, batch_size=batch_size, jit_compatible=jit_compatible)
    # base 5 has smallest number of rows
    return generate_oa_q3_strength3_base5(m_base5, batch_size=batch_size, jit_compatible=jit_compatible)


def _generate_oa_strength3(m: int, q: int, batch_size: int = 0, jit_compatible: bool = False) -> OABatchSequence:
    """Returns an OA(q^(3m+1), q^(2m), 3, q). This has N = k^(3/2), which is the best we
    have for strength 3 arrays *except* if q=2 or q=3."""
    assert galois.is_prime(q)
    assert m >= 1
    cap_set = construct_cap_set(q, m)
    oa_generator = generate_oa_from_generalised_cap_set(cap_set, q, s=3, batch_size=batch_size, jit_compatible=jit_compatible)
    assert oa_generator.runs == q ** (3 * m + 1)
    return oa_generator


def construct_cap_set(q: int, m: int):
    """Given a prime q and integer m, returns a set of q^(2m) vectors in F_q^(3m) that
    are 3-wise affinely independent (i.e. a cap set).

    Construction is based on Example 1.4(1) of the article below, which constructs a cap
    set in PG(4,3) of size q^2+1 that can be chosen such that all but 1 point have 1 in
    the first coordinate; after discarding the exceptional point and deleting the all
    1-column, we get a cap set of size q^2 in AG(4,3).

    ```bibtex
    @article{keefe96,
        title = {Ovoids in PG(3, q): a survey},
        author = {Christine M. O'Keefe},
        journal = {Discrete Mathematics},
        volume = {151},
        number = {1},
        pages = {175-188},
        year = {1996},
        issn = {0012-365X},
        doi = {https://doi.org/10.1016/0012-365X(94)00095-Z}
    }
    ```
    """
    assert galois.is_prime(q)
    assert m >= 1
    f = np.array(galois.primitive_poly(q, degree=2).coefficients(), dtype=int)
    if f[1] == 0:
        # make sure the term for x is non-zero
        f = np.array(
            galois.primitive_poly(q, degree=2, terms=3).coefficients(), dtype=int
        )
    if f[1] != 1:
        gf = galois.GF(q)
        inverse_of_linear_coefficient = gf(1) / gf(f[1])
        f = np.mod(f * inverse_of_linear_coefficient.item(), q)
    assert f[1] == 1

    def g(xy):
        x = xy[:, 0]
        y = xy[:, 1]
        return np.mod(f[0] * x * x + f[1] * x * y + f[2] * y * y, q)

    first_two_columns = np.vstack([oa for oa in generate_trivial_oa(2, q)])
    third_column = g(first_two_columns)
    cap_set = np.hstack((first_two_columns, third_column[:, np.newaxis])).T
    assert cap_set.shape == (3, q * q)
    cap_set = repeat_vectors(cap_set, m)
    assert cap_set.shape == (3 * m, q ** (2 * m))
    return cap_set


def repeat_vectors(vecs: np.ndarray, z: int):
    """Takes a set of k, d-dimensional vectors (as columns), and returns the set of k^z,
    (dz)-dimensional vectors that can be obtained through all possible combinations of
    concatenating z of the k vectors (including repetitions of the same vector)."""
    d, k = vecs.shape
    output = np.zeros((d * z, k**z), dtype=np.uint8)

    def fill(M, j):
        assert M.shape == (d * j, k**j)
        step = k ** (j - 1)
        for n in range(k):
            M[:d, n * step : (n + 1) * step] = np.repeat(
                vecs[:, n : n + 1], step, axis=1
            )
            if j > 1:
                fill(M[d:, n * step : (n + 1) * step], j - 1)

    fill(output, z)
    return output


def generate_oa_from_generalised_cap_set(
    cap_set: np.ndarray, q: int, s: int, batch_size: int = 0, jit_compatible: bool = False
) -> OABatchSequence:
    """Takes a set of k, d-dimensional vectors in F_q (q prime), and returns a matrix
    which, assuming the given vectors are s-wise affinely independent
    (i.e. a generalized cap set), is an OA(q^(d+1), k, s, q)

    Runtime: O(n log n) where n = output-size
    """
    assert galois.is_prime(q)
    d, k = cap_set.shape
    # turn s-wise affinely independent vectors into s-wise linearly independent vectors
    # by adding a coordinate that is equal to 1 for every vector
    s_wise_linearly_independent_vectors = np.vstack(
        (np.ones((1, k), dtype=np.uint8), cap_set)
    )

    OA = generate_oa_from_s_wise_linearly_independent_vectors(
        s_wise_linearly_independent_vectors, q, s, batch_size=batch_size, jit_compatible=jit_compatible
    )
    assert OA.shape == (q ** (d + 1), k)
    return OA


def generate_oa_q3_strength3_base4(m: int = 1, batch_size: int = 0, jit_compatible: bool = False) -> OABatchSequence:
    """Constructs an OA(3^(4m+1), 20^m, 3, 3), which asymptotically has N = k^(1.466)

    Runtime: O(n log n) where n = output size.

    Based on a cap set of size 20 in PG(4,3) (see Fiure 1 in the reference below) whose
    representatives can be chosen such that all of them have a non-zero last entry;
    Chosing it to be 1 and then deleting it gives a cap set of size 20 in AG(4,3).

    ```bibtex
    @incollection{hill83,
        title = {On Pellegrino's 20-Caps in S_(4,3)},
        author = {R. Hill},
        series = {North-Holland Mathematics Studies},
        publisher = {North-Holland},
        volume = {78},
        pages = {433-447},
        year = {1983},
        booktitle = {Combinatorics '81 in honour of Beniamino Segre},
        doi = {https://doi.org/10.1016/S0304-0208(08)73322-X}
    }
    ```
    """
    vecs = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 1],
            [2, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 2, 1, 1],
            [1, 2, 1, 2],
            [2, 1, 1, 2],
            [1, 1, 1, 2],
            [2, 2, 1, 2],
        ],
        dtype=np.uint8,
    ).T
    # this is a cap set of size 20 in AG(4,3) (the largest possible)
    cap_set = np.hstack((vecs, np.mod(2 * vecs, 3)))
    assert cap_set.shape == (4, 20)
    # this is a cap set of size 20^m in AG(4m, 3)
    cap_set = repeat_vectors(cap_set, m)
    assert cap_set.shape == (4 * m, 20**m)
    OA = generate_oa_from_generalised_cap_set(cap_set, q=3, s=3, batch_size=batch_size, jit_compatible=jit_compatible)
    assert len(OA) == 3 ** (4 * m + 1)
    return OA


def generate_oa_q3_strength3_base5(m: int = 1, batch_size: int = 0, jit_compatible: bool = False) -> OABatchSequence:
    """Returns an OA(3^(5m+1), 45^m, 3, 3), which asymptotically has N = k^(1.443).

    Runtime: O(n log n) where n = output-size

    Cap set of size 56 in PG(5,3) is taken from section 2 of

    ```bibtex
    @article{hill73,
        author = {Hill, Raymond},
        title = {On the largest size of cap in S53},
        journal = {Rendiconti del Seminario Matematico della Università di Padova},
        volume = {54},
        pages = {378--380},
        year = {1973},
        url = {http://www.bdim.eu/item?id=RLINA_1973_8_54_3_378_0}
    }
    ```

    The cap set of size 45 in AG(5,3) constructed from it is maximal, see Theorem 1.2 in

    ```bibtex
    @article{edel2002,
        title = {The Classification of the Largest Caps in AG(5, 3)},
        author = {Y. Edel and S. Ferret and I. Landjev and L. Storme},
        journal = {Journal of Combinatorial Theory, Series A},
        volume = {99},
        number = {1},
        pages = {95-110},
        year = {2002},
        doi = {https://doi.org/10.1006/jcta.2002.3261}
    }
    ```
    """
    # cap of size 56 in PG(5,3):
    K = np.array(
        [
            [2, 1, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 0],
            [0, 0, 2, 1, 0, 0],
            [0, 0, 0, 2, 1, 0],
            [0, 0, 0, 0, 2, 1],
            [2, 2, 2, 2, 2, 1],
            [2, 1, 1, 1, 1, 1],
            [2, 0, 1, 0, 1, 0],
            [0, 2, 0, 1, 0, 1],
            [2, 2, 1, 2, 0, 2],
            [1, 0, 0, 2, 0, 1],
            [2, 0, 2, 2, 1, 2],
            [1, 0, 1, 0, 0, 2],
            [1, 2, 1, 2, 1, 1],
            [1, 1, 2, 0, 0, 0],
            [0, 1, 1, 2, 0, 0],
            [0, 0, 1, 1, 2, 0],
            [0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 2, 2, 2, 2, 0],
            [0, 1, 2, 2, 2, 2],
            [1, 1, 0, 2, 0, 1],
            [2, 0, 0, 2, 1, 2],
            [1, 0, 1, 1, 0, 2],
            [1, 2, 1, 2, 2, 1],
            [2, 0, 1, 0, 1, 1],
            [2, 1, 2, 0, 2, 0],
            [0, 2, 1, 2, 0, 2],
            [1, 1, 0, 2, 0, 2],
            [1, 2, 2, 1, 0, 1],
            [2, 0, 1, 1, 0, 2],
            [1, 0, 1, 2, 2, 1],
            [2, 0, 2, 0, 1, 1],
            [2, 1, 2, 1, 2, 0],
            [0, 2, 1, 2, 1, 2],
            [1, 1, 1, 2, 0, 0],
            [0, 1, 1, 1, 2, 0],
            [0, 0, 1, 1, 1, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 2, 2, 2, 0, 0],
            [0, 1, 2, 2, 2, 0],
            [0, 0, 1, 2, 2, 2],
            [1, 1, 2, 2, 0, 0],
            [0, 1, 1, 2, 2, 0],
            [0, 0, 1, 1, 2, 2],
            [1, 1, 1, 2, 2, 0],
            [0, 1, 1, 1, 2, 2],
            [1, 1, 2, 2, 2, 0],
            [0, 1, 1, 2, 2, 2],
            [2, 1, 1, 0, 1, 2],
            [1, 0, 2, 2, 1, 2],
            [1, 2, 1, 0, 0, 2],
            [1, 2, 0, 2, 1, 1],
            [2, 0, 1, 2, 1, 0],
            [0, 2, 0, 1, 2, 1],
            [2, 2, 1, 2, 0, 1],
        ],
        dtype=np.uint8,
    )
    nonzeros_by_column = np.sum((K != 0).astype(np.uint8), axis=0)
    # one of the columns has exactly 45 non-zero elements; if we discard zero-rows, and
    # normalise the remaining entries to 1 in this column (we are in projective geometry
    # so it remains a cap-set), then we get a cap set of size 45 in AG(5,3), which is
    # the largest possible.
    normalisation_index = -1
    for i in range(6):
        if nonzeros_by_column[i] == 45:
            normalisation_index = i
            break
    assert normalisation_index != -1
    # discard entries with zero entry in normalisation column
    keep_rows = K[:, normalisation_index] != 0
    K = K[keep_rows]
    assert K.shape == (45, 6)
    # both 1 and 2 are their own inverse in F_3 -> normalising to one is same as squaring
    K = np.mod(K * np.expand_dims(K[:, normalisation_index], axis=1), 3)
    assert np.all(K[:, normalisation_index] == 1)
    # cap set of size 45 in AG(5,3) (best possible)
    cap_set = np.delete(K, normalisation_index, axis=1).T
    assert cap_set.shape == (5, 45)
    # cap set of size 45^m in AG(5m,3)
    cap_set = repeat_vectors(cap_set, m)
    OA = generate_oa_from_generalised_cap_set(cap_set, q=3, s=3, batch_size=batch_size, jit_compatible=jit_compatible)
    assert len(OA) == 3 ** (5 * m + 1)
    return OA


def generate_oa_q3_strength4(m: int, batch_size: int = 0, jit_compatible: bool = False) -> OABatchSequence:
    """Constructs an OA(3^(2m), 3^m, 3, 4) (i.e. ternary of strength 4), which has N = k^2

    For the construction, see section 3.1 in

    ```bibtex
    @article{Huang_2019,
        title={Sidon sets and 2-caps in F3n},
        author={Huang, Yixuan and Tait, Michael and Won, Robert},
        volume={12},
        ISSN={1944-4176},
        DOI={10.2140/involve.2019.12.995},
        number={6},
        journal={Involve, a Journal of Mathematics},
        publisher={Mathematical Sciences Publishers},
        year={2019},
        pages={995--1003}
    }
    ```"""
    # generate a 4-wise affinely independent set of size 3^n in AG(2n, 3)
    gf = galois.GF(3**m)
    N = 3**m
    cap_set = np.zeros((N, 2 * m), dtype=np.uint8)
    for x in range(3**m):
        x_gf = gf(x)
        cap_set[x, :m] = x_gf.vector()
        cap_set[x, m:] = (x_gf * x_gf).vector()
    OA = generate_oa_from_generalised_cap_set(cap_set.T, q=3, s=4, batch_size=batch_size, jit_compatible=jit_compatible)
    return OA