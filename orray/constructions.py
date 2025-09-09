import itertools
import operator

import galois
import jax.numpy as jnp
from jaxtyping import Int8, Array

from orray.oa import LinearOrthogonalArray


def generate_oa_bose_ray(
    m: int, strength: int
):
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
    assert strength >= 2
    t = strength // 2
    # (mt) x max(mt,2^m-1), matrix with 2t-wise linearly independent columns
    vecs = generate_2t_wise_linearly_independent_vectors(m, t)
    return LinearOrthogonalArray(
        generator_matrix=vecs,
        arities=[(m * t, 2)],
        mod=2,
        num_levels=2,
        strength=strength,
        binary_oa_even_to_odd_strength=(strength % 2 == 1),
    )


def generate_2t_wise_linearly_independent_vectors(
    m: int, t: int
) -> Int8[Array, "m*t ..."]:
    """Returns a set of 2^m-1 vectors in F_2^{mt} such that any set of 2t of
    them are linearly independent (i.e. a 2t-Sidon set in F_2^{mt}). Returns the result
    as a 0-1 matrix of shape (mt) x 2^m-1."""
    galois_field = galois.GF(2**m)
    assert m >= 1
    assert t >= 1

    def get_mt_vector(_m: ...) -> Int8[Array, " mt"]:
        """Given a vector alpha in F_2^m, returns the vector (alpha, alpha^3, ...,
        alpha^(2t-1)) in F_2^(mt), where multiplication in F_2^m is defined through the
        identification of F_2^m with the Galois group GF(2^m)."""
        repeated_m_vector = itertools.chain([_m], itertools.repeat(_m**2, t - 1))
        mt_sequence = itertools.accumulate(repeated_m_vector, operator.mul)
        return jnp.concatenate([mt.vector() for mt in mt_sequence])

    mt_vector_generator = (get_mt_vector(m) for m in galois_field.units)
    return jnp.asarray(jnp.column_stack(list(mt_vector_generator)), dtype=jnp.int8)
