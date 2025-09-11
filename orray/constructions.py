import itertools
import operator
from typing import Optional

import galois
import jax
import jax.numpy as jnp
from jaxtyping import Array, UInt8

from orray.oa import LinearOrthogonalArray, OrthogonalArray

##################################################
#          "Master" Construction Method          #
##################################################


def construct_oa(
    num_cols: int,
    num_levels: int,
    strength: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
    verbose: bool = False,
) -> OrthogonalArray:
    """constructs an orthogonal array with the given strength and number of levels and
    columns, with (best-effort) minimal number of rows.

    if `rng` is not `None` then all columns have a uniform random integer in
    {0,...,num_levels-1} added to them (modulo num_levels), which preserves orthogonality
    but ensures that each row is (marginally) distributed as a random sequence of independent
    uniform {0,...,num_levels-1} variables.
    """
    assert strength >= 1
    assert num_cols >= 1
    assert num_levels >= 2

    if strength == 1:
        oa = construct_oa_strength1(num_levels, device=device, rng=rng)
    elif num_cols <= strength:
        oa = construct_trivial_oa(num_cols, num_levels, device=device, rng=rng)
    # now since strength < num_cols, the final array must have at least num_num_levels^strength rows
    elif num_cols <= num_levels:
        assert galois.is_prime(num_levels)
        # has exactly num_levels^strength rows, so is now optimal
        oa = construct_oa_vandermonde(num_levels, 1, strength, device=device, rng=rng)
    elif strength == 2:
        q = num_levels
        assert galois.is_prime(q)
        # need smallest m with (q^m-1)/(q-1) >= num_cols. since LHS > q^(m-1), a lower
        # bound on m is obtained by choosing the smallest m such that q^(m-1) >= num_cols
        m = 1 + int(jnp.ceil(jnp.log(num_cols) / jnp.log(q)))
        assert (q**m - 1) // (q - 1) <= num_cols
        while (q**m - 1) // (q - 1) < num_cols:
            m += 1
        oa = construct_oa_strength2(m, q, device=device, rng=rng)
    # now num_cols > max(strength, num_num_levels), strength >= 3
    elif num_levels == 2:
        oa = construct_binary_oa(
            num_cols, strength, device=device, rng=rng, verbose=verbose
        )
    elif strength == 3:
        oa = construct_oa_strength3(num_cols, num_levels, device=device, rng=rng)
    elif strength == 4 and num_levels == 3:
        # cap set construction OA(3^(2m), 3^m, 3, 4)
        m = int(jnp.ceil(jnp.log(num_cols) / jnp.log(3)))
        oa = construct_oa_q3_strength4(m, device=device, rng=rng)
    else:
        # strength >= 4, num_levels >= 3 and at least one of them ">". Only remaining option
        # is vandermonde construction OA(q^(ms), q^m, q, s)
        m = int(jnp.ceil(jnp.log(num_cols) / jnp.log(num_levels)))
        assert num_levels**m >= num_cols and num_levels ** (m - 1) < num_cols
        oa = construct_oa_vandermonde(num_levels, m, strength, device=device, rng=rng)
    num_rows = len(oa)

    # check if trivial array is better (for example if num_cols=4 and strength=5 binary array)
    if num_levels**num_cols <= num_rows:
        oa = construct_trivial_oa(num_cols, num_levels, device=device, rng=rng)
    assert oa.num_cols >= num_cols, (
        f"bug in `construct_oa` method, generated oa has {oa.num_cols} num_cols (columns), but {num_cols} were requested"
    )
    # TODO truncate to requested number of columns
    # TODO test this method rigorously
    return oa


##################################################
#            Individual Constructions            #
##################################################


def construct_binary_oa(
    num_cols: int,
    strength: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
    verbose: bool = False,
):
    # smallest m such that 2^m >= num_cols
    m = int(jnp.ceil(jnp.log(num_cols) / jnp.log(2)))
    assert 2**m >= num_cols
    assert m == 1 or 2 ** (m - 1) < num_cols

    m_is_even = m % 2 == 0
    match strength:
        # Kerdock: OA(2^(2m), 2^m, 2, 5), m>=4 even
        case 5 if m >= 4 and m_is_even:
            if verbose:
                print("OA(Selection): Kerdock")
            return construct_oa_kerdock(m, device=device, rng=rng)
        # Delsarte-Goethals: OA(2^(3m-1), 2^m, 2, 7), m>=4 even
        # Strength 6 is an exceptional case where the strength 7 DG construction is more
        # efficient (has fewer rows) than the 'weaker' strength 6 BR construction,
        # respectively 2^(3m-1) vs 2^(3m) rows!
        case 6 | 7 if m >= 4 and m_is_even:
            if verbose:
                print("OA(Selection): Delsarte-Goethals")
            return construct_oa_delsarte_goethals(m, device=device, rng=rng)
        # Bose: OA(2^(2m+1), 2^m, 2, 5)
        # Bose: OA(2^(3m+1), 2^m, 2, 7)
        case _:
            # if strength is even, this has 2^m-1 columns, if strength is odd 2^m >= num_cols
            if strength % 2 == 0 and 2**m - 1 < num_cols:
                m += 1
                assert 2**m - 1 >= num_cols
            if verbose:
                print("OA(Selection): Bose-Ray")
            return construct_oa_bose_ray(m, strength, device=device, rng=rng)


def construct_oa_bose_ray(
    m: int,
    strength: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
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
    vecs = _generate_2t_wise_linearly_independent_vectors(m, t, device=device)
    assert device is None or vecs.device == device
    oa = LinearOrthogonalArray(
        generator_matrix=vecs,
        arities=[(m * t, 2)],
        mod=2,
        num_levels=2,
        strength=strength,
        binary_oa_even_to_odd_strength=(strength % 2 == 1),
        device=device,
        rng=rng,
    )
    if strength % 2 == 1:
        assert oa.shape == (2 ** (m * t + 1), 2**m)
    else:
        assert oa.shape == (2 ** (m * t), 2**m - 1)
    return oa


def construct_oa_delsarte_goethals(
    m: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
):
    """Generates an OA(2^(3m-1), 2^m, 2, 7), i.e. a binary 8^m/2 x 2^m array of
    strength 7, where m >= 4 is an even integer.  Based on the linear construction of
    the Delsarte-Goethals code (reference below).

    When defined, it satisfies N = .5 k^3, which is a quarter as many columns given k
    than the best linear array (bose-ray), which has N = 2k^3

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
    xi_table = _calculate_xi_table(m, 3 * (n - 1), device=device)
    # the generator of the DG code is made up of three vertically stacked blocks of
    # respective lengths 1, m, m
    first_block = jnp.ones((1, n + 1), dtype=jnp.uint8, device=device)
    second_block = xi_table[: n + 1, :].T
    powers = list(range(3, 3 * n, 3))  # e.g. third power of xi has index 4 in xi_table
    indices = jnp.asarray(
        [0, 1] + [1 + i for i in powers], dtype=jnp.uint8, device=device
    )
    third_block = 2 * xi_table[indices, :].T
    # (2m+1) x 2^m, the generator of the Delsarte-Goethals code
    G = jnp.vstack((first_block, second_block, third_block))
    assert device is None or G.device == device
    oa = LinearOrthogonalArray(
        generator_matrix=G,
        arities=[(m + 1, 4), (m, 2)],
        mod=4,
        num_levels=2,
        strength=7,
        post_linear_combination_processor=_gray_map,
        device=device,
        rng=rng,
    )
    assert oa.shape == (2 ** (3 * (m + 1) - 1), 2 ** (m + 1))
    return oa


def construct_oa_kerdock(
    m: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
):
    """Generates an OA(4^m, 2^m, 2, 5), i.e. a (non-linear) binary 4^m x 2^m array of
    strength 5, where m >= 4 is an even integer. Based on the linear construction of the
    Kerdock code (reference below).

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
    xi_table = _calculate_xi_table(m, device=device)
    ones_column = jnp.ones((2**m, 1), dtype=jnp.uint8, device=device)
    # (m+1) x 2^m, generator of the kerdock code
    G = jnp.hstack((ones_column, xi_table)).T
    assert device is None or G.device == device
    oa = LinearOrthogonalArray(
        generator_matrix=G,
        arities=[(m + 1, 4)],
        mod=4,
        num_levels=2,
        strength=5,
        post_linear_combination_processor=_gray_map,
        device=device,
        rng=rng,
    )
    assert oa.shape == (4 ** (m + 1), 2 ** (m + 1))
    return oa


def construct_oa_from_s_wise_linearly_independent_vectors(
    vecs: UInt8[Array, "d n"],
    q: int,
    s: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
    """
    Takes a matrix of shape say (d, n) such that its columns are s-wise linearly
    independent over F_q, and returns an orthogonal array of strength s and shape
    (q^d, n). q must be a prime number.

    Runtime: q^d * d * n = O(n log n) where n = output-size
    """
    assert galois.is_prime(q)
    d, n = vecs.shape
    assert q >= 2
    oa = LinearOrthogonalArray(
        generator_matrix=jax.device_put(vecs.astype(jnp.uint8), device=device),
        arities=[(d, q)],
        mod=q,
        num_levels=q,
        strength=s,
        device=device,
        rng=rng,
    )
    assert oa.shape == (q**d, n)
    return oa


def construct_oa_vandermonde(
    q: int,
    m: int,
    strength: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
):
    """Returns an OA(q^(m*strength), q^m, q, strength) for a prime number q

    Runtime: O(n log n) where n = output-size"""
    assert q >= 2
    assert m >= 1
    assert strength >= 1
    assert galois.is_prime(q)
    # construct the vandermonde matrix whose columns are [1 x x^2 ... x^(s-1)] and x
    # loops through all elements of F_q (including zero)
    galois_field = galois.GF(q**m)

    def get_vector_from_galois_element(_x: ...):
        r"""
        Given an element x of GF(q^m), calculates (1,x,x^2,...,x^(s-1)) \in GF(q^m)^t,
        and returns it as a vector in F_q^(m*t)
        """
        repeated_x = itertools.chain(
            [galois_field(1)], itertools.repeat(_x, strength - 1)
        )
        galois_vector = itertools.accumulate(repeated_x, operator.mul)
        return jax.device_put(
            jnp.concatenate([y.vector() for y in galois_vector], dtype=jnp.uint8),
            device=device,
        )

    columns = [
        jnp.asarray([1] + (m * strength - 1) * [0], dtype=jnp.uint8, device=device)
    ] + [get_vector_from_galois_element(x) for x in galois_field.units]
    # now M is a (m*s, q^m) matrix over F_q whose columns are s-wise linearly
    # independent over F_q
    M = jnp.asarray(jnp.column_stack(columns), dtype=jnp.uint8, device=device)
    assert device is None or M.device == device
    oa = LinearOrthogonalArray(
        generator_matrix=M,
        arities=[(m * strength, q)],
        mod=q,
        num_levels=q,
        strength=strength,
        device=device,
        rng=rng,
    )
    assert oa.shape == (q ** (m * strength), q**m)
    return oa


def construct_oa_strength1(
    num_levels: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
    """Returns an orthogonal array of shape (num_levels, num_levels) of the form:

    [   0 ...   0 ]
    [   1 ...   1 ]
    [ l-1 ... l-1 ]

    where l = num_levels
    """
    oa = LinearOrthogonalArray(
        generator_matrix=jnp.ones((1, num_levels), dtype=jnp.uint8, device=device),
        arities=[(1, num_levels)],
        mod=num_levels,
        num_levels=num_levels,
        strength=1,
        device=device,
        rng=rng,
    )
    assert oa.shape == (num_levels, num_levels)
    return oa


def construct_oa_strength2(
    m: int,
    q: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
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
    two_wise_linearly_independent_vectors = jnp.zeros(
        (m, k), dtype=jnp.uint8, device=device
    )

    def fill(mat):
        d = mat.shape[0]
        assert mat.shape[1] == (q**d - 1) // (q - 1)
        mat = mat.at[0, : q ** (d - 1)].set(1)

        if d > 1:
            submatrix = construct_trivial_oa(d - 1, q, device=device).materialize().T
            mat = mat.at[1:, : q ** (d - 1)].set(submatrix)

            mat = mat.at[1:, q ** (d - 1) :].set(fill(mat[1:, q ** (d - 1) :]))
        return mat

    two_wise_linearly_independent_vectors = fill(two_wise_linearly_independent_vectors)
    assert two_wise_linearly_independent_vectors[-1, -1] == 1
    assert device is None or two_wise_linearly_independent_vectors.device == device
    oa = construct_oa_from_s_wise_linearly_independent_vectors(
        two_wise_linearly_independent_vectors,
        q,
        2,
        device=device,
        rng=rng,
    )
    assert oa.shape == (N, k), f"expected shape {(N, k)} but got {oa.shape}"
    return oa


def construct_oa_strength3(
    num_cols: int,
    num_levels: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
    if num_levels > 3:
        # AG construction, OA(q^(3m+1), q^(2m), 3, q)
        m = int(jnp.ceil(0.5 * jnp.log(num_cols) / jnp.log(num_levels)))
        assert (
            num_levels ** (2 * m) >= num_cols and num_levels ** (2 * (m - 1)) < num_cols
        )
        return construct_oa_strength3_base3(m, num_levels, device=device, rng=rng)

    # strength 3, and 3 num_levels -> cap set constructions
    # base 3: OA(3^(3m+1),  9^m, 3, 3)
    # base 4: OA(3^(4m+1), 20^m, 3, 3)
    # base 5: OA(3^(5m+1), 45^m, 3, 3)
    m_base3 = int(jnp.ceil(jnp.log(num_cols) / jnp.log(9)))
    m_base4 = int(jnp.ceil(jnp.log(num_cols) / jnp.log(20)))
    m_base5 = int(jnp.ceil(jnp.log(num_cols) / jnp.log(45)))
    if 3 * m_base3 < 4 * m_base4 and 3 * m_base3 < 5 * m_base5:
        # base 3 has smallest number of rows
        return construct_oa_strength3_base3(m_base3, 3, device=device, rng=rng)
    elif 4 * m_base4 < 5 * m_base5:
        # base 4 has smallest number of rows
        return construct_oa_q3_strength3_base4(m_base4, device=device, rng=rng)
    # base 5 has smallest number of rows
    return construct_oa_q3_strength3_base5(m_base5, device=device, rng=rng)


def construct_oa_strength3_base3(
    m: int,
    q: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
    """Returns an OA(q^(3m+1), q^(2m), 3, q). This has N = k^(3/2), which is the best we
    have for strength 3 arrays *except* if q=2 or q=3."""
    assert galois.is_prime(q)
    assert m >= 1
    cap_set = _construct_cap_set(q, m, device=device)
    assert device is None or cap_set.device == device
    oa = construct_oa_from_generalised_cap_set(cap_set, q, s=3, device=device, rng=rng)
    # assert oa.runs == q ** (3 * m + 1)
    assert oa.shape == (q ** (3 * m + 1), q ** (2 * m))
    return oa


def construct_oa_q3_strength3_base4(
    m: int = 1,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
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
    vecs = jnp.array(
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
        dtype=jnp.uint8,
        device=device,
    ).T
    # this is a cap set of size 20 in AG(4,3) (the largest possible)
    cap_set = jnp.hstack((vecs, jnp.mod(2 * vecs, 3)))
    assert cap_set.shape == (4, 20)
    assert device is None or cap_set.device == device
    # this is a cap set of size 20^m in AG(4m, 3)
    cap_set = repeat_vectors(cap_set, m)
    assert cap_set.shape == (4 * m, 20**m)
    assert device is None or cap_set.device == device
    oa = construct_oa_from_generalised_cap_set(
        cap_set, q=3, s=3, device=device, rng=rng
    )
    assert oa.num_rows == 3 ** (4 * m + 1)
    assert oa.shape == (3 ** (4 * m + 1), 20**m)
    return oa


def construct_oa_q3_strength3_base5(
    m: int = 1,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
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
    K = jnp.array(
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
        dtype=jnp.uint8,
        device=device,
    )
    nonzeros_by_column = jnp.sum((K != 0).astype(jnp.uint8), axis=0)
    assert device is None or nonzeros_by_column.device == device
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
    K = jnp.mod(K * jnp.expand_dims(K[:, normalisation_index], axis=1), 3)
    assert jnp.all(K[:, normalisation_index] == 1)
    # cap set of size 45 in AG(5,3) (best possible)
    cap_set = jnp.delete(K, normalisation_index, axis=1).T
    assert cap_set.shape == (5, 45)
    # cap set of size 45^m in AG(5m,3)
    cap_set = repeat_vectors(cap_set, m)
    assert device is None or cap_set.device == device
    oa = construct_oa_from_generalised_cap_set(
        cap_set, q=3, s=3, device=device, rng=rng
    )
    assert oa.num_rows == 3 ** (5 * m + 1)
    assert oa.shape == (3 ** (5 * m + 1), 45**m)
    return oa


def construct_trivial_oa(
    n_cols: int,
    q: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
    """returns the trivial OA with q^n_cols rows"""
    return LinearOrthogonalArray(
        generator_matrix=jnp.eye(n_cols, dtype=jnp.uint8, device=device),
        arities=[(n_cols, q)],
        mod=q,
        num_levels=q,
        strength=n_cols,
        device=device,
        rng=rng,
    )


def construct_oa_from_generalised_cap_set(
    cap_set: UInt8[Array, "d k"],
    q: int,
    s: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
    """Takes a set of k, d-dimensional vectors in F_q (q prime), and returns a matrix
    which, assuming the given vectors are s-wise affinely independent
    (i.e. a generalized cap set), is an OA(q^(d+1), k, s, q)

    Runtime: O(n log n) where n = output-size
    """
    assert galois.is_prime(q)
    d, k = cap_set.shape
    cap_set = jax.device_put(cap_set, device=device)
    # turn s-wise affinely independent vectors into s-wise linearly independent vectors
    # by adding a coordinate that is equal to 1 for every vector
    s_wise_linearly_independent_vectors = jnp.vstack(
        (jnp.ones((1, k), dtype=jnp.uint8, device=device), cap_set)
    )

    oa = construct_oa_from_s_wise_linearly_independent_vectors(
        s_wise_linearly_independent_vectors,
        q,
        s,
        device=device,
        rng=rng,
    )
    assert oa.shape == (q ** (d + 1), k)
    return oa


def construct_oa_q3_strength4(
    m: int,
    device: jax.Device | None = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> LinearOrthogonalArray:
    """Constructs an OA(3^(2m+1), 3^m, 3, 4) (i.e. ternary of strength 4), which has N = 3k^2

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
    cap_set = jnp.zeros((N, 2 * m), dtype=jnp.uint8, device=device)
    for x in range(3**m):
        x_gf = gf(x)
        cap_set = cap_set.at[x, :m].set(
            jnp.asarray(x_gf.vector(), dtype=jnp.uint8, device=device)
        )
        cap_set = cap_set.at[x, m:].set(
            jnp.asarray((x_gf * x_gf).vector(), dtype=jnp.uint8, device=device)
        )
    oa = construct_oa_from_generalised_cap_set(
        cap_set.T, q=3, s=4, device=device, rng=rng
    )
    assert oa.shape == (3 ** (2 * m + 1), 3**m)
    return oa


###################################################
#                     Helpers                     #
###################################################


def _generate_2t_wise_linearly_independent_vectors(
    m: int, t: int, device: jax.Device | None = None
) -> UInt8[Array, "m*t ..."]:
    """Returns a set of 2^m-1 vectors in F_2^{mt} such that any set of 2t of
    them are linearly independent (i.e. a 2t-Sidon set in F_2^{mt}). Returns the result
    as a 0-1 matrix of shape (mt) x 2^m-1."""
    galois_field = galois.GF(2**m)
    assert m >= 1
    assert t >= 1

    def get_mt_vector(_m: ...) -> UInt8[Array, " mt"]:
        """Given a vector alpha in F_2^m, returns the vector (alpha, alpha^3, ...,
        alpha^(2t-1)) in F_2^(mt), where multiplication in F_2^m is defined through the
        identification of F_2^m with the Galois group GF(2^m)."""
        repeated_m_vector = itertools.chain([_m], itertools.repeat(_m**2, t - 1))
        mt_sequence = itertools.accumulate(repeated_m_vector, operator.mul)
        return jnp.concatenate(
            [
                jnp.asarray(mt.vector(), dtype=jnp.uint8, device=device)
                for mt in mt_sequence
            ]
        )

    mt_vector_generator = (get_mt_vector(m) for m in galois_field.units)
    return jnp.asarray(
        jnp.column_stack(list(mt_vector_generator)), dtype=jnp.uint8, device=device
    )


def _calculate_xi_table(
    m: int,
    max_power_of_xi: int = -1,
    include_zero_row: bool = True,
    device: jax.Device | None = None,
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

    h = _get_h_polynomial(m, device=device)
    feedback = jnp.mod(
        -h[1:][::-1], 4
    )  # delete leading one, reverse order, take minus modulo 4
    assert feedback[0] == 1
    offset = int(include_zero_row)
    table = jnp.zeros((offset + 1 + max_power_of_xi, m), dtype=jnp.uint8, device=device)

    # === jax version of: ===
    # table[offset, 0] = 1
    # for i in range(offset + 1, offset + 1 + max_power_of_xi):
    #     table[i, 1:] = table[i - 1, :-1]  # shift right
    #     table[i, :] = np.mod(table[i, :] + table[i - 1, -1] * feedback, 4)

    # row 1
    table = table.at[offset, 0].set(1)

    def body_fun(i, table):
        prev_row = table[i - 1]
        new_row = jnp.zeros(m, dtype=jnp.uint8, device=device)
        new_row = new_row.at[1:].set(prev_row[:-1])  # shift right
        new_row = jnp.mod(new_row + prev_row[-1] * feedback, 4)
        return table.at[i].set(new_row)

    table = jax.lax.fori_loop(offset + 1, offset + 1 + max_power_of_xi, body_fun, table)
    assert device is None or table.device == device
    return table


def _gray_map(
    orthogonal_array: UInt8[Array, "runs factors"],
) -> UInt8[Array, "runs factors"]:
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


def _get_h_polynomial(m: int, device: jax.Device | None = None):
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
    h = _get_h_polynomial_from_primitive_F2(primitive.coefficients(), device=device)
    return h


def _get_h_polynomial_from_primitive_F2(
    primitive: UInt8[Array, " n"], device: jax.Device | None = None
):
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
    primitive = jnp.asarray(primitive, dtype=jnp.uint8, device=device)
    assert jnp.all(primitive >= 0) and jnp.all(primitive <= 1)
    assert len(primitive.shape) == 1
    m = len(primitive) - 1
    assert m % 2 == 1
    # split primitive in even and odd coefficients
    e = primitive.at[::2].set(0)
    d = (-primitive).at[1::2].set(0)
    # square e and d
    e_sq = jnp.polymul(e, e)
    d_sq = jnp.polymul(d, d)
    assert len(d_sq) == 2 * m + 1
    e_sq = jnp.pad(e_sq, (2 * m + 1 - len(e_sq), 0), constant_values=0)
    # now h(x^2) = (+ or -) e(x)^2 - d(x)^2
    h_of_x_sq = jnp.mod(e_sq - d_sq, 4).astype(jnp.uint8)
    assert jnp.all(h_of_x_sq[1::2] == 0)
    h = h_of_x_sq[::2]
    if h[0] == 3:  # i.e. h[0] = -1 in Z_4
        h = jnp.mod(-h, 4)
    assert h[0] == 1
    assert h[-1] == 3
    assert device is None or h.device == device
    return h


def _construct_cap_set(q: int, m: int, device: jax.Device | None = None):
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
    f = jnp.asarray(
        galois.primitive_poly(q, degree=2).coefficients(),
        dtype=jnp.uint8,
        device=device,
    )
    if f[1] == 0:
        # make sure the term for x is non-zero
        f = jnp.asarray(
            galois.primitive_poly(q, degree=2, terms=3).coefficients(),
            dtype=jnp.uint8,
            device=device,
        )
    if f[1] != 1:
        gf = galois.GF(q)
        inverse_of_linear_coefficient = gf(1) / gf(f[1])
        f = jnp.mod(f * inverse_of_linear_coefficient.item(), q)
    assert f[1] == 1

    def g(xy):
        x = xy[:, 0]
        y = xy[:, 1]
        return jnp.mod(f[0] * x * x + f[1] * x * y + f[2] * y * y, q)

    first_two_columns = construct_trivial_oa(n_cols=2, q=q, device=device).materialize()
    # jnp.vstack([oa for oa in construct_trivial_oa(num_cols=2, q=q)])
    third_column = g(first_two_columns)
    cap_set = jnp.hstack((first_two_columns, third_column[:, None])).T
    assert cap_set.shape == (3, q * q)
    cap_set = repeat_vectors(cap_set, m)
    assert device is None or cap_set.device == device
    assert cap_set.shape == (3 * m, q ** (2 * m))
    return cap_set


def repeat_vectors(vecs: UInt8[Array, "d k"], z: int):
    """Takes a set of k, d-dimensional vectors (as columns), and returns the set of k^z,
    (dz)-dimensional vectors that can be obtained through all possible combinations of
    concatenating z of the k vectors (including repetitions of the same vector)."""
    d, k = vecs.shape
    device = vecs.device
    output = jnp.zeros((d * z, k**z), dtype=jnp.uint8, device=device)

    def fill(M, j):
        assert M.shape == (d * j, k**j)
        step = k ** (j - 1)
        for n in range(k):
            M = M.at[:d, n * step : (n + 1) * step].set(
                jnp.repeat(vecs[:, n : n + 1], step, axis=1)
            )
            if j > 1:
                M = M.at[d:, n * step : (n + 1) * step].set(
                    fill(M[d:, n * step : (n + 1) * step], j - 1)
                )
        return M

    output = fill(output, z)
    return output
