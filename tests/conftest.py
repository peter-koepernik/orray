import typing

import jax
import pytest

import orray.constructions as constructions

typing.TESTING = True  # pyright: ignore


jax.config.update("jax_numpy_dtype_promotion", "strict")
jax.config.update("jax_numpy_rank_promotion", "raise")

device = jax.devices()[0]

orthogonal_arrays = {
    "bose-ray(m=4, s=5)": constructions.construct_oa_bose_ray(m=4, strength=5, device=device),
    "bose-ray(m=5, s=4)": constructions.construct_oa_bose_ray(m=5, strength=4, device=device),
    "vandermonde(q=3, m=2, s=3)": constructions.construct_oa_vandermonde(q=3, m=2, strength=3, device=device),
    "delsarge-geothals(m=4)": constructions.construct_oa_delsarte_goethals(m=4, device=device),
    "kerdock(m=6)": constructions.construct_oa_kerdock(m=6, device=device),
    "strength1(l=5)": constructions.construct_oa_strength1(num_levels=5, device=device),
    "strength2(m=3, l=5)": constructions.construct_oa_strength2(m=3, q=5, device=device),
    "strength3_base3(m=2, l=3)": constructions.construct_oa_strength3_base3(m=2, q=3, device=device),
    "strength3_q3_base4(m=2)": constructions.construct_oa_q3_strength3_base4(m=2, device=device),
    "strength3_q3_base5(m=1)": constructions.construct_oa_q3_strength3_base5(m=1, device=device),
    "strength4_q3(m=3)": constructions.construct_oa_q3_strength4(m=3, device=device),
    "trivial(cols=3, l=3)": constructions.construct_trivial_oa(n_cols=5, q=3, device=device),
}


@pytest.fixture(params=orthogonal_arrays.values(), ids=orthogonal_arrays.keys())
def oa(request):
    return request.param
