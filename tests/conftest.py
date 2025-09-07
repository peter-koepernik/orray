import typing

import jax


typing.TESTING = True  # pyright: ignore


jax.config.update("jax_numpy_dtype_promotion", "strict")
jax.config.update("jax_numpy_rank_promotion", "raise")