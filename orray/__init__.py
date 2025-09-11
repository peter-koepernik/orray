import jax

from orray.constructions import construct_oa
from orray.oa import OrthogonalArray

# TODO include individual constructions


jax.config.update("jax_enable_x64", True)

__all__ = ["OrthogonalArray", "construct_oa"]
