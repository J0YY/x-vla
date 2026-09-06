"""Legacy callback orchestration, deliberately outside the numerical sweeps."""

from functools import wraps

from xvla.train.odt_engine_v2.core import diagonalize_implicit_dag_full_rank as _diagonalize
from xvla.train.odt_engine_v2.graph import _clone_network


@wraps(_diagonalize)
def diagonalize_implicit_dag_full_rank(network, **options):
    """Preserve rank-bank identity when the historical API copies its input."""
    callback = options.get("compact_spectrum_callback")
    bind = getattr(callback, "bind_network", None)
    if bind is not None:
        if options.get("copy_network", True):
            network = _clone_network(network, unfold=False)
        bind(network)
        options["copy_network"] = False
    return _diagonalize(network, **options)
