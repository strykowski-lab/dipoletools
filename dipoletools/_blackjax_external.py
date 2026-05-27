"""External JAX-lnlike child support for joint BlackJAX nested sampling.

This is a **special-case** API — not the generic "drop in a custom likelihood"
hook. The use case is:

* You have a normal dipoletools ``Analyser`` for one dataset (the "partner"),
  with the usual ``model()`` / ``priors()`` / parameter list ``(N, v, theta,
  phi, [gp_dispersion], …)``.
* You also have an *opaque* JAX log-likelihood for a second dataset, which
  takes its own parameter dict and returns a scalar. It is **not** wrapped as
  an Analyser, has no ``_model_config``, and may use parameters that are not
  in dipoletools' canonical name list (e.g. ``log10_N``, ``beta``, ``lclus``
  for an SBI-NLE evaluation).
* You want them analysed jointly with the standard ``_FORCED_SHARED =
  ('v', 'theta', 'phi')`` layout.

The motivating user is the racs-low3 SBI integration in
``racs-dipole/dipole_selected_flux_cuts_racs_sbi.py``, where the racs-low3
log-likelihood comes from a neural likelihood estimator (see
``dipolesbi.tools.sbi_io.get_lnlike_data_prior``) and is opaque to dipoletools.

How to use:

>>> partner_an = Analyser(map=..., D=..., map_coords='G')
>>> partner_an.model(type='general_poisson', ell=[0, 1])
>>> partner_an.priors(v=[0, 8])
>>> partner_an.add_external_jax_child(
...     name='low3',
...     lnlike_fn=wrapped_lnlike,         # signature: dict[str,Array] -> scalar
...     param_specs=[                      # in the order lnlike_fn consumes keys
...         {'name': 'v',       'shared': True},
...         {'name': 'theta',   'shared': True},
...         {'name': 'phi',     'shared': True},
...         {'name': 'log10_N', 'shared': False, 'low': 6.2,  'high': 6.8,
...          'is_polar': False},
...         {'name': 'beta',    'shared': False, 'low': 0.0,  'high': 0.05,
...          'is_polar': False},
...         {'name': 'lclus',   'shared': False, 'low': 0.0,  'high': 8.0,
...          'is_polar': False},
...     ],
... )
>>> partner_an.blackjax(savedir=..., name='run', n_live=1000, n_delete=700)

The shared params **must** be a subset of ``('v', 'theta', 'phi')`` (the
joint forced layout); their priors are taken from the partner Analyser. The
unshared params get suffixed with ``_<name>`` in the combined parameter
vector. ``lnlike_fn`` always receives the dict keyed by the **base** names
the user listed in ``param_specs`` — the suffix mapping is internal.

Why not piggy-back on the existing ``type='custom'`` hook? Because that one
takes a parameter vector positionally and assumes the canonical param-name
schema. An NLE log-likelihood has its own schema (own param names, own
units), and we want to share *only* ``(v, theta, phi)`` semantically — with
unit conversion happening inside the user's wrapper. The opaque-dict
contract makes that explicit and avoids contaminating the standard
``Analyser.model()`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ExternalJaxChild:
    """A joint child described only by a JAX lnlike + a parameter spec.

    Attributes
    ----------
    name : str
        Suffix applied to unshared parameter names in the combined param vector.
    lnlike_fn : Callable[[dict[str, Array]], Array]
        Pure JAX log-likelihood. Receives a dict keyed by **base** parameter
        names (as listed in ``param_specs``); returns a scalar.
    param_specs : list of dict
        Each spec must have ``name`` and ``shared``. Unshared specs must also
        carry ``low``, ``high``, ``is_polar`` (bool).
    """

    name: str
    lnlike_fn: Callable[[dict[str, Any]], Any]
    param_specs: list[dict] = field(default_factory=list)

    def shared_names(self) -> list[str]:
        return [s['name'] for s in self.param_specs if s.get('shared')]

    def unshared_specs(self) -> list[dict]:
        return [s for s in self.param_specs if not s.get('shared')]


def attach_to_analyser(analyser, name: str,
                       lnlike_fn: Callable,
                       param_specs: list[dict]) -> None:
    """Body of ``Analyser.add_external_jax_child(...)``.

    Validates the spec, records the child on the analyser, and flips
    ``_is_composite`` so ``run_blackjax`` routes to the joint builder.
    """
    from ._blackjax import _FORCED_SHARED

    if not isinstance(name, str) or not name:
        raise ValueError("External child name must be a non-empty string.")
    if not callable(lnlike_fn):
        raise TypeError("lnlike_fn must be a callable taking a dict.")
    if not param_specs:
        raise ValueError("param_specs must be non-empty.")

    seen: set[str] = set()
    for s in param_specs:
        if 'name' not in s:
            raise ValueError(f"External param spec missing 'name': {s!r}")
        pname = s['name']
        if pname in seen:
            raise ValueError(f"Duplicate external param name: {pname!r}")
        seen.add(pname)
        if 'shared' not in s:
            raise ValueError(
                f"External param spec {pname!r} missing 'shared' bool.")
        if s['shared']:
            if pname not in _FORCED_SHARED:
                raise ValueError(
                    f"Shared external param {pname!r} must be one of "
                    f"{_FORCED_SHARED!r} (the BlackJAX forced-shared layout)."
                )
        else:
            for key in ('low', 'high', 'is_polar'):
                if key not in s:
                    raise ValueError(
                        f"Unshared external param {pname!r} missing {key!r}.")

    terms: list[ExternalJaxChild] = getattr(analyser, '_external_jax_terms', [])
    if any(t.name == name for t in terms):
        raise ValueError(f"External child {name!r} already attached.")
    terms.append(ExternalJaxChild(
        name=name, lnlike_fn=lnlike_fn, param_specs=list(param_specs),
    ))
    analyser._external_jax_terms = terms
    analyser._is_composite = True
    if not analyser._shared_parameters:
        analyser._shared_parameters = list(_FORCED_SHARED)


def extend_joint_with_externals(loglike_inner, combined_params, lows, highs,
                                is_polar, external_terms):
    """Wrap a joint Analyser loglike with extra opaque-JAX children.

    Returns ``(new_loglike, new_combined_params, new_lows, new_highs,
    new_is_polar)``. The combined param vector is extended with
    ``<base>_<child.name>`` entries for each child's unshared params (preserving
    the spec order).

    The new ``loglike`` is **not** jit-compiled here — the caller
    (``run_blackjax``) re-wraps with ``jax.jit``.
    """
    import jax.numpy as jnp
    import numpy as _np

    new_params = list(combined_params)
    new_lows = _np.asarray(lows).copy().tolist()
    new_highs = _np.asarray(highs).copy().tolist()
    new_polar = _np.asarray(is_polar).copy().tolist()

    # Per-child: build a list of (base_name, index_in_x)
    per_child_index: list[list[tuple[str, int]]] = []

    for child in external_terms:
        pairs: list[tuple[str, int]] = []
        for s in child.param_specs:
            base = s['name']
            if s.get('shared'):
                if base not in new_params:
                    raise ValueError(
                        f"External child {child.name!r}: shared param "
                        f"{base!r} not found in combined params {new_params!r}."
                    )
                idx = new_params.index(base)
            else:
                mapped = f'{base}_{child.name}'
                if mapped in new_params:
                    raise ValueError(
                        f"Combined param name collision: {mapped!r} is already "
                        "in the combined parameter list."
                    )
                new_params.append(mapped)
                new_lows.append(float(s['low']))
                new_highs.append(float(s['high']))
                new_polar.append(bool(s['is_polar']))
                idx = len(new_params) - 1
            pairs.append((base, idx))
        per_child_index.append(pairs)

    # Snapshot the lnlike functions so the closure doesn't depend on mutable
    # iteration state.
    child_fns = [c.lnlike_fn for c in external_terms]

    lows_arr = jnp.asarray(new_lows)
    highs_arr = jnp.asarray(new_highs)

    def loglike(x):
        # Inner Analyser loglike sees only its own slice; padding with extra
        # external dims is harmless because that builder indexes by name.
        total = loglike_inner(x[:len(combined_params)])
        for pairs, fn in zip(per_child_index, child_fns):
            param_dict = {base: x[idx] for base, idx in pairs}
            total = total + fn(param_dict)
        in_box = jnp.all((x >= lows_arr) & (x <= highs_arr))
        return jnp.where(in_box, total, -jnp.inf)

    return (loglike, new_params, _np.asarray(new_lows),
            _np.asarray(new_highs), _np.asarray(new_polar))
