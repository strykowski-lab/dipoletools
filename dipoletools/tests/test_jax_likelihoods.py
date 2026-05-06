"""Float-precision parity tests: JAX log-likelihoods must match the NumPy
ones used by the UltraNest backend to ~1e-10.

These tests are skipped if JAX/BlackJAX are not installed.
"""

import healpy as hp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# Enable x64 BEFORE any JAX arrays are created (parity is a 1e-10 claim)
jax.config.update("jax_enable_x64", True)

from dipoletools.analyser import Analyser  # noqa: E402
from dipoletools import _blackjax as _bj  # noqa: E402
from dipoletools import _ultranest as _un  # noqa: E402


NSIDE = 4
NPIX = hp.nside2npix(NSIDE)
RNG = np.random.default_rng(42)

D1 = 4.5e-3
D2 = 3.7e-3
D3 = 5.1e-3

_mp1 = RNG.poisson(150.0, NPIX).astype(float)
_mp2 = RNG.poisson(200.0, NPIX).astype(float)
_mp3 = RNG.poisson(120.0, NPIX).astype(float)


def _make(mp, D, model_type, bias=False):
    a = Analyser(map=mp, D=D, map_coords='G')
    a.model(type=model_type, ell=[0, 1], bias=bias)
    return a


def _draw_x(param_names, seed=7):
    rng = np.random.default_rng(seed)
    defaults = {
        'v': 0.004, 'theta': 1.2, 'phi': 2.4, 'N': 150.0,
        'gp_dispersion': 0.3, 'bias': 0.8,
    }
    x = []
    for p in param_names:
        base = None
        for key in defaults:
            if p == key or p == key + '2' or p.startswith(key + '_'):
                base = key
                break
        if base is None:
            x.append(rng.uniform(0.1, 0.5))
        else:
            x.append(defaults[base] + rng.uniform(-0.001, 0.001))
    return np.array(x, dtype=float)


@pytest.mark.parametrize("model_type", ["gaussian", "poisson", "general_poisson"])
@pytest.mark.parametrize("bias", [False, True])
def test_single_dataset_parity(model_type, bias):
    """JAX single-dataset loglike == NumPy single-dataset loglike to 1e-10."""
    a = _make(_mp1, D1, model_type, bias=bias)
    # NumPy reference
    np_param_names, np_loglike, _ = _un.build_single(a)
    # JAX twin
    jax_loglike, jax_param_names, lows, highs, is_polar = _bj._make_jax_loglike_single(a)
    assert np_param_names == jax_param_names

    for seed in (3, 11, 19):
        x = _draw_x(np_param_names, seed=seed)
        np_val = float(np_loglike(x))
        jax_val = float(jax_loglike(jnp.asarray(x)))
        assert np.isclose(np_val, jax_val, rtol=1e-10, atol=1e-10), (
            f"{model_type} bias={bias} seed={seed}: numpy={np_val} jax={jax_val} "
            f"diff={np_val - jax_val}"
        )


@pytest.mark.parametrize("model_type", ["gaussian", "poisson", "general_poisson"])
@pytest.mark.parametrize("bias", [False, True])
def test_two_dataset_joint_parity(model_type, bias):
    """JAX joint loglike (N=2) matches NumPy joint loglike to 1e-10.

    Uses the forced shared layout (v,theta,phi shared; everything else
    unshared). The NumPy reference uses the same forced layout via
    ``_shared_parameters = ['v','theta','phi']``.
    """
    a1 = _make(_mp1, D1, model_type, bias=bias)
    a2 = _make(_mp2, D2, model_type, bias=bias)
    a1.add(a2, name='a2')
    a1._shared_parameters = ['v', 'theta', 'phi']

    np_param_names, np_loglike, _ = _un.build_joint_n(a1)
    jax_loglike, jax_param_names, lows, highs, is_polar = _bj._make_jax_loglike_joint(a1)

    # The two backends may differ in suffix convention for N=2 (legacy '2'
    # vs the unified '_<name>' used by JAX). Translate by mapping.
    # We just assert both vectors carry the same SET of physical params.
    assert sorted([p.split('_')[0].rstrip('012') if p not in ('v','theta','phi') else p
                   for p in np_param_names]) == \
           sorted([p.split('_')[0] if p not in ('v','theta','phi') else p
                   for p in jax_param_names])

    # Build a parameter vector for each ordering; values must agree on shared
    # params and per-dataset on unshared.
    # Strategy: draw values for each base name once, then materialise both
    # vectors from the same dict.
    rng = np.random.default_rng(7)
    base_to_value = {}
    for p in jax_param_names:
        # split shared vs unshared
        if p in ('v', 'theta', 'phi'):
            base_to_value[p] = _draw_x([p], seed=hash(p) % 2**32)[0]
        else:
            base_to_value[p] = rng.uniform(0.1, 200.0) if p.startswith('N') else rng.uniform(0.0, 0.9)

    # Map for NumPy backend names: replace '_a2' suffix with '2'
    def _np_name_to_jax_name(p):
        if p in ('v', 'theta', 'phi'):
            return p
        if p.endswith('2'):
            return p[:-1] + '_a2'
        return p  # no suffix (self in N=2)

    x_jax = np.array([base_to_value[p] for p in jax_param_names])
    x_np = np.array([base_to_value[_np_name_to_jax_name(p)]
                     if _np_name_to_jax_name(p) in base_to_value
                     else base_to_value[p + '_a1' if p not in ('v','theta','phi') else p]
                     for p in np_param_names])

    # The cleaner way: for each NumPy param name, derive the corresponding
    # JAX-vector name and pull the same scalar.
    np_to_jax = {}
    for p in np_param_names:
        if p in ('v', 'theta', 'phi'):
            np_to_jax[p] = p
        elif p.endswith('2'):
            np_to_jax[p] = p[:-1] + '_a2'
        else:
            # Self in N=2 has no suffix; in JAX we add '_a1'
            np_to_jax[p] = p + '_a1'
    # But JAX combined_params doesn't currently include '_a1' suffix because
    # the parent dataset's name is 'a1' and we always suffix unshared with
    # the dataset name. So jax has v,theta,phi,N_a1,...,N_a2,...
    # Verify that's what we built:
    assert any(name.endswith('_a1') for name in jax_param_names)

    x_np = np.array([base_to_value[np_to_jax[p]] for p in np_param_names])

    np_val = float(np_loglike(x_np))
    jax_val = float(jax_loglike(jnp.asarray(x_jax)))
    assert np.isclose(np_val, jax_val, rtol=1e-10, atol=1e-10), (
        f"{model_type} bias={bias}: numpy={np_val} jax={jax_val} "
        f"diff={np_val - jax_val}"
    )


@pytest.mark.parametrize("model_type", ["gaussian", "poisson", "general_poisson"])
@pytest.mark.parametrize("bias", [False, True])
def test_three_dataset_joint_parity(model_type, bias):
    """JAX joint loglike (N=3) matches NumPy joint loglike to 1e-10."""
    a1 = _make(_mp1, D1, model_type, bias=bias)
    a2 = _make(_mp2, D2, model_type, bias=bias)
    a3 = _make(_mp3, D3, model_type, bias=bias)
    a1.add(a2, name='a2')
    a1.add(a3, name='a3')
    a1._shared_parameters = ['v', 'theta', 'phi']

    np_param_names, np_loglike, _ = _un.build_joint_n(a1)
    jax_loglike, jax_param_names, lows, highs, is_polar = _bj._make_jax_loglike_joint(a1)

    # For N>=3 both backends use the '_<dname>' suffix, but the NumPy backend
    # uses 'a1' for self while the JAX backend also uses 'a1'. So names should
    # agree as a SET (order may differ).
    assert set(np_param_names) == set(jax_param_names), (
        f"NP: {np_param_names}\nJAX: {jax_param_names}"
    )

    # Build a value dict and materialise both vectors
    rng = np.random.default_rng(11)
    base_to_value = {}
    for p in jax_param_names:
        if p in ('v',): base_to_value[p] = 0.004
        elif p == 'theta': base_to_value[p] = 1.2
        elif p == 'phi': base_to_value[p] = 2.4
        elif p.startswith('N_'): base_to_value[p] = float(rng.uniform(100, 200))
        elif p.startswith('bias_'): base_to_value[p] = float(rng.uniform(-0.5, 0.5))
        elif p.startswith('gp_dispersion_'): base_to_value[p] = float(rng.uniform(0.05, 0.5))
        else: base_to_value[p] = float(rng.uniform(0.1, 0.5))

    x_jax = np.array([base_to_value[p] for p in jax_param_names])
    x_np = np.array([base_to_value[p] for p in np_param_names])

    np_val = float(np_loglike(x_np))
    jax_val = float(jax_loglike(jnp.asarray(x_jax)))
    assert np.isclose(np_val, jax_val, rtol=1e-10, atol=1e-10), (
        f"{model_type} bias={bias}: numpy={np_val} jax={jax_val} "
        f"diff={np_val - jax_val}"
    )
