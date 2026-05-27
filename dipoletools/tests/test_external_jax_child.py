"""Smoke test for the opaque-JAX-lnlike joint partner API.

This is a *special-case* API for partners that don't fit the Analyser
shape (e.g. an SBI NLE evaluation taking its own parameter dict). It is
exercised end-to-end here on a tiny synthetic problem to lock in the
attach → blackjax → posterior round-trip.
"""

import os
import tempfile

import numpy as np
import healpy as hp
import pytest

from dipoletools import Analyser, Posterior


@pytest.mark.skipif(
    not hasattr(__import__('importlib').util.find_spec('blackjax'), 'origin'),
    reason="blackjax not installed",
)
def test_attach_and_blackjax_round_trip():
    import jax.numpy as jnp

    nside = 8
    npix = hp.nside2npix(nside)
    rng = np.random.RandomState(0)
    m = rng.poisson(50.0, size=npix).astype(float)
    mask = np.ones(npix, dtype=bool)
    mask[:30] = False

    a = Analyser(map=m, mask=mask, D=0.004, map_coords='G')
    a.model(type='poisson', ell=[0, 1])
    a.priors(v=[0, 8])

    def fake_lnlike(d):
        return jnp.asarray(-0.5 * (d['v'] - 1.0) ** 2)

    a.add_external_jax_child(
        name='xtest',
        lnlike_fn=fake_lnlike,
        param_specs=[
            {'name': 'v',     'shared': True},
            {'name': 'theta', 'shared': True},
            {'name': 'phi',   'shared': True},
            {'name': 'aux',   'shared': False, 'low': 0.0, 'high': 1.0,
             'is_polar': False},
        ],
    )
    assert a._is_composite is True
    assert len(a._external_jax_terms) == 1
    assert a._external_jax_terms[0].name == 'xtest'

    out = tempfile.mkdtemp(prefix='dt_ext_test_')
    a.blackjax(savedir=out, name='run', n_live=80, n_delete=40,
               dlogz=2.0, max_iterations=20)

    chain_path = os.path.join(out, 'run', 'chains', 'equal_weighted_post.txt')
    assert os.path.exists(chain_path)

    p = Posterior(os.path.join(out, 'run'), coords='G')
    assert 'v' in p._param_names
    assert 'theta' in p._param_names
    assert 'phi' in p._param_names
    assert 'aux_xtest' in p._param_names


def test_attach_validates_specs():
    a = Analyser(map=np.zeros(hp.nside2npix(8)), D=0.004, map_coords='G')

    with pytest.raises(ValueError):
        a.add_external_jax_child('x', lambda d: 0.0, [])

    with pytest.raises(ValueError):
        # shared param not in (v, theta, phi)
        a.add_external_jax_child(
            'x', lambda d: 0.0,
            [{'name': 'banana', 'shared': True}],
        )

    with pytest.raises(ValueError):
        # unshared missing low/high
        a.add_external_jax_child(
            'x', lambda d: 0.0,
            [{'name': 'aux', 'shared': False}],
        )
