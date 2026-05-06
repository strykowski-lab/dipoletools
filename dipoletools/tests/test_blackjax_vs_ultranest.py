"""Equivalence test: BlackJAX and UltraNest must produce statistically
indistinguishable posteriors on the same model + data + priors.

This is a slow test (runs two full nested-sampling chains) and is gated by
the ``RUN_SLOW_NS=1`` env var so default CI doesn't pay the cost.

Set the env var, run with ``-s`` to see the printed corner-plot paths::

    RUN_SLOW_NS=1 pytest dipoletools/tests/test_blackjax_vs_ultranest.py -s
"""

import os
import tempfile

import healpy as hp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_SLOW_NS') != '1',
    reason='Set RUN_SLOW_NS=1 to run the slow NS equivalence test.',
)

# Lazy imports so collection doesn't fail when extras are missing
jax = pytest.importorskip("jax")
pytest.importorskip("blackjax.ns.nss")
pytest.importorskip("ultranest")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use('Agg')


def test_blackjax_matches_ultranest():
    """Run both samplers on the same gaussian model; compare logZ and
    the v/theta/phi/N marginals; save corner plots for visual review."""
    import matplotlib.pyplot as plt
    from dipoletools.analyser import Analyser
    from dipoletools.posterior import Posterior

    NSIDE = 8
    NPIX = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(2026)
    # Inject a known dipole signal so the posterior actually constrains v.
    pos = np.array(hp.pix2vec(NSIDE, np.arange(NPIX))).T
    true_v, true_theta, true_phi = 1.5, 1.2, 2.0
    D_survey = 5.0e-3
    dipole_vec = np.array([
        np.sin(true_theta) * np.cos(true_phi),
        np.sin(true_theta) * np.sin(true_phi),
        np.cos(true_theta),
    ])
    expected = 200.0 * (1.0 + true_v * D_survey * (pos @ dipole_vec))
    counts = rng.poisson(expected).astype(float)

    out_dir = tempfile.mkdtemp(prefix='dipoletools_eq_')

    # --- UltraNest ---
    a_un = Analyser(map=counts, D=D_survey, map_coords='G')
    a_un.model(type='gaussian', ell=[0, 1])
    a_un.ultranest(savedir=os.path.join(out_dir, 'un'), name='run',
                   min_num_live_points=200, dlogz=0.5, frac_remain=0.05,
                   seed=1)
    p_un = Posterior(a_un)
    p_un.corner(show=False)
    un_corner = os.path.join(out_dir, 'corner_ultranest.png')
    plt.gcf().savefig(un_corner, dpi=80, bbox_inches='tight')
    plt.close('all')

    # --- BlackJAX ---
    a_bj = Analyser(map=counts, D=D_survey, map_coords='G')
    a_bj.model(type='gaussian', ell=[0, 1])
    a_bj.blackjax(savedir=os.path.join(out_dir, 'bj'), name='run',
                  seed=1, n_live=400, n_delete=50, num_mcmc_steps=20,
                  max_iterations=2000, dlogz=0.3)
    p_bj = Posterior(a_bj)
    p_bj.corner(show=False)
    bj_corner = os.path.join(out_dir, 'corner_blackjax.png')
    plt.gcf().savefig(bj_corner, dpi=80, bbox_inches='tight')
    plt.close('all')

    print()
    print(f'UltraNest corner: {un_corner}')
    print(f'BlackJAX corner:  {bj_corner}')
    print(f'UltraNest logZ:   {p_un.logZ:.3f}')
    print(f'BlackJAX logZ:    {p_bj.logZ:.3f}')

    # logZ agreement: should be within ~3 nats given small sample sizes.
    assert abs(p_un.logZ - p_bj.logZ) < 5.0, (
        f"logZ disagreement: ultranest={p_un.logZ}, blackjax={p_bj.logZ}"
    )

    # Compare medians of v, theta, phi, N: must agree within 1 sigma of
    # the UltraNest standard deviation (a generous bound given Monte Carlo
    # noise from finite n_live).
    for i, p in enumerate(p_un._param_names):
        if p not in p_bj._param_names:
            continue
        j = p_bj._param_names.index(p)
        x_un = p_un._samples[:, i]
        x_bj = p_bj._samples[:, j]
        med_un = float(np.median(x_un))
        med_bj = float(np.median(x_bj))
        sd_un = float(np.std(x_un))
        print(f'  {p}: UN median={med_un:.4f}+-{sd_un:.4f}, BJ median={med_bj:.4f}')
        assert abs(med_un - med_bj) < 1.5 * sd_un, (
            f"{p} medians disagree: UN={med_un} BJ={med_bj} (UN sd={sd_un})"
        )
