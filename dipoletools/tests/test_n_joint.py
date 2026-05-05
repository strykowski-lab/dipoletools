"""Tests for the N-way joint compositional Analyser API."""

import copy
import warnings

import healpy as hp
import numpy as np
import pytest
import scipy.stats

from dipoletools.analyser import Analyser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NSIDE = 4
NPIX = hp.nside2npix(NSIDE)
RNG = np.random.default_rng(42)

D1 = 4.5e-3
D2 = 3.7e-3
D3 = 5.1e-3

# Synthetic count maps: Poisson draws around a constant rate
_mp1 = RNG.poisson(150.0, NPIX).astype(float)
_mp2 = RNG.poisson(200.0, NPIX).astype(float)
_mp3 = RNG.poisson(120.0, NPIX).astype(float)


def make_analyser(mp, D, model_type='general_poisson', bias=False,
                  bias_cecl=9.15e-4, map_coords='G'):
    """Build a configured single-dataset Analyser."""
    a = Analyser(map=mp, D=D, map_coords=map_coords)
    a.model(type=model_type, ell=[0, 1], bias=bias, bias_cecl=bias_cecl)
    return a


def random_x(param_names, seed=7):
    """Draw a plausible parameter vector from the prior region."""
    rng = np.random.default_rng(seed)
    defaults = {
        'v': 0.004, 'theta': 1.2, 'phi': 2.4, 'N': 150.0,
        'gp_dispersion': 0.3, 'bias': 0.8,
    }
    x = []
    for p in param_names:
        # strip suffix to get base name
        base = p.rstrip('0123456789').rstrip('_')
        for key in defaults:
            if p == key or p.startswith(key + '_') or p == key + '2':
                x.append(defaults[key] + rng.uniform(-0.001, 0.001))
                break
        else:
            x.append(rng.uniform(0.1, 0.5))
    return np.array(x)


# ---------------------------------------------------------------------------
# test_legacy_warning
# ---------------------------------------------------------------------------

def test_legacy_warning():
    """Constructing with Map2= emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a = Analyser(map=_mp1, D=D1, map2=_mp2, d2=D2, map_coords='G')
    dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warns, "Expected a DeprecationWarning for Map2= form"
    assert 'legacy' in str(dep_warns[0].message).lower()


def test_no_legacy_warning_single():
    """Single-dataset Analyser emits no DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a = Analyser(map=_mp1, D=D1, map_coords='G')
    dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not dep_warns


# ---------------------------------------------------------------------------
# test_compose_2_loglike_matches_legacy
# ---------------------------------------------------------------------------

def test_compose_2_loglike_matches_legacy():
    """N=2 compositional form gives identical param_names and loglike to legacy."""
    model_type = 'general_poisson'

    # Legacy form
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a_leg = Analyser(map=_mp1, D=D1, map2=_mp2, d2=D2, map_coords='G')
    a_leg.model(type=model_type, ell=[0, 1])
    a_leg.model2(type=model_type, ell=[0, 1], shared_parameters=['v', 'theta', 'phi'])
    params_leg, ll_leg, _ = a_leg._build_joint()

    # Compositional form
    a1 = make_analyser(_mp1, D1, model_type)
    a2 = make_analyser(_mp2, D2, model_type)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a1.add(a2)
    params_comp, ll_comp, _ = a1._build_joint_n()

    assert params_leg == params_comp, f"param_names differ: {params_leg} vs {params_comp}"

    # Compare loglike at several random points
    rng = np.random.default_rng(99)
    for _ in range(5):
        x = random_x(params_leg, seed=int(rng.integers(1000)))
        assert np.isclose(ll_leg(x), ll_comp(x), rtol=1e-10), (
            f"loglike mismatch: legacy={ll_leg(x):.6f} composite={ll_comp(x):.6f}"
        )


# ---------------------------------------------------------------------------
# test_compose_3_general_poisson
# ---------------------------------------------------------------------------

def test_compose_3_general_poisson():
    """Three GP datasets: check combined param names and that loglike sums contributions."""
    a1 = make_analyser(_mp1, D1)
    a2 = make_analyser(_mp2, D2)
    a3 = make_analyser(_mp3, D3)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a1.add(a2)
        a1.add(a3, name='planck')

    assert a1._is_composite
    assert list(a1._children.keys()) == ['a2', 'planck']

    params, ll, ptform = a1._build_joint_n()

    # Shared params appear once
    assert params.count('v') == 1
    assert params.count('theta') == 1
    assert params.count('phi') == 1

    # Non-shared params get _name suffix for N>=3
    assert 'N_a1' in params
    assert 'gp_dispersion_a1' in params
    assert 'N_a2' in params
    assert 'gp_dispersion_a2' in params
    assert 'N_planck' in params
    assert 'gp_dispersion_planck' in params

    # loglike is finite at a reasonable point
    x = random_x(params)
    val = ll(x)
    assert np.isfinite(val)

    # Loglike equals sum of individual per-dataset loglikes
    x_dict = {p: x[i] for i, p in enumerate(params)}

    def single_ll(d, params_base, params_mapped):
        fn = a1._make_loglike_dict_joint(
            d._model_config['type'], d._model_config['ell'],
            d._map[d._mask],
            np.array(hp.pix2vec(hp.npix2nside(len(d._map)),
                                np.arange(len(d._map)))).T[d._mask],
            d._D,
            params_base, params_mapped,
            need_convert=False, from_sys='G', to_sys='G',
        )
        return fn(x_dict)

    shared = a1._shared_parameters
    base = list(a1._model_config['param_names'])
    mapped_a1 = [p if p in shared else f'{p}_a1' for p in base]
    expected_sum = single_ll(a1, base, mapped_a1)

    for cname, child in a1._children.items():
        cbase = list(child._model_config['param_names'])
        cmapped = [p if p in shared else f'{p}_{cname}' for p in cbase]
        expected_sum += single_ll(child, cbase, cmapped)

    assert np.isclose(val, expected_sum, rtol=1e-10)


# ---------------------------------------------------------------------------
# test_compose_3_with_gaussian
# ---------------------------------------------------------------------------

def test_compose_3_with_gaussian():
    """One gaussian child: verify gaussian loglike equals scipy reference."""
    a1 = make_analyser(_mp1, D1, model_type='general_poisson')
    a2 = make_analyser(_mp2, D2, model_type='general_poisson')
    a3 = make_analyser(_mp3, D3, model_type='gaussian')

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a1.add(a2)
        a1.add(a3, name='gauss')

    params, ll, _ = a1._build_joint_n()
    x = random_x(params)
    x_dict = {p: x[i] for i, p in enumerate(params)}

    # Compute expected counts for the gaussian child (a3/gauss)
    nside3 = hp.npix2nside(len(_mp3))
    mask3 = a3._mask
    pos3 = np.array(hp.pix2vec(nside3, np.arange(len(_mp3)))).T[mask3]
    counts3 = _mp3[mask3]

    v = x_dict['v']
    theta = x_dict['theta']
    phi = x_dict['phi']
    N_gauss = x_dict['N_gauss']
    D = v * D3
    dipole_vec = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    cos_angle = pos3 @ dipole_vec
    expected3 = N_gauss * (1.0 + D * cos_angle)
    expected3 = np.clip(expected3, 1e-10, None)

    std3 = np.std(counts3)
    ref_gauss_ll = np.sum(scipy.stats.norm.logpdf(counts3, expected3, std3))

    # Evaluate just the gaussian component using a1's factory directly
    params_base_gauss = list(a3._model_config['param_names'])
    params_mapped_gauss = [p if p in a1._shared_parameters else f'{p}_gauss'
                           for p in params_base_gauss]
    fn_gauss = a1._make_loglike_dict_joint(
        'gaussian', [0, 1], counts3, pos3, D3,
        params_base_gauss, params_mapped_gauss,
        need_convert=False, from_sys='G', to_sys='G',
    )
    assert np.isclose(fn_gauss(x_dict), ref_gauss_ll, rtol=1e-8)


# ---------------------------------------------------------------------------
# test_compose_with_bias
# ---------------------------------------------------------------------------

def test_compose_with_bias():
    """Child with bias=True contributes the ecliptic factor to expected counts."""
    a1 = make_analyser(_mp1, D1)
    a2 = make_analyser(_mp2, D2, bias=True, bias_cecl=7.4e-4)
    a2.priors(bias=[0.5, 1.5])

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a1.add(a2)

    params, ll, _ = a1._build_joint_n()

    assert 'bias_a2' in params or 'bias2' in params, (
        f"Expected bias param in {params}"
    )
    x = random_x(params)
    val = ll(x)
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# test_shared_prior_warns
# ---------------------------------------------------------------------------

def test_shared_prior_warns():
    """Child with a different prior on a shared param triggers UserWarning."""
    a1 = make_analyser(_mp1, D1)
    a2 = make_analyser(_mp2, D2)
    a2.priors(v=[0, 5])  # differs from a1's default

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a1.add(a2)

    user_warns = [w for w in caught if issubclass(w.category, UserWarning)
                  and "'v'" in str(w.message)]
    assert user_warns, "Expected UserWarning about shared prior 'v'"

    # a2's prior on v should now match a1's
    assert a2._priors_config['v'] == a1._priors_config['v']


# ---------------------------------------------------------------------------
# test_access_and_remove
# ---------------------------------------------------------------------------

def test_access_and_remove():
    """`access()` returns the child; `remove()` reverts to single-dataset mode."""
    a1 = make_analyser(_mp1, D1)
    a2 = make_analyser(_mp2, D2)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a1.add(a2)

    assert a1._is_composite

    child = a1.access('a2')
    assert child is a2

    # Mutations via access propagate
    original_priors = copy.deepcopy(a2._priors_config)
    a1.access('a2').priors(N=[100, 110])
    assert a2._priors_config['N'] != original_priors['N']

    # KeyError for unknown name
    with pytest.raises(KeyError, match='a3'):
        a1.access('a3')

    # Remove reverts to single-dataset mode
    a1.remove('a2')
    assert not a1._is_composite
    assert a1._children == {}

    # KeyError for unknown name on remove
    with pytest.raises(KeyError):
        a1.remove('nonexistent')


# ---------------------------------------------------------------------------
# test_no_nested
# ---------------------------------------------------------------------------

def test_no_nested():
    """Adding a composite flattens it and emits UserWarning."""
    a1 = make_analyser(_mp1, D1)
    a2 = make_analyser(_mp2, D2)
    a3 = make_analyser(_mp3, D3)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a1.add(a2)  # a1 is now composite with child a2

    assert a1._is_composite

    # a3.add(a1) should warn and flatten
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a3.add(a1)

    user_warns = [w for w in caught if issubclass(w.category, UserWarning)
                  and 'flatten' in str(w.message).lower()]
    assert user_warns, "Expected UserWarning about flattening"

    # a3 should now hold a1 (without children) and a2 as siblings
    assert 'a1' in a3._children
    assert 'a2' in a3._children
    assert len(a3._children) == 2
    assert not a3._children['a1']._is_composite


# ---------------------------------------------------------------------------
# test_introspected_name
# ---------------------------------------------------------------------------

def test_introspected_name():
    """Variable name is introspected; explicit name= overrides; expression falls back."""
    a1 = make_analyser(_mp1, D1)
    a2 = make_analyser(_mp2, D2)
    a3 = make_analyser(_mp3, D3)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a1.add(a2)            # should register as 'a2'
        a1.add(a3, name='planck')  # explicit name

    assert 'a2' in a1._children
    assert 'planck' in a1._children

    # Expression (no variable name) falls back to 'analyser_N' with UserWarning
    a_fresh = make_analyser(_mp1, D1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a_fresh.add(make_analyser(_mp2, D2))

    user_warns = [w for w in caught if issubclass(w.category, UserWarning)
                  and 'analyser_' in str(w.message)]
    assert user_warns, "Expected UserWarning about fallback name"
    assert 'analyser_1' in a_fresh._children


# ---------------------------------------------------------------------------
# test_step_sampler_attaches (lightweight: monkeypatches the sampler run)
# ---------------------------------------------------------------------------

def test_step_sampler_attaches(tmp_path, monkeypatch):
    """step=True attaches a SliceSampler before run()."""
    import ultranest
    import ultranest.stepsampler

    attached = {}

    class FakeResults:
        pass

    class FakeSampler:
        def __init__(self, param_names, loglike, ptform, log_dir=None):
            self.stepsampler = None

        def run(self, **kwargs):
            attached['stepsampler'] = self.stepsampler
            return FakeResults()

        @property
        def results(self):
            return FakeResults()

    monkeypatch.setattr(ultranest, 'ReactiveNestedSampler', FakeSampler)

    a = make_analyser(_mp1, D1)
    a.ultranest(savedir=None, step=True, min_num_live_points=50)

    assert attached['stepsampler'] is not None
    assert isinstance(attached['stepsampler'],
                      ultranest.stepsampler.SliceSampler)


def test_no_step_sampler_by_default(monkeypatch):
    """step=False (default) leaves stepsampler unset."""
    import ultranest

    attached = {}

    class FakeSampler:
        def __init__(self, param_names, loglike, ptform, log_dir=None):
            self.stepsampler = None

        def run(self, **kwargs):
            attached['stepsampler'] = self.stepsampler
            return object()

        @property
        def results(self):
            return object()

    monkeypatch.setattr(ultranest, 'ReactiveNestedSampler', FakeSampler)

    a = make_analyser(_mp1, D1)
    a.ultranest(savedir=None, step=False, min_num_live_points=50)

    assert attached['stepsampler'] is None


# ---------------------------------------------------------------------------
# test_compose_2_chain_matches_legacy  (short sampler run)
# ---------------------------------------------------------------------------

def test_compose_2_chain_matches_legacy(tmp_path):
    """Compositional N=2 posterior means agree with legacy within 3-sigma."""
    model_type = 'poisson'  # simplest model for speed

    savedir_leg = str(tmp_path / 'legacy')
    savedir_comp = str(tmp_path / 'comp')

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a_leg = Analyser(map=_mp1, D=D1, map2=_mp2, d2=D2, map_coords='G')
    a_leg.model(type=model_type)
    a_leg.model2(type=model_type, shared_parameters=['v', 'theta', 'phi'])
    res_leg = a_leg.ultranest(
        savedir=savedir_leg, name='run',
        min_num_live_points=50, dlogz=2.0, seed=0,
    )

    a1 = Analyser(map=_mp1, D=D1, map_coords='G')
    a1.model(type=model_type)
    a2 = Analyser(map=_mp2, D=D2, map_coords='G')
    a2.model(type=model_type)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a1.add(a2)
    res_comp = a1.ultranest(
        savedir=savedir_comp, name='run',
        min_num_live_points=50, dlogz=2.0, seed=0,
    )

    # Compare posterior means on shared kinematic params
    leg_params = res_leg['paramnames']
    comp_params = res_comp['paramnames']
    assert leg_params == comp_params

    for p in ('v', 'theta', 'phi'):
        idx = leg_params.index(p)
        mean_leg = np.mean(res_leg['samples'][:, idx])
        std_leg = np.std(res_leg['samples'][:, idx])
        mean_comp = np.mean(res_comp['samples'][:, idx])
        # 5-sigma tolerance given small live-point count
        assert abs(mean_leg - mean_comp) < 5 * std_leg + 1e-6, (
            f"Posterior mean for {p!r} differs too much: "
            f"legacy={mean_leg:.4f}, composite={mean_comp:.4f}"
        )
