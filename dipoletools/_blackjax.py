"""BlackJAX backend: JAX-traceable likelihoods, prior transforms, and the
runner invoked by ``Analyser.blackjax(...)``.

Reduced scope vs. ``_ultranest.py``: supports ``gaussian``, ``poisson``,
``general_poisson`` and the ecliptic-bias correction; only ell=[0] or ell=[0,1]
(no quadrupole, no second/additional dipole). For joint analyses, forces
``v, theta, phi`` shared and all other parameters unshared.

JAX/BlackJAX are imported lazily inside functions so the package stays
importable without the ``[blackjax]`` extra installed.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import healpy as hp

from ._defaults import DEFAULT_PRIORS
from ._ultranest import compute_ecliptic_lat


# ----------------------------------------------------------------------
# JAX setup helpers
# ----------------------------------------------------------------------
_X64_CONFIGURED = False
_FLOAT32_FALLBACK = False


def _configure_jax():
    """Detect device and enable float64 unless on Metal (jax-metal lacks x64).

    Returns True if x64 is enabled, False if we fell back to x32.
    Emits a single UserWarning on float32 fallback.
    """
    global _X64_CONFIGURED, _FLOAT32_FALLBACK
    import jax

    if _X64_CONFIGURED:
        return not _FLOAT32_FALLBACK

    devices = jax.devices()
    backend = devices[0].platform if devices else 'cpu'
    is_metal = backend.lower() == 'metal'

    if is_metal:
        _FLOAT32_FALLBACK = True
        warnings.warn(
            "JAX is running on Apple Metal which does not support float64. "
            "Falling back to float32 for BlackJAX nested sampling. Numerical "
            "precision will be reduced; for full float64 precision use a CPU "
            "or CUDA backend.",
            UserWarning,
            stacklevel=3,
        )
    else:
        jax.config.update('jax_enable_x64', True)

    _X64_CONFIGURED = True
    return not _FLOAT32_FALLBACK


# ----------------------------------------------------------------------
# JAX twins of the math
# ----------------------------------------------------------------------
def _jax_ang2vec(theta, phi):
    """JAX twin of ``_utils.ang2vec``: theta=colatitude, phi=longitude (rad)."""
    import jax.numpy as jnp
    sin_t = jnp.sin(theta)
    return jnp.stack([sin_t * jnp.cos(phi),
                       sin_t * jnp.sin(phi),
                       jnp.cos(theta)], axis=-1)


def _jax_compute_expected(N, v, theta, phi, positions, D_survey,
                          bias=None, ecl_lat=None, cecl=None):
    """Monopole + dipole expected counts, with optional ecliptic-bias factor.

    All inputs scalars / 1-D arrays. ``positions`` shape (n_pix, 3).
    """
    import jax.numpy as jnp
    D = v * D_survey
    dipole_vec = _jax_ang2vec(theta, phi)
    cos_angle = jnp.sum(dipole_vec * positions, axis=1)
    expected = N * (1.0 + D * cos_angle)
    if bias is not None and ecl_lat is not None and cecl is not None:
        expected = expected * (1.0 - bias * cecl * jnp.abs(ecl_lat))
    return expected


def _jax_loglike_value(model_type, counts, expected, gp_dispersion=None):
    """JAX twin of ``_ultranest.loglike_value`` for the three supported types."""
    import jax.numpy as jnp
    from jax.scipy.special import gammaln
    expected = jnp.clip(expected, 1e-10, None)
    if model_type == 'poisson':
        # Poisson logpmf: k*log(lam) - lam - gammaln(k+1)
        k = counts
        return jnp.sum(k * jnp.log(expected) - expected - gammaln(k + 1.0))
    if model_type == 'general_poisson':
        b = gp_dispersion if gp_dispersion is not None else 0.0
        k = counts
        lam = expected
        term1 = jnp.log(jnp.clip(lam * (1 - b), 1e-10, None))
        term2 = (k - 1) * jnp.log(jnp.clip(lam * (1 - b) + k * b, 1e-10, None))
        term3 = gammaln(k + 1)
        term4 = lam * (1 - b)
        term5 = k * b
        return jnp.sum(term1 + term2 - term3 - term4 - term5)
    if model_type == 'gaussian':
        std = jnp.std(counts)
        # Normal logpdf
        return jnp.sum(-0.5 * ((counts - expected) / std) ** 2
                       - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi))
    raise ValueError(f"Unknown likelihood type: {model_type}")


# ----------------------------------------------------------------------
# Scope checks and shared/unshared layout for joint analyses
# ----------------------------------------------------------------------
class BlackjaxScopeError(NotImplementedError):
    """Raised when the model uses features outside the BlackJAX scope."""


def _check_supported(analyser):
    """Validate model config(s) against the BlackJAX scope.

    Raises ``BlackjaxScopeError`` (a NotImplementedError) for ell>=2 or any
    second_dipole spec, with a pointer back to ``.ultranest(...)``.
    """
    msg_suffix = (
        " The BlackJAX backend supports only ell=[0] or ell=[0,1] without a "
        "second dipole. Use Analyser.ultranest(...) for this configuration."
    )

    def _check_one(cfg, label):
        if cfg is None:
            return
        ell = cfg.get('ell', [])
        if any(l > 1 for l in ell):
            raise BlackjaxScopeError(
                f"{label} uses ell={ell}." + msg_suffix
            )
        if cfg.get('second_dipole') is not None:
            raise BlackjaxScopeError(
                f"{label} uses second_dipole." + msg_suffix
            )
        if cfg.get('type') == 'custom':
            raise BlackjaxScopeError(
                f"{label} uses a custom likelihood." + msg_suffix
            )

    _check_one(analyser._model_config, "model")
    if analyser._is_composite:
        for cname, child in analyser._children.items():
            _check_one(child._model_config, f"child {cname!r}")
    if analyser._map2 is not None:
        _check_one(analyser._model2_config, "model2")


# Forced shared/unshared layout: v, theta, phi shared; everything else unshared.
_FORCED_SHARED = ('v', 'theta', 'phi')


def _maybe_announce_forced_layout(analyser):
    """Print a one-line note if the user's shared_parameters differs from forced."""
    user_shared = list(analyser._shared_parameters or [])
    forced = list(_FORCED_SHARED)
    if analyser._is_composite or analyser._map2 is not None:
        if set(user_shared) != set(forced):
            print(
                "[dipoletools.blackjax] Forcing shared_parameters="
                f"{forced} (other parameters unshared). "
                "Use Analyser.ultranest(...) for custom shared/unshared layouts."
            )


# ----------------------------------------------------------------------
# Prior-box description and prior transform
# ----------------------------------------------------------------------
def _prior_box(param_names, priors):
    """Return (lows, highs, is_polar) numpy arrays describing the prior box.

    Uniform priors -> lows[i], highs[i] used directly.
    Polar priors -> theta = arccos(2u - 1); for the box description we still
    record lows=0, highs=pi (the support of arccos(2U-1) where U~U(0,1)).
    """
    n = len(param_names)
    lows = np.zeros(n, dtype=float)
    highs = np.zeros(n, dtype=float)
    is_polar = np.zeros(n, dtype=bool)
    for i, p in enumerate(param_names):
        cfg = priors.get(p, {'type': 'uniform', 'low': 0.0, 'high': 1.0})
        if cfg['type'] == 'uniform':
            lows[i] = cfg['low']
            highs[i] = cfg['high']
        elif cfg['type'] == 'polar':
            lows[i] = 0.0
            highs[i] = np.pi
            is_polar[i] = True
        else:
            lows[i] = 0.0
            highs[i] = 1.0
    return lows, highs, is_polar


def _make_jax_logprior(lows, highs, is_polar):
    """Log-prior for a single particle x.

    Uniform on the box (lows, highs) for non-polar dimensions; polar prior
    p(theta) = sin(theta)/2 on [0, pi]. Returns -inf outside support.
    """
    import jax.numpy as jnp
    lows_j = jnp.asarray(lows)
    highs_j = jnp.asarray(highs)
    polar_j = jnp.asarray(is_polar)

    def logprior(x):
        in_box = jnp.all((x >= lows_j) & (x <= highs_j))
        # Uniform contribution: -log(high - low) per non-polar dim
        widths = highs_j - lows_j
        log_uniform = -jnp.sum(jnp.where(polar_j, 0.0, jnp.log(widths)))
        # Polar contribution: log(sin(theta)/2) per polar dim
        log_polar = jnp.sum(jnp.where(polar_j,
                                       jnp.log(jnp.clip(jnp.sin(x), 1e-30, None))
                                       - jnp.log(2.0),
                                       0.0))
        return jnp.where(in_box, log_uniform + log_polar, -jnp.inf)
    return logprior


def _sample_from_prior(rng_key, n_samples, lows, highs, is_polar):
    """Sample n_samples draws from the prior box. Shape (n_samples, n_dim)."""
    import jax
    import jax.numpy as jnp
    n_dim = len(lows)
    lows_j = jnp.asarray(lows)
    highs_j = jnp.asarray(highs)
    polar_j = jnp.asarray(is_polar)
    u = jax.random.uniform(rng_key, shape=(n_samples, n_dim))
    uniform_x = lows_j + u * (highs_j - lows_j)
    polar_x = jnp.arccos(2.0 * u - 1.0)
    return jnp.where(polar_j, polar_x, uniform_x)


# ----------------------------------------------------------------------
# JAX log-likelihood factories
# ----------------------------------------------------------------------
def _per_dataset_arrays(analyser):
    """Extract (counts, positions, D, ecl_lat, cecl, type) for one dataset."""
    nside = hp.npix2nside(len(analyser._map))
    npix = len(analyser._map)
    mask = analyser._mask
    counts = analyser._map[mask]
    pos = np.array(hp.pix2vec(nside, np.arange(npix))).T[mask]
    D_survey = analyser._D
    cfg = analyser._model_config
    ecl_lat = None
    cecl = None
    if cfg.get('bias'):
        cecl = float(analyser._bias_cecl)
        ecl_lat = compute_ecliptic_lat(nside, mask, analyser._map_coords)
    return counts, pos, float(D_survey), ecl_lat, cecl, cfg['type'], bool(cfg.get('bias'))


def _make_jax_loglike_single(analyser):
    """Build a JAX log-likelihood for a single-dataset Analyser.

    Returns
    -------
    loglike : Callable
        ``loglike(x)`` taking a 1-D JAX array of params -> scalar.
    param_names : list of str
        Parameter names matching the order of ``x``.
    lows, highs, is_polar : numpy arrays
        Prior-box description for ``param_names``.
    """
    import jax
    import jax.numpy as jnp

    cfg = analyser._model_config
    priors = analyser._priors_config
    param_names = list(cfg['param_names'])

    counts, pos, D_survey, ecl_lat, cecl, mtype, has_bias = _per_dataset_arrays(analyser)
    counts_j = jnp.asarray(counts)
    pos_j = jnp.asarray(pos)
    ecl_lat_j = jnp.asarray(ecl_lat) if ecl_lat is not None else None

    # Indices of named params in the param vector x
    idx = {p: param_names.index(p) for p in param_names}

    def loglike(x):
        N = x[idx['N']]
        v = x[idx['v']] if 'v' in idx else 0.0
        theta = x[idx['theta']] if 'theta' in idx else 0.0
        phi = x[idx['phi']] if 'phi' in idx else 0.0
        bias = x[idx['bias']] if has_bias else None
        expected = _jax_compute_expected(
            N, v, theta, phi, pos_j, D_survey,
            bias=bias, ecl_lat=ecl_lat_j, cecl=cecl,
        )
        gp = x[idx['gp_dispersion']] if mtype == 'general_poisson' else None
        return _jax_loglike_value(mtype, counts_j, expected, gp_dispersion=gp)

    lows, highs, is_polar = _prior_box(param_names, priors)
    return jax.jit(loglike), param_names, lows, highs, is_polar


def _make_jax_loglike_joint(analyser):
    """Build a JAX log-likelihood for a joint N-dataset Analyser with the
    forced shared layout (v,theta,phi shared; everything else unshared).

    The combined parameter list is::

        [v, theta, phi, <unshared of dataset 1>, <unshared of dataset 2>, ...]

    Unshared params are renamed with the dataset's name suffix (matching the
    convention used in ``_ultranest.build_joint_n`` for N>=3).
    """
    import jax
    import jax.numpy as jnp

    if analyser._is_composite:
        datasets = [analyser] + list(analyser._children.values())
        names = ['a1'] + list(analyser._children.keys())
    elif analyser._map2 is not None:
        # Legacy 2-dataset analyser exposed via .blackjax(...) — treat the
        # second dataset as a single child for parameter renaming. Use the '2'
        # suffix to match the legacy chain layout.
        raise BlackjaxScopeError(
            "Legacy 2-dataset Analyser (Map2=) is not supported by .blackjax(). "
            "Compose Analysers with a1.add(a2) instead."
        )
    else:
        raise ValueError("Not a composite Analyser; use _make_jax_loglike_single.")

    forced_shared = set(_FORCED_SHARED)
    combined_params: list[str] = []
    # Always start with shared params in canonical order
    for p in _FORCED_SHARED:
        if p not in combined_params:
            combined_params.append(p)

    per_dataset_specs = []  # (mtype, idx_dict, counts_j, pos_j, ecl_lat_j, cecl, D, has_bias)

    for d, dname in zip(datasets, names):
        cfg = d._model_config
        if cfg is None:
            raise ValueError(f"Dataset {dname!r} has no model configured.")
        if d._priors_config is None:
            raise ValueError(f"Dataset {dname!r} has no priors configured.")
        if d._map is None:
            raise ValueError(f"Dataset {dname!r} has no map.")
        if d._D is None:
            raise ValueError(f"Dataset {dname!r} has D not set.")
        if d._map_coords != analyser._map_coords:
            raise ValueError(
                f"Dataset {dname!r} has map_coords={d._map_coords!r}; "
                f"self has {analyser._map_coords!r}. "
                "Heterogeneous coordinate systems are not yet supported by .blackjax()."
            )
        # Build mapped names: shared keep base, unshared get _<dname>
        params_base = list(cfg['param_names'])
        params_mapped = []
        for p in params_base:
            if p in forced_shared:
                params_mapped.append(p)
            else:
                params_mapped.append(f'{p}_{dname}')
        for pm in params_mapped:
            if pm not in combined_params:
                combined_params.append(pm)
        # Save dataset arrays
        counts, pos, D_survey, ecl_lat, cecl, mtype, has_bias = _per_dataset_arrays(d)
        counts_j = jnp.asarray(counts)
        pos_j = jnp.asarray(pos)
        ecl_lat_j = jnp.asarray(ecl_lat) if ecl_lat is not None else None
        # Map base -> position in combined_params
        idx_map = {base: combined_params.index(mapped)
                   for base, mapped in zip(params_base, params_mapped)}
        per_dataset_specs.append((mtype, idx_map, counts_j, pos_j,
                                   ecl_lat_j, cecl, D_survey, has_bias))

    def loglike(x):
        total = 0.0
        for (mtype, idx, counts_j, pos_j, ecl_lat_j, cecl, D_survey,
             has_bias) in per_dataset_specs:
            N = x[idx['N']]
            v = x[idx['v']] if 'v' in idx else 0.0
            theta = x[idx['theta']] if 'theta' in idx else 0.0
            phi = x[idx['phi']] if 'phi' in idx else 0.0
            bias = x[idx['bias']] if has_bias else None
            expected = _jax_compute_expected(
                N, v, theta, phi, pos_j, D_survey,
                bias=bias, ecl_lat=ecl_lat_j, cecl=cecl,
            )
            gp = x[idx['gp_dispersion']] if mtype == 'general_poisson' else None
            total = total + _jax_loglike_value(mtype, counts_j, expected,
                                                gp_dispersion=gp)
        return total

    # Build combined priors: shared from analyser, unshared from each dataset
    combined_priors = {}
    for p in _FORCED_SHARED:
        combined_priors[p] = analyser._priors_config.get(
            p, DEFAULT_PRIORS.get(p, {'type': 'uniform', 'low': 0.0, 'high': 1.0})
        )
    for d, dname in zip(datasets, names):
        for p_base in d._model_config['param_names']:
            if p_base in forced_shared:
                continue
            p_mapped = f'{p_base}_{dname}'
            combined_priors[p_mapped] = d._priors_config.get(
                p_base,
                DEFAULT_PRIORS.get(p_base, {'type': 'uniform', 'low': 0.0, 'high': 1.0}),
            )

    lows, highs, is_polar = _prior_box(combined_params, combined_priors)
    return jax.jit(loglike), combined_params, lows, highs, is_polar


# ----------------------------------------------------------------------
# Anesthetic adapter: write equal_weighted_post.txt + anesthetic_stats.npz
# ----------------------------------------------------------------------
def _write_outputs_via_anesthetic(savedir, name, param_names, dead_info):
    """Translate BlackJAX dead-points into anesthetic NestedSamples and write
    the on-disk format expected by ``Posterior``.

    Writes:
      - ``<savedir>/<name>/chains/equal_weighted_post.txt``
      - ``<savedir>/<name>/chains/anesthetic_stats.npz``
    """
    import jax
    import jax.numpy as jnp
    from anesthetic import NestedSamples

    target = os.path.join(savedir, name)
    chains_dir = os.path.join(target, 'chains')
    os.makedirs(chains_dir, exist_ok=True)

    # dead_info.particles is a StateWithLogLikelihood with fields:
    # position, logdensity (log-prior), loglikelihood, loglikelihood_birth.
    pos = np.asarray(dead_info.particles.position)
    logL = np.asarray(dead_info.particles.loglikelihood)
    logL_birth = np.asarray(dead_info.particles.loglikelihood_birth)

    # anesthetic NestedSamples needs columns + logL + logL_birth
    ns = NestedSamples(
        data=pos, columns=list(param_names),
        logL=logL, logL_birth=logL_birth,
    )

    # Equal-weighted samples: anesthetic exposes .compress() or sampling
    # from importance weights; use posterior_points()
    eq = ns.posterior_points()
    samples_arr = eq[list(param_names)].to_numpy()

    # Header line + samples
    eq_path = os.path.join(chains_dir, 'equal_weighted_post.txt')
    np.savetxt(
        eq_path, samples_arr,
        header=' '.join(param_names), comments='',
    )

    # Statistics: logZ, D_KL, d via .stats(nsamples)
    stats = ns.stats(nsamples=200)
    # Each is an anesthetic Sample with mean/std; convert to scalar floats
    def _val(s):
        try:
            return float(np.asarray(s).mean())
        except Exception:
            return float(s)
    def _err(s):
        try:
            arr = np.asarray(s)
            return float(np.std(arr) * 2.0)  # 2 sigma
        except Exception:
            return 0.0

    logZ = _val(stats['logZ']); logZ_err = _err(stats['logZ'])
    DKL = _val(stats['D_KL']); DKL_err = _err(stats['D_KL'])
    d = _val(stats['d_G']); d_err = _err(stats['d_G'])

    # Posterior loads stats from <savedir>/anesthetic_stats.npz; the
    # 2-sigma errors are stored as length-2 arrays (upper, lower).
    np.savez(
        os.path.join(target, 'anesthetic_stats.npz'),
        logZ=logZ, logZ_err=np.array([logZ_err, logZ_err]),
        kl=DKL, kl_err=np.array([DKL_err, DKL_err]),
        d=d, d_err=np.array([d_err, d_err]),
    )

    return target


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def run_blackjax(analyser, savedir=None, name=None, seed=0,
                 n_live=500, n_delete=50, num_mcmc_steps=None,
                 dlogz=0.5, max_iterations=10000):
    """Run nested sampling with BlackJAX. Body of ``Analyser.blackjax(...)``."""
    import jax
    import jax.numpy as jnp
    import blackjax
    import blackjax.ns.nss as nss
    import blackjax.ns.utils as ns_utils

    # Scope checks first (don't import jax-heavy things until we know we run)
    _check_supported(analyser)
    _maybe_announce_forced_layout(analyser)

    if analyser._map is None:
        raise ValueError("No map loaded.")
    if analyser._D is None:
        import scipy.constants as _sc
        analyser._D = 369.83e3 / _sc.c
        warnings.warn(
            "D not specified; falling back to v_CMB/c = "
            f"{analyser._D:.4e}. Pass D= or call "
            "Analyser.expected_amplitude() for a catalogue-derived value."
        )
    if analyser._model_config is None:
        analyser.model()

    _configure_jax()

    # Override shared parameters to the forced layout for joint analyses
    if analyser._is_composite:
        analyser._shared_parameters = list(_FORCED_SHARED)
        loglike_fn, param_names, lows, highs, is_polar = _make_jax_loglike_joint(analyser)
    else:
        loglike_fn, param_names, lows, highs, is_polar = _make_jax_loglike_single(analyser)

    n_dim = len(param_names)
    if num_mcmc_steps is None:
        num_mcmc_steps = max(2 * n_dim, 10)

    logprior_fn = _make_jax_logprior(lows, highs, is_polar)

    # Initialise particles by drawing from the prior
    rng_key = jax.random.PRNGKey(int(seed))
    rng_key, subkey = jax.random.split(rng_key)
    init_positions = _sample_from_prior(subkey, n_live, lows, highs, is_polar)

    algo = nss.as_top_level_api(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglike_fn,
        num_inner_steps=num_mcmc_steps,
        num_delete=n_delete,
    )
    rng_key, init_key = jax.random.split(rng_key)
    state = algo.init(init_positions, rng_key=init_key)

    step_jit = jax.jit(algo.step)

    # Termination: run until the remaining log-evidence in live points is
    # smaller than dlogz vs. the accumulated log-evidence in dead points.
    # A simple fixed-iteration cap is the safe fallback.
    dead = []
    logZ_dead = -jnp.inf
    for it in range(max_iterations):
        rng_key, step_key = jax.random.split(rng_key)
        state, info = step_jit(step_key, state)
        dead.append(info)
        # Cheap termination heuristic: stop when the largest live log-L is
        # smaller than the log-sum-exp of the dead points minus dlogz —
        # i.e. the remaining evidence in live points is < dlogz.
        live_logL = state.particles.loglikelihood
        max_live = jnp.max(live_logL)
        dead_logL = jnp.concatenate(
            [d.particles.loglikelihood for d in dead], axis=0
        )
        logZ_dead = jax.scipy.special.logsumexp(dead_logL)
        if float(max_live) < float(logZ_dead) - dlogz:
            break

    final_info = ns_utils.finalise(state, dead)

    # Resolve savedir / name
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
    used_name = name or 'run1'
    if savedir is not None:
        target = _write_outputs_via_anesthetic(
            savedir, used_name, param_names, final_info,
        )
        analyser._savedir = target

    return final_info
