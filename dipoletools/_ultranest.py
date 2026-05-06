"""UltraNest backend: model-build helpers, likelihood factories, and the
runner invoked by ``Analyser.ultranest(...)``.

This module hosts the CPU/UltraNest path. The JAX/BlackJAX path lives in
``_blackjax.py``. ``analyser.py`` keeps thin method shims that delegate here
so the public API is unchanged.
"""

import os
import warnings

import numpy as np
import healpy as hp
import scipy as sp
from scipy.special import gammaln

from ._utils import ang2vec, convert_thetaphi
from ._defaults import DEFAULT_PRIORS


# ----------------------------------------------------------------------
# Ecliptic latitude (used by both single and joint builders)
# ----------------------------------------------------------------------
def compute_ecliptic_lat(nside, mask, map_coords):
    """Compute ecliptic latitudes (degrees) for unmasked pixels."""
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    npix = hp.nside2npix(nside)
    theta_pix, phi_pix = hp.pix2ang(nside, np.arange(npix))
    lon_deg = np.degrees(phi_pix)
    lat_deg = 90.0 - np.degrees(theta_pix)

    if map_coords == 'G':
        sc = SkyCoord(l=lon_deg * u.deg, b=lat_deg * u.deg, frame='galactic')
    elif map_coords == 'C':
        sc = SkyCoord(ra=lon_deg * u.deg, dec=lat_deg * u.deg, frame='icrs')
    elif map_coords == 'E':
        return lat_deg[mask]
    else:
        raise ValueError(f"Unknown coordinate system: {map_coords}")

    ecl = sc.transform_to('barycentricmeanecliptic')
    return ecl.lat.deg[mask]


# ----------------------------------------------------------------------
# Expected counts and log-likelihood evaluation
# ----------------------------------------------------------------------
def compute_expected(params, data_positions, D_survey, ell, second_dipole=None):
    """Compute expected counts from model parameters."""
    N = params.get('N', 1.0)
    expected = np.full(len(data_positions), N, dtype=float)

    if 1 in ell and 'v' in params:
        v = params['v']
        theta = params['theta']
        phi = params['phi']
        D = v * D_survey
        dipole_vec = ang2vec(theta, phi)
        cos_angle = np.sum(dipole_vec * data_positions, axis=1)
        expected = N * (1 + D * cos_angle)

    if 2 in ell and 'Q' in params:
        Q = params['Q']
        theta_a = params['theta_a']
        phi_a = params['phi_a']
        theta_b = params['theta_b']
        phi_b = params['phi_b']
        a = ang2vec(theta_a, phi_a)
        b = ang2vec(theta_b, phi_b)
        Q_prime = np.outer(a, b)
        Q_star = 0.5 * (Q_prime + Q_prime.T)
        Q_hat = Q_star - np.trace(Q_star) / 3
        quad_term = np.einsum('ij,jk,ik->i', data_positions, Q_hat, data_positions)
        if 1 in ell and 'v' in params:
            v = params['v']
            D = v * D_survey
            cos_angle = np.sum(ang2vec(params['theta'], params['phi'])
                               * data_positions, axis=1)
            expected = N * (1 + D * cos_angle + Q * quad_term)
        else:
            expected = N * (1 + Q * quad_term)

    if second_dipole is not None:
        if second_dipole['dir_vec'] is not None:
            sd_vec = second_dipole['dir_vec']
        else:
            sd_vec = ang2vec(params['theta_sd'], params['phi_sd'])
        if second_dipole['fix_v'] is not None:
            v_sd = second_dipole['fix_v']
        else:
            v_sd = params['v_sd']
        D_sd = v_sd * D_survey
        cos_angle_sd = np.sum(sd_vec * data_positions, axis=1)
        expected = expected + N * D_sd * cos_angle_sd

    for l_mode in sorted(ell):
        if l_mode > 2:
            for m in range(-l_mode, l_mode + 1):
                key = f'a_{l_mode}_{m}'
                if key in params:
                    theta_pix = np.arccos(data_positions[:, 2] /
                                          np.linalg.norm(data_positions, axis=1))
                    phi_pix = np.arctan2(data_positions[:, 1], data_positions[:, 0])
                    if m >= 0:
                        Ylm = sp.special.sph_harm(m, l_mode, phi_pix, theta_pix).real
                    else:
                        Ylm = sp.special.sph_harm(-m, l_mode, phi_pix, theta_pix).imag
                    expected += N * params[key] * Ylm

    return expected


def loglike_value(type, data_counts, expected, params=None):
    """Compute the log-likelihood value."""
    expected = np.clip(expected, 1e-10, None)

    if type == 'poisson':
        return np.sum(sp.stats.poisson.logpmf(data_counts.astype(int), expected))
    elif type == 'general_poisson':
        b = params.get('gp_dispersion', 0.0) if params else 0.0
        k = data_counts
        lam = expected
        term1 = np.log(np.clip(lam * (1 - b), 1e-10, None))
        term2 = (k - 1) * np.log(np.clip(lam * (1 - b) + k * b, 1e-10, None))
        term3 = gammaln(k + 1)
        term4 = lam * (1 - b)
        term5 = k * b
        return np.sum(term1 + term2 - term3 - term4 - term5)
    elif type == 'gaussian':
        std = np.std(data_counts)
        return np.sum(sp.stats.norm.logpdf(data_counts, expected, std))
    else:
        raise ValueError(f"Unknown likelihood type: {type}")


def make_ptform(param_names, priors):
    """Create a prior transform function for UltraNest."""
    def ptform(u):
        x = np.zeros(len(param_names))
        for i, p in enumerate(param_names):
            cfg = priors.get(p, {'type': 'uniform', 'low': 0.0, 'high': 1.0})
            if cfg['type'] == 'uniform':
                x[i] = cfg['low'] + u[i] * (cfg['high'] - cfg['low'])
            elif cfg['type'] == 'polar':
                x[i] = np.arccos(2 * u[i] - 1)
            else:
                x[i] = u[i]
        return x
    return ptform


# ----------------------------------------------------------------------
# Likelihood factories
# ----------------------------------------------------------------------
def make_loglike(type, ell, data_counts, data_positions, D_survey,
                 param_names, ecl_lat=None, cecl=None, second_dipole=None):
    """Create a log-likelihood function for single analysis."""
    def loglike(x):
        params = {p: x[i] for i, p in enumerate(param_names)}
        expected = compute_expected(params, data_positions, D_survey, ell,
                                    second_dipole=second_dipole)
        if ecl_lat is not None:
            bias = params.get('bias', 0.0)
            expected = expected * (1 - bias * cecl * np.abs(ecl_lat))
        return loglike_value(type, data_counts, expected, params)
    return loglike


def make_loglike_dict(type, ell, data_counts, data_positions, D_survey,
                     param_names, ecl_lat=None, cecl=None):
    """Create a log-likelihood function that takes a param dict."""
    def loglike(x_dict):
        params = {p: x_dict[p] for p in param_names}
        expected = compute_expected(params, data_positions, D_survey, ell)
        if ecl_lat is not None:
            bias = params.get('bias', 0.0)
            expected = expected * (1 - bias * cecl * np.abs(ecl_lat))
        return loglike_value(type, data_counts, expected, params)
    return loglike


def make_loglike_dict_joint(type, ell, data_counts, data_positions, D_survey,
                            params_base, params_mapped, need_convert, from_sys, to_sys,
                            ecl_lat=None, cecl=None):
    """Create a log-likelihood for the second model in a joint analysis."""
    def loglike(x_dict):
        params = {}
        for base, mapped in zip(params_base, params_mapped):
            params[base] = x_dict[mapped]

        if need_convert and 'theta' in params and 'phi' in params:
            theta_orig = params['theta']
            phi_orig = params['phi']
            theta_new, phi_new = convert_thetaphi(
                np.atleast_1d(theta_orig), np.atleast_1d(phi_orig),
                from_sys, to_sys
            )
            params['theta'] = theta_new[0]
            params['phi'] = phi_new[0]

        expected = compute_expected(params, data_positions, D_survey, ell)
        if ecl_lat is not None:
            bias = params.get('bias', 0.0)
            expected = expected * (1 - bias * cecl * np.abs(ecl_lat))
        return loglike_value(type, data_counts, expected, params)
    return loglike


# ----------------------------------------------------------------------
# Build (param_names, loglike, ptform) for single / legacy-joint / N-joint
# ----------------------------------------------------------------------
def build_single(analyser):
    """Build likelihood and prior for a single-dataset analysis."""
    config = analyser._model_config
    priors = analyser._priors_config
    param_names = list(config['param_names'])

    nside = hp.npix2nside(len(analyser._map))
    npix = len(analyser._map)
    mask = analyser._mask
    data_counts = analyser._map[mask]
    pos = np.array(hp.pix2vec(nside, np.arange(npix))).T
    data_positions = pos[mask]
    D_survey = analyser._D

    ecl_lat = None
    cecl = None
    if config.get('bias'):
        cecl = analyser._bias_cecl
        ecl_lat = compute_ecliptic_lat(nside, mask, analyser._map_coords)

    if config['type'] == 'custom':
        custom_like = config['likelihood']

        def loglike(x):
            return custom_like(x, data_counts, data_positions, D_survey)
    else:
        loglike = make_loglike(
            config['type'], config['ell'], data_counts, data_positions,
            D_survey, param_names, ecl_lat, cecl,
            second_dipole=config.get('second_dipole'),
        )

    ptform = make_ptform(param_names, priors)
    return param_names, loglike, ptform


def build_joint(analyser):
    """Build likelihood and prior for a joint two-dataset analysis."""
    config1 = analyser._model_config
    config2 = analyser._model2_config
    priors1 = analyser._priors_config
    priors2 = analyser._priors2_config

    params1 = list(config1['param_names'])
    params2_base = list(config2['param_names'])

    params2_mapped = []
    for p in params2_base:
        if p in analyser._shared_parameters:
            params2_mapped.append(p)
        else:
            params2_mapped.append(p + '2')

    combined_params = list(params1)
    for p in params2_mapped:
        if p not in combined_params:
            combined_params.append(p)

    combined_priors = {}
    for p in combined_params:
        if p in priors1:
            combined_priors[p] = priors1[p]
        elif p in priors2:
            combined_priors[p] = priors2[p]

    nside1 = hp.npix2nside(len(analyser._map))
    mask1 = analyser._mask
    counts1 = analyser._map[mask1]
    pos1 = np.array(hp.pix2vec(nside1, np.arange(len(analyser._map)))).T[mask1]

    nside2 = hp.npix2nside(len(analyser._map2))
    mask2 = analyser._mask2
    counts2 = analyser._map2[mask2]
    pos2 = np.array(hp.pix2vec(nside2, np.arange(len(analyser._map2)))).T[mask2]

    D1 = analyser._D
    D2 = analyser._D2

    need_coord_convert = (analyser._map_coords != analyser._map2_coords)
    from_sys = analyser._map_coords
    to_sys = analyser._map2_coords

    ecl_lat1 = None
    cecl1 = None
    if config1.get('bias'):
        cecl1 = analyser._bias_cecl
        ecl_lat1 = compute_ecliptic_lat(nside1, mask1, analyser._map_coords)

    ecl_lat2 = None
    cecl2 = None
    if config2.get('bias'):
        cecl2 = analyser._bias_cecl_2
        ecl_lat2 = compute_ecliptic_lat(nside2, mask2, analyser._map2_coords)

    if config1['type'] == 'custom':
        def loglike1_fn(x_dict):
            x = [x_dict[p] for p in params1]
            return config1['likelihood'](x, counts1, pos1, D1)
    else:
        loglike1_fn = make_loglike_dict(
            config1['type'], config1['ell'], counts1, pos1, D1, params1,
            ecl_lat1, cecl1
        )

    if config2['type'] == 'custom':
        def loglike2_fn(x_dict):
            x = [x_dict[p2] for p2 in params2_mapped]
            return config2['likelihood'](x, counts2, pos2, D2)
    else:
        loglike2_fn = make_loglike_dict_joint(
            config2['type'], config2['ell'], counts2, pos2, D2,
            params2_base, params2_mapped, need_coord_convert, from_sys, to_sys,
            ecl_lat2, cecl2
        )

    def loglike(x):
        x_dict = {p: x[i] for i, p in enumerate(combined_params)}
        return loglike1_fn(x_dict) + loglike2_fn(x_dict)

    ptform = make_ptform(combined_params, combined_priors)
    return combined_params, loglike, ptform


def build_joint_n(analyser):
    """Build likelihood and prior for N-way joint analysis (N >= 2 datasets).

    Handles both the compositional N=2 case (uses legacy '2' suffix for
    byte-for-byte chain compatibility) and the N>=3 case (uses '_name' suffix).
    """
    datasets = [analyser] + list(analyser._children.values())
    names = ['a1'] + list(analyser._children.keys())
    n_total = len(datasets)
    use_legacy_n2 = (n_total == 2)

    for d, dname in zip(datasets, names):
        if d._model_config is None:
            raise ValueError(f"Dataset {dname!r} has no model configured.")
        if d._priors_config is None:
            raise ValueError(f"Dataset {dname!r} has no priors configured.")
        if d._map is None:
            raise ValueError(f"Dataset {dname!r} has no map.")
        if d._D is None:
            raise ValueError(f"Dataset {dname!r} has D not set.")
        ell = d._model_config['ell']
        if any(l > 1 for l in ell):
            raise NotImplementedError(
                f"N-way joint analysis supports only ell=[0] or ell=[0,1]. "
                f"Dataset {dname!r} has ell={ell}. "
                "Quadrupole/higher support is planned for a follow-up PR."
            )
        if d._map_coords != analyser._map_coords:
            raise ValueError(
                f"Dataset {dname!r} has map_coords={d._map_coords!r} but "
                f"self has map_coords={analyser._map_coords!r}. "
                "Heterogeneous coordinate systems are not yet supported."
            )

    combined_params = []
    combined_priors = {}
    per_dataset_loglikes = []

    for i, (d, dname) in enumerate(zip(datasets, names)):
        config = d._model_config
        priors = d._priors_config
        params_base = list(config['param_names'])

        params_mapped = []
        for p in params_base:
            if p in analyser._shared_parameters:
                params_mapped.append(p)
            elif use_legacy_n2 and i == 0:
                params_mapped.append(p)
            elif use_legacy_n2 and i == 1:
                params_mapped.append(p + '2')
            else:
                params_mapped.append(f'{p}_{dname}')

        for pm in params_mapped:
            if pm not in combined_params:
                combined_params.append(pm)

        for p_base, p_mapped in zip(params_base, params_mapped):
            if p_mapped not in combined_priors:
                if p_base in analyser._shared_parameters:
                    combined_priors[p_mapped] = analyser._priors_config[p_base]
                else:
                    combined_priors[p_mapped] = priors.get(
                        p_base,
                        DEFAULT_PRIORS.get(
                            p_base,
                            {'type': 'uniform', 'low': 0.0, 'high': 1.0}
                        )
                    )

        nside = hp.npix2nside(len(d._map))
        mask = d._mask
        data_counts = d._map[mask]
        pos = np.array(hp.pix2vec(nside, np.arange(len(d._map)))).T
        data_positions = pos[mask]
        D_survey = d._D

        ecl_lat = None
        cecl = None
        if config.get('bias'):
            cecl = d._bias_cecl
            ecl_lat = compute_ecliptic_lat(nside, mask, d._map_coords)

        fn = make_loglike_dict_joint(
            config['type'], config['ell'], data_counts, data_positions,
            D_survey, params_base, params_mapped,
            need_convert=False, from_sys=analyser._map_coords,
            to_sys=analyser._map_coords, ecl_lat=ecl_lat, cecl=cecl,
        )
        per_dataset_loglikes.append(fn)

    def loglike(x):
        x_dict = {p: x[i] for i, p in enumerate(combined_params)}
        return sum(fn(x_dict) for fn in per_dataset_loglikes)

    ptform = make_ptform(combined_params, combined_priors)
    return combined_params, loglike, ptform


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def run_ultranest(analyser, savedir=None, name=None, min_num_live_points=400,
                  dlogz=0.5, frac_remain=0.01, step=False, step_nsteps=None,
                  seed=None, **sampler_kwargs):
    """Run nested sampling with UltraNest. Body of ``Analyser.ultranest(...)``."""
    import ultranest

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

    is_legacy_joint = analyser._map2 is not None

    if is_legacy_joint and analyser._D2 is None:
        raise ValueError("d2 must be set for joint analysis.")
    if is_legacy_joint and analyser._model2_config is None:
        raise ValueError("model2 must be configured for joint analysis.")

    if savedir is not None:
        analyser._savedir = savedir
        os.makedirs(savedir, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    if analyser._is_composite:
        param_names, loglike, ptform = build_joint_n(analyser)
    elif is_legacy_joint:
        param_names, loglike, ptform = build_joint(analyser)
    else:
        param_names, loglike, ptform = build_single(analyser)

    log_dir = savedir
    sampler = ultranest.ReactiveNestedSampler(
        param_names, loglike, ptform, log_dir=log_dir
    )

    if step:
        import ultranest.stepsampler as _ss
        nsteps = step_nsteps if step_nsteps is not None else 2 * len(param_names)
        sampler.stepsampler = _ss.SliceSampler(
            nsteps=nsteps,
            generate_direction=_ss.generate_mixture_random_direction,
        )

    run_kwargs = {
        'min_num_live_points': min_num_live_points,
        'dlogz': dlogz,
        'frac_remain': frac_remain,
    }
    run_kwargs.update(sampler_kwargs)
    sampler.run(**run_kwargs)

    if savedir is not None and name is not None:
        run1_path = os.path.join(savedir, 'run1')
        target_path = os.path.join(savedir, name)
        if os.path.exists(run1_path) and not os.path.exists(target_path):
            os.rename(run1_path, target_path)
            analyser._savedir = target_path
        elif os.path.exists(target_path):
            analyser._savedir = target_path
    elif savedir is not None:
        run1_path = os.path.join(savedir, 'run1')
        if os.path.exists(run1_path):
            analyser._savedir = run1_path
        else:
            analyser._savedir = savedir

    if analyser._savedir is not None:
        eq_path = os.path.join(analyser._savedir, 'chains',
                               'equal_weighted_post.txt')
        if os.path.exists(eq_path):
            with open(eq_path, 'r') as f:
                n_eq = sum(1 for _ in f) - 1  # minus header
            print(f"[dipoletools.ultranest] Wrote {n_eq} equal-weighted "
                  f"posterior samples to {eq_path}")

    return sampler.results
