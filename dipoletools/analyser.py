"""Analyser class: model setup, nested sampling, and smoothed map generation."""

import os
import numpy as np
import healpy as hp
import scipy as sp
import matplotlib.pyplot as plt
from scipy.special import gammaln

from ._utils import ang2vec, r2d, d2r, convert_thetaphi
from ._defaults import DEFAULT_PRIORS


class Analyser:
    """Set up models and run nested sampling on HEALPix count maps.

    Parameters
    ----------
    map : numpy.ndarray
        Full HEALPix count map (length npix). NaN = missing/masked pixels.
    mask : numpy.ndarray, optional
        Boolean mask (True=keep). If None, derived from non-NaN pixels in map.
    D : float, optional
        Expected kinematic dipole amplitude for this dataset. No default.
    map2 : numpy.ndarray, optional
        Second map for joint analysis.
    mask2 : numpy.ndarray, optional
        Mask for the second map.
    d2 : float, optional
        Dipole amplitude for the second dataset.
    map_coords : str
        Coordinate system of map. Default 'C'.
    map2_coords : str, optional
        Coordinate system of map2. If None, same as map_coords.
    """

    def __init__(self, map=None, mask=None, D=None, map2=None, mask2=None, d2=None,
                 map_coords='C', map2_coords=None,
                 # Accept old-style kwargs for backwards compat during transition
                 Map=None, Mask=None, Map2=None, Mask2=None, D2=None):
        # Support both old and new kwarg names
        if map is None and Map is not None:
            map = Map
        if mask is None and Mask is not None:
            mask = Mask
        if map2 is None and Map2 is not None:
            map2 = Map2
        if mask2 is None and Mask2 is not None:
            mask2 = Mask2
        if d2 is None and D2 is not None:
            d2 = D2

        # Accept MapMaker / MaskMaker objects directly
        from .mapmaker import MapMaker
        from .maskmaker import MaskMaker
        if isinstance(map, MapMaker):
            map = np.asarray(map.map)
        if isinstance(mask, MaskMaker):
            mask = np.asarray(mask.mask)
        if isinstance(map2, MapMaker):
            map2 = np.asarray(map2.map)
        if isinstance(mask2, MaskMaker):
            mask2 = np.asarray(mask2.mask)

        # Primary dataset
        self._map = np.asarray(map, dtype=float) if map is not None else None
        self._mask = None
        self._D = D
        self._map_coords = map_coords

        if self._map is not None:
            if mask is not None:
                self._mask = np.asarray(mask, dtype=bool)
            else:
                self._mask = ~np.isnan(self._map)

        # Second dataset (optional, for joint analysis)
        self._map2 = np.asarray(map2, dtype=float) if map2 is not None else None
        self._mask2 = None
        self._D2 = d2
        self._map2_coords = map2_coords if map2_coords else map_coords

        if self._map2 is not None:
            if mask2 is not None:
                self._mask2 = np.asarray(mask2, dtype=bool)
            else:
                self._mask2 = ~np.isnan(self._map2)

        # Model configuration
        self._model_config = None
        self._priors_config = None
        self._model2_config = None
        self._priors2_config = None
        self._shared_parameters = []

        # Bias coefficients (set by model/model2 when bias=True)
        self._bias_cecl = None
        self._bias_cecl_2 = None

        # Sampling results
        self._savedir = None
        self._smooth_map = None

    # ------------------------------------------------------------------
    # Properties for map, mask, D, map2, mask2, d2
    # ------------------------------------------------------------------
    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, value):
        self._map = np.asarray(value, dtype=float)
        if self._mask is None:
            self._mask = ~np.isnan(self._map)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = np.asarray(value, dtype=bool)

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, value):
        self._D = value

    @property
    def map2(self):
        return self._map2

    @map2.setter
    def map2(self, value):
        self._map2 = np.asarray(value, dtype=float)
        if self._mask2 is None:
            self._mask2 = ~np.isnan(self._map2)

    @property
    def mask2(self):
        return self._mask2

    @mask2.setter
    def mask2(self, value):
        self._mask2 = np.asarray(value, dtype=bool)

    @property
    def d2(self):
        return self._D2

    @d2.setter
    def d2(self, value):
        self._D2 = value

    @property
    def savedir(self):
        return self._savedir

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    def model(self, type='poisson', ell=None, bias=False, bias_cecl=9.15e-4,
              likelihood=None, param_names=None):
        """Configure the model for nested sampling.

        Parameters
        ----------
        type : str
            'poisson' (default), 'general_poisson', or 'gaussian'.
        ell : list of int, optional
            Multipole modes to fit. Default [0, 1] (monopole + dipole).
        bias : bool
            Whether to include an ecliptic latitude bias term. Default False.
            When True, adds a 'bias' parameter and multiplies the expected
            counts by ``(1 - bias * cecl * |ecl_lat|)``. Ecliptic latitudes
            are computed automatically from the map coordinate system.
        bias_cecl : float
            Ecliptic bias coefficient. Only used when bias=True.
            Default 9.15e-4 (S22 CatWISE sample). Use 7.4e-4 for the
            S21 CatWISE sample.
        likelihood : callable, optional
            Custom likelihood function. Overrides type/ell.
            Signature: loglike(params, data_counts, data_positions, D).
        param_names : list of str, optional
            Parameter names for custom likelihood. Required if likelihood is set.
        """
        if ell is None:
            ell = [0, 1]

        if likelihood is not None:
            if param_names is None:
                raise ValueError("Must provide param_names with a custom likelihood.")
            self._model_config = {
                'type': 'custom',
                'ell': ell,
                'bias': False,
                'likelihood': likelihood,
                'param_names': param_names,
            }
        else:
            if type not in ('poisson', 'general_poisson', 'gaussian'):
                raise ValueError(
                    f"Unknown model type '{type}'. "
                    "Choose 'poisson', 'general_poisson', or 'gaussian'."
                )
            param_names = self._params_from_ell(ell, type=type, bias=bias)
            self._model_config = {
                'type': type,
                'ell': ell,
                'bias': bias,
                'likelihood': None,
                'param_names': param_names,
            }

        # Store bias data
        if bias:
            self._bias_cecl = bias_cecl

        # Auto-setup priors if not already set
        if self._priors_config is None:
            self.priors()

        if self._model_config is not None:
            return self._model_summary(self._model_config)

    def model2(self, type='poisson', ell=None, bias=False, bias_cecl=9.15e-4,
               likelihood=None, param_names=None, shared_parameters=None):
        """Configure the second model for joint analysis.

        Parameters
        ----------
        type : str
            'poisson' (default), 'general_poisson', or 'gaussian'.
        ell : list of int, optional
            Multipole modes to fit. Default [0, 1].
        bias : bool
            Whether to include an ecliptic latitude bias term.
        bias_cecl : float
            Ecliptic bias coefficient. Only used when bias=True.
            Default 9.15e-4 (S22 CatWISE sample). Use 7.4e-4 for S21.
        likelihood : callable, optional
            Custom likelihood function.
        param_names : list of str, optional
            Parameter names for custom likelihood.
        shared_parameters : list of str, optional
            Parameter names (from model's namespace) to share between models.
        """
        if self._map2 is None:
            raise ValueError("No map2 loaded. Set map2 first.")

        if ell is None:
            ell = [0, 1]

        if likelihood is not None:
            if param_names is None:
                raise ValueError("Must provide param_names with a custom likelihood.")
            self._model2_config = {
                'type': 'custom',
                'ell': ell,
                'bias': False,
                'likelihood': likelihood,
                'param_names': param_names,
            }
        else:
            if type not in ('poisson', 'general_poisson', 'gaussian'):
                raise ValueError(
                    f"Unknown model type '{type}'. "
                    "Choose 'poisson', 'general_poisson', or 'gaussian'."
                )
            param_names = self._params_from_ell(ell, type=type, bias=bias)
            self._model2_config = {
                'type': type,
                'ell': ell,
                'bias': bias,
                'likelihood': None,
                'param_names': param_names,
            }

        # Store bias data for model 2
        if bias:
            self._bias_cecl_2 = bias_cecl

        if shared_parameters is not None:
            self._shared_parameters = shared_parameters

        # Auto-setup priors2
        self.priors2(shared_parameters=shared_parameters)

        if self._model2_config is not None:
            return self._model_summary(self._model2_config)

    @staticmethod
    def _params_from_ell(ell, type='poisson', bias=False):
        """Generate parameter names from ell modes, model type, and bias flag."""
        params = []
        if 1 in ell:
            params.extend(['v', 'theta', 'phi'])
        if 0 in ell:
            params.append('N')
        if 2 in ell:
            params.extend(['Q', 'theta_a', 'phi_a', 'theta_b', 'phi_b'])
        for l_mode in sorted(ell):
            if l_mode > 2:
                for m in range(-l_mode, l_mode + 1):
                    params.append(f'a_{l_mode}_{m}')
        if bias:
            params.append('bias')
        if type == 'general_poisson':
            params.append('gp_dispersion')
        return params

    @staticmethod
    def _model_summary(config):
        """Return a summary string of the model configuration."""
        summary = f"Model type: {config['type']}\n"
        summary += f"Ell modes: {config['ell']}\n"
        if config.get('bias'):
            summary += "Bias: ecliptic latitude\n"
        summary += f"Parameters: {config['param_names']}"
        return summary

    # ------------------------------------------------------------------
    # priors
    # ------------------------------------------------------------------
    def priors(self, shared_parameters=None, **kwargs):
        """Configure priors for the model parameters.

        With no arguments, returns a summary of current priors.
        Pass keyword arguments to set specific priors, e.g.:
            priors(v={'type': 'uniform', 'low': 0, 'high': 10})
            priors(v=[0, 10])  # shorthand for uniform

        Parameters
        ----------
        shared_parameters : list, optional
            Ignored for priors (used only in priors2).
        **kwargs : dict
            Prior settings for parameters.
        """
        if self._model_config is None:
            self.model()  # use defaults

        param_names = self._model_config['param_names']

        if self._priors_config is None:
            self._priors_config = {}

        # Set defaults for known parameters
        for p in param_names:
            if p not in self._priors_config:
                if p in DEFAULT_PRIORS:
                    self._priors_config[p] = DEFAULT_PRIORS[p].copy()
                else:
                    self._priors_config[p] = {'type': 'uniform', 'low': 0.0, 'high': 1.0}

        # Handle N auto-prior
        if 'N' in self._priors_config and self._priors_config['N']['type'] == 'auto':
            if self._map is not None and self._mask is not None:
                mean_counts = np.nanmean(self._map[self._mask])
                self._priors_config['N'] = {
                    'type': 'uniform',
                    'low': mean_counts * 0.9,
                    'high': mean_counts * 1.1,
                }

        # Apply user overrides
        for p, val in kwargs.items():
            if isinstance(val, (list, tuple)) and len(val) == 2:
                self._priors_config[p] = {'type': 'uniform', 'low': val[0], 'high': val[1]}
            elif isinstance(val, dict):
                self._priors_config[p] = val

        if not kwargs:
            return self._priors_summary(self._priors_config)

    def priors2(self, shared_parameters=None, **kwargs):
        """Configure priors for the second model (joint analysis).

        Parameter names get a '2' suffix unless shared.
        """
        if self._model2_config is None:
            raise ValueError("No model2 configured. Call model2() first.")

        base_params = self._model2_config['param_names']

        if shared_parameters is not None:
            self._shared_parameters = shared_parameters

        if self._priors2_config is None:
            self._priors2_config = {}

        for p in base_params:
            if p in self._shared_parameters:
                # Shared parameter: use same name and prior as model 1
                self._priors2_config[p] = self._priors_config.get(
                    p, DEFAULT_PRIORS.get(p, {'type': 'uniform', 'low': 0.0, 'high': 1.0})
                ).copy()
            else:
                p2 = p + '2'
                if p2 not in self._priors2_config:
                    if p in DEFAULT_PRIORS:
                        self._priors2_config[p2] = DEFAULT_PRIORS[p].copy()
                    else:
                        self._priors2_config[p2] = {
                            'type': 'uniform', 'low': 0.0, 'high': 1.0
                        }

        # Handle N2 auto-prior
        if 'N2' in self._priors2_config and self._priors2_config['N2']['type'] == 'auto':
            if self._map2 is not None and self._mask2 is not None:
                mean_counts = np.nanmean(self._map2[self._mask2])
                self._priors2_config['N2'] = {
                    'type': 'uniform',
                    'low': mean_counts * 0.9,
                    'high': mean_counts * 1.1,
                }

        # Apply user overrides
        for p, val in kwargs.items():
            if isinstance(val, (list, tuple)) and len(val) == 2:
                self._priors2_config[p] = {'type': 'uniform', 'low': val[0], 'high': val[1]}
            elif isinstance(val, dict):
                self._priors2_config[p] = val

        if not kwargs and shared_parameters is None:
            return self._priors_summary(self._priors2_config)

    @staticmethod
    def _priors_summary(config):
        """Return a summary string of prior configuration."""
        lines = []
        for p, cfg in config.items():
            if cfg['type'] == 'uniform':
                lines.append(f"  {p}: uniform [{cfg['low']}, {cfg['high']}]")
            elif cfg['type'] == 'polar':
                lines.append(f"  {p}: polar (isotropic on sphere)")
            elif cfg['type'] == 'auto':
                lines.append(f"  {p}: auto (±10% of mean)")
            else:
                lines.append(f"  {p}: {cfg}")
        return "Priors:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # ultranest
    # ------------------------------------------------------------------
    def ultranest(self, savedir=None, name=None, min_num_live_points=400,
                  dlogz=0.5, frac_remain=0.01, **sampler_kwargs):
        """Run nested sampling with UltraNest.

        Parameters
        ----------
        savedir : str, optional
            Directory to save chains. If None, no chains saved.
        name : str, optional
            Name for this run (subdirectory under savedir).
        min_num_live_points : int
            Minimum number of live points. Default 400.
        dlogz : float
            Target evidence accuracy. Default 0.5.
        frac_remain : float
            Fraction of remaining evidence. Default 0.01.
        **sampler_kwargs
            Additional kwargs passed to ReactiveNestedSampler.run().
        """
        import ultranest

        if self._map is None:
            raise ValueError("No map loaded.")
        if self._D is None:
            raise ValueError("D (dipole amplitude) must be set before sampling.")
        if self._model_config is None:
            self.model()

        is_joint = self._map2 is not None

        if is_joint and self._D2 is None:
            raise ValueError("d2 must be set for joint analysis.")
        if is_joint and self._model2_config is None:
            raise ValueError("model2 must be configured for joint analysis.")

        # Set up save directory
        if savedir is not None:
            self._savedir = savedir
            os.makedirs(savedir, exist_ok=True)

        # Build the likelihood and prior transform
        if is_joint:
            param_names, loglike, ptform = self._build_joint()
        else:
            param_names, loglike, ptform = self._build_single()

        log_dir = savedir
        sampler = ultranest.ReactiveNestedSampler(
            param_names, loglike, ptform, log_dir=log_dir
        )

        run_kwargs = {
            'min_num_live_points': min_num_live_points,
            'dlogz': dlogz,
            'frac_remain': frac_remain,
        }
        run_kwargs.update(sampler_kwargs)
        sampler.run(**run_kwargs)

        # Rename run directory if name is specified
        if savedir is not None and name is not None:
            run1_path = os.path.join(savedir, 'run1')
            target_path = os.path.join(savedir, name)
            if os.path.exists(run1_path) and not os.path.exists(target_path):
                os.rename(run1_path, target_path)
                self._savedir = target_path
            elif os.path.exists(target_path):
                self._savedir = target_path
        elif savedir is not None:
            run1_path = os.path.join(savedir, 'run1')
            if os.path.exists(run1_path):
                self._savedir = run1_path
            else:
                self._savedir = savedir

        return sampler.results

    # ------------------------------------------------------------------
    # Build likelihood + prior transform
    # ------------------------------------------------------------------
    def _build_single(self):
        """Build likelihood and prior for a single-dataset analysis."""
        config = self._model_config
        priors = self._priors_config
        param_names = list(config['param_names'])

        # Extract masked data and positions
        nside = hp.npix2nside(len(self._map))
        npix = len(self._map)
        mask = self._mask
        data_counts = self._map[mask]
        pos = np.array(hp.pix2vec(nside, np.arange(npix))).T
        data_positions = pos[mask]
        D_survey = self._D

        # Precompute ecliptic latitudes if bias is enabled
        ecl_lat = None
        cecl = None
        if config.get('bias'):
            cecl = self._bias_cecl
            ecl_lat = self._compute_ecliptic_lat(nside, mask, self._map_coords)

        if config['type'] == 'custom':
            custom_like = config['likelihood']

            def loglike(x):
                return custom_like(x, data_counts, data_positions, D_survey)
        else:
            loglike = self._make_loglike(
                config['type'], config['ell'], data_counts, data_positions,
                D_survey, param_names, ecl_lat, cecl
            )

        ptform = self._make_ptform(param_names, priors)

        return param_names, loglike, ptform

    def _build_joint(self):
        """Build likelihood and prior for a joint two-dataset analysis."""
        config1 = self._model_config
        config2 = self._model2_config
        priors1 = self._priors_config
        priors2 = self._priors2_config

        params1 = list(config1['param_names'])
        params2_base = list(config2['param_names'])

        # Build the combined parameter list
        # params2 with '2' suffix for non-shared, original name for shared
        params2_mapped = []
        for p in params2_base:
            if p in self._shared_parameters:
                params2_mapped.append(p)
            else:
                params2_mapped.append(p + '2')

        # Unique combined parameters (preserving order)
        combined_params = list(params1)
        for p in params2_mapped:
            if p not in combined_params:
                combined_params.append(p)

        # Combined priors
        combined_priors = {}
        for p in combined_params:
            if p in priors1:
                combined_priors[p] = priors1[p]
            elif p in priors2:
                combined_priors[p] = priors2[p]

        # Extract data for both datasets
        nside1 = hp.npix2nside(len(self._map))
        mask1 = self._mask
        counts1 = self._map[mask1]
        pos1 = np.array(hp.pix2vec(nside1, np.arange(len(self._map)))).T[mask1]

        nside2 = hp.npix2nside(len(self._map2))
        mask2 = self._mask2
        counts2 = self._map2[mask2]
        pos2 = np.array(hp.pix2vec(nside2, np.arange(len(self._map2)))).T[mask2]

        D1 = self._D
        D2 = self._D2

        # Check if coordinate systems differ
        need_coord_convert = (self._map_coords != self._map2_coords)
        from_sys = self._map_coords
        to_sys = self._map2_coords

        # Precompute ecliptic latitudes if bias is enabled on either model
        ecl_lat1 = None
        cecl1 = None
        if config1.get('bias'):
            cecl1 = self._bias_cecl
            ecl_lat1 = self._compute_ecliptic_lat(nside1, mask1, self._map_coords)

        ecl_lat2 = None
        cecl2 = None
        if config2.get('bias'):
            cecl2 = self._bias_cecl_2
            ecl_lat2 = self._compute_ecliptic_lat(nside2, mask2, self._map2_coords)

        # Build individual likelihoods
        if config1['type'] == 'custom':
            def loglike1_fn(x_dict):
                x = [x_dict[p] for p in params1]
                return config1['likelihood'](x, counts1, pos1, D1)
        else:
            loglike1_fn = self._make_loglike_dict(
                config1['type'], config1['ell'], counts1, pos1, D1, params1,
                ecl_lat1, cecl1
            )

        if config2['type'] == 'custom':
            def loglike2_fn(x_dict):
                x = [x_dict[p2] for p2 in params2_mapped]
                return config2['likelihood'](x, counts2, pos2, D2)
        else:
            loglike2_fn = self._make_loglike_dict_joint(
                config2['type'], config2['ell'], counts2, pos2, D2,
                params2_base, params2_mapped, need_coord_convert, from_sys, to_sys,
                ecl_lat2, cecl2
            )

        # Build the joint likelihood
        def loglike(x):
            x_dict = {p: x[i] for i, p in enumerate(combined_params)}
            return loglike1_fn(x_dict) + loglike2_fn(x_dict)

        ptform = self._make_ptform(combined_params, combined_priors)

        return combined_params, loglike, ptform

    # ------------------------------------------------------------------
    # Ecliptic latitude computation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_ecliptic_lat(nside, mask, map_coords):
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
            # Already ecliptic
            return lat_deg[mask]
        else:
            raise ValueError(f"Unknown coordinate system: {map_coords}")

        ecl = sc.transform_to('barycentricmeanecliptic')
        return ecl.lat.deg[mask]

    # ------------------------------------------------------------------
    # Likelihood factories
    # ------------------------------------------------------------------
    def _make_loglike(self, type, ell, data_counts, data_positions, D_survey,
                      param_names, ecl_lat=None, cecl=None):
        """Create a log-likelihood function for single analysis."""
        def loglike(x):
            params = {p: x[i] for i, p in enumerate(param_names)}
            expected = self._compute_expected(params, data_positions, D_survey, ell)
            if ecl_lat is not None:
                bias = params.get('bias', 0.0)
                expected = expected * (1 - bias * cecl * np.abs(ecl_lat))
            return self._loglike_value(type, data_counts, expected, params)
        return loglike

    def _make_loglike_dict(self, type, ell, data_counts, data_positions, D_survey,
                           param_names, ecl_lat=None, cecl=None):
        """Create a log-likelihood function that takes a param dict."""
        def loglike(x_dict):
            params = {p: x_dict[p] for p in param_names}
            expected = self._compute_expected(params, data_positions, D_survey, ell)
            if ecl_lat is not None:
                bias = params.get('bias', 0.0)
                expected = expected * (1 - bias * cecl * np.abs(ecl_lat))
            return self._loglike_value(type, data_counts, expected, params)
        return loglike

    def _make_loglike_dict_joint(self, type, ell, data_counts, data_positions, D_survey,
                                 params_base, params_mapped, need_convert, from_sys, to_sys,
                                 ecl_lat=None, cecl=None):
        """Create a log-likelihood for the second model in a joint analysis."""
        def loglike(x_dict):
            # Map combined param names back to model2's base param names
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

            expected = self._compute_expected(params, data_positions, D_survey, ell)
            if ecl_lat is not None:
                bias = params.get('bias', 0.0)
                expected = expected * (1 - bias * cecl * np.abs(ecl_lat))
            return self._loglike_value(type, data_counts, expected, params)
        return loglike

    # ------------------------------------------------------------------
    # Expected counts and log-likelihood evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_expected(params, data_positions, D_survey, ell):
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

        # Higher ell modes using spherical harmonics
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

    @staticmethod
    def _loglike_value(type, data_counts, expected, params=None):
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

    @staticmethod
    def _make_ptform(param_names, priors):
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

    # ------------------------------------------------------------------
    # show
    # ------------------------------------------------------------------
    def show(self, **kwargs):
        """Display the (masked) count map with healpy.projview.

        Masked pixels are shown as NaN (background). Default colormap: plasma.
        Default title: none. Any keyword arguments are forwarded to hp.projview.
        """
        if self._map is None:
            raise ValueError("No map loaded.")
        m = self._map.copy()
        if self._mask is not None:
            m[~self._mask] = np.nan
        kwargs.setdefault('cmap', 'plasma')
        kwargs.setdefault('title', '')
        hp.projview(m, **kwargs)
        plt.show()

    # ------------------------------------------------------------------
    # smooth
    # ------------------------------------------------------------------
    def smooth(self, steradians=1.0, cbar_units=None, plot=True, cmap='viridis'):
        """Generate a smoothed count map using a running average.

        Parameters
        ----------
        steradians : float
            Smoothing radius in steradians. Default 1.
        cbar_units : str, optional
            Label for the colorbar.
        plot : bool
            Whether to plot the result. Default True.
        cmap : str
            Colormap. Default 'viridis'.

        Returns
        -------
        numpy.ndarray
            Smoothed count values for unmasked pixels.
        """
        if self._map is None:
            raise ValueError("No map loaded.")

        from scipy.spatial import cKDTree

        counts = self._map.copy()
        nside = hp.npix2nside(len(counts))
        npix = len(counts)
        pos = np.array(hp.pix2vec(nside, np.arange(npix))).T

        radius = np.arccos(1 - steradians / (2 * np.pi))
        chord = 2 * np.sin(radius / 2)

        if self._mask is not None:
            unmasked_indices = np.where(self._mask)[0]
        else:
            unmasked_indices = np.arange(npix)

        is_unmasked = np.zeros(npix, dtype=bool)
        is_unmasked[unmasked_indices] = True

        tree = cKDTree(pos)
        neighbors_list = tree.query_ball_point(pos[unmasked_indices], chord,
                                               workers=-1)

        average_counts = np.zeros(len(unmasked_indices))
        for j, nbrs in enumerate(neighbors_list):
            nbrs_arr = np.asarray(nbrs, dtype=int)
            valid = nbrs_arr[is_unmasked[nbrs_arr]]
            if len(valid) > 0:
                average_counts[j] = counts[valid].mean()

        if plot:
            m = np.empty(npix)
            m[:] = np.nan
            m[unmasked_indices] = average_counts
            hp.mollview(m, min=np.min(average_counts), max=np.max(average_counts),
                        cmap=cmap, unit=cbar_units or '')
            hp.graticule()
            plt.show()

        self._smooth_map = average_counts
        return average_counts

    @property
    def smooth_map(self):
        """Return the smoothed map. If not yet computed, compute without plotting."""
        if self._smooth_map is None:
            self.smooth(plot=False)
        return self._smooth_map
