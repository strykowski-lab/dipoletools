"""Analyser class: model setup, nested sampling, and smoothed map generation."""

import copy
import inspect
import warnings

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from ._utils import (ang2vec, d2r, convert_lonlat, lonlat_names)
from ._defaults import DEFAULT_PRIORS


def _resolve_second_dipole(second_dipole, map_coords):
    """Validate a ``second_dipole`` spec and pre-compute the fixed-direction
    unit vector in ``map_coords``.

    Returns a dict with keys:
      - ``dir_vec`` : (3,) np.ndarray unit vector in map_coords, or None if
        the direction is a free parameter.
      - ``fix_v``   : float amplitude (in units of D_survey), or None if
        v_sd is a free parameter.
    Or None if second_dipole is falsy.
    """
    if not second_dipole:
        return None
    fix_direction = second_dipole.get('fix_direction')
    direction_coords = second_dipole.get('direction_coords', map_coords)
    fix_v = second_dipole.get('fix_v')

    dir_vec = None
    if fix_direction is not None:
        lon_deg, lat_deg = fix_direction
        # Convert from the user-specified coord system to the map's coord
        # system so the dot product with data_positions is consistent.
        lon_arr = np.atleast_1d(float(lon_deg))
        lat_arr = np.atleast_1d(float(lat_deg))
        new_lon, new_lat = convert_lonlat(lon_arr, lat_arr,
                                          direction_coords, map_coords)
        theta = d2r(90.0 - new_lat[0])
        phi = d2r(new_lon[0]) % (2 * np.pi)
        dir_vec = ang2vec(theta, phi)

    return {
        'dir_vec': dir_vec,
        'fix_v': None if fix_v is None else float(fix_v),
    }


def _introspect_name(obj, frame):
    """Return the first local variable name in *frame* that is *obj*, or None."""
    if frame is None:
        return None
    for var_name, val in frame.f_locals.items():
        if val is obj and not var_name.startswith('_'):
            return var_name
    return None


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
        # Detect legacy 2-dataset form before aliasing
        _using_legacy_2dataset = (
            map2 is not None or mask2 is not None or d2 is not None or
            Map2 is not None or Mask2 is not None or D2 is not None
        )

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

        if _using_legacy_2dataset:
            warnings.warn(
                "The Map2/map2 form of Analyser is legacy. Prefer composing "
                "single-dataset Analysers with a1.add(a2). See README.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Accept MapMaker / MaskMaker objects directly
        from .mapmaker import MapMaker
        from .maskmaker import MaskMaker
        self._mapmaker = map if isinstance(map, MapMaker) else None
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

        # Compositional N-way analysis
        self._children: dict = {}          # ordered dict of child Analysers
        self._is_composite: bool = False

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
              likelihood=None, param_names=None, second_dipole=None):
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
        second_dipole : dict, optional
            Add a second dipole component to the model. Supports three
            modes: fix direction only (new free v_sd), fix amplitude only
            (new free theta_sd, phi_sd), or fix both (constant offset).
            Keys:
              - ``fix_direction`` : (lon_deg, lat_deg) tuple or None (free).
              - ``direction_coords`` : 'C'/'G'/'E'. Coord system of
                ``fix_direction`` or of the prior. Default = map_coords.
                When the direction is fixed, the vector is converted into
                the map's coord system internally.
              - ``fix_v`` : float or None (free). If a float, the secondary
                dipole amplitude is fixed at ``fix_v * D``.
            Ignored for custom likelihoods.
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
            sd_resolved = _resolve_second_dipole(second_dipole, self._map_coords)
            param_names = self._params_from_ell(ell, type=type, bias=bias,
                                                second_dipole=sd_resolved)
            self._model_config = {
                'type': type,
                'ell': ell,
                'bias': bias,
                'likelihood': None,
                'param_names': param_names,
                'second_dipole': sd_resolved,
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
    def _params_from_ell(ell, type='poisson', bias=False, second_dipole=None):
        """Generate parameter names from ell modes, model type, and bias flag.

        If ``second_dipole`` is provided (a dict from _resolve_second_dipole),
        appends v_sd (if v is free), theta_sd/phi_sd (if direction is free).
        """
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
        if second_dipole is not None:
            if second_dipole['fix_v'] is None:
                params.append('v_sd')
            if second_dipole['dir_vec'] is None:
                params.extend(['theta_sd', 'phi_sd'])
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
    # expected_amplitude
    # ------------------------------------------------------------------
    def expected_amplitude(self, alpha=0.75, alpha_std=0.5, n_mc=100,
                           fluxcut=None, cutoff=None,
                           flux_label=None, flux_err_label=None,
                           catalogue=None, return_std=False, plot=True,
                           seed=None):
        """Estimate the expected kinematic dipole amplitude from the catalogue.

        Two-stage flow: with ``cutoff=None`` (default) runs 5 quick MC
        passes and shows an inspection plot of amplitude vs flux limit so
        the user can pick a sensible lower cutoff. Re-call with a
        numerical ``cutoff`` (mJy offset above the flux cut) to run the
        full MC and get a value.

        Sources surviving any cuts/crossmatch on the MapMaker catalogue
        and falling in unmasked HEALPix pixels of the Analyser map are
        used.

        Parameters
        ----------
        alpha, alpha_std : float
            Mean and stddev of the per-source spectral-index distribution.
        n_mc : int
            Number of MC iterations in full mode.
        fluxcut : float, optional
            Flux threshold (same units as catalogue flux). Defaults to
            the most recent ``min`` cut on the flux column.
        cutoff : float, optional
            Lower bound (above ``fluxcut``) where the linear fit starts.
            If None, runs the inspection-plot pass.
        flux_label, flux_err_label : str, optional
            Override the label keys ('flux', 'flux_err') used to look up
            the column names in the MapMaker labels dict.
        catalogue : str, optional
            Name of the MapMaker catalogue. Defaults to the first one.
        return_std : bool
            If True (and cutoff is set), return ``(mean, std)``.
        plot : bool
            If True (default) and cutoff is set, show a single-trial
            amp-vs-flux plot with the linear fit and the recovered
            intercept.
        seed : int, optional
            RNG seed for reproducibility.
        """
        import scipy.constants as _sc
        if self._mapmaker is None:
            raise ValueError(
                "expected_amplitude requires the Analyser to be built from "
                "a MapMaker (so the catalogue is accessible)."
            )

        mm = self._mapmaker
        cat_name = catalogue if catalogue is not None else mm._catalogue_order[0]
        if cat_name not in mm._catalogues:
            raise ValueError(f"No catalogue named '{cat_name}'.")

        cat = mm._catalogues[cat_name]
        labels = mm._labels[cat_name]

        flbl = flux_label if flux_label is not None else 'flux'
        elbl = flux_err_label if flux_err_label is not None else 'flux_err'
        flux_col = labels.get(flbl, flbl)
        flux_err_col = labels.get(elbl, elbl)
        if flux_col not in cat.columns:
            raise ValueError(f"Flux column '{flux_col}' not in catalogue.")
        if flux_err_col not in cat.columns:
            raise ValueError(
                f"Flux-error column '{flux_err_col}' not in catalogue."
            )

        # Resolve fluxcut from the MapMaker cuts log if not supplied.
        if fluxcut is None:
            log = mm._cuts_log.get(cat_name, [])
            for entry in reversed(log):
                if entry['col'] == flux_col and entry['min'] is not None:
                    fluxcut = float(entry['min'])
                    break
            if fluxcut is None:
                raise ValueError(
                    "Could not infer fluxcut: no min cut on the flux column "
                    "is recorded on the MapMaker. Pass fluxcut= explicitly."
                )

        # Restrict to sources falling in unmasked pixels of self._map.
        if self._map is None:
            raise ValueError("No map loaded on the Analyser.")
        nside = hp.npix2nside(len(self._map))
        cat_lon, cat_lat = self._catalogue_lonlat(cat, labels, mm)
        # Convert catalogue coords to map coords if needed.
        cat_sys = mm._coord_system
        map_sys = self._map_coords
        if cat_sys != map_sys:
            cat_lon, cat_lat = convert_lonlat(cat_lon, cat_lat,
                                              cat_sys, map_sys)
        pix = hp.ang2pix(nside, cat_lon, cat_lat, lonlat=True)
        mask = self._mask
        if mask is not None:
            keep = mask[pix]
        else:
            keep = np.ones(len(cat), dtype=bool)

        flux_arr = np.asarray(cat[flux_col], dtype=float)[keep]
        ferr_arr = np.asarray(cat[flux_err_col], dtype=float)[keep]
        if len(flux_arr) == 0:
            raise ValueError("No catalogue sources survived the mask.")

        v_cmb = 369.83e3
        beta = v_cmb / _sc.c
        delta = (1 + beta) / np.sqrt(1 - beta**2)

        rng = np.random.default_rng(seed)

        def _one_pass():
            jitter = rng.normal(flux_arr, ferr_arr)
            sp_idx = rng.normal(alpha, alpha_std, size=len(jitter))
            boosted = jitter * delta**(1 + sp_idx)
            Fs_full = np.linspace(fluxcut, fluxcut + 21, 100)
            return jitter, boosted, Fs_full

        if cutoff is None:
            # Inspection mode: 5 quick passes, plot only.
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for _ in range(5):
                jitter, boosted, Fs_full = _one_pass()
                amps = np.empty(len(Fs_full))
                for j, F in enumerate(Fs_full):
                    Ni = np.sum(jitter > F)
                    if Ni == 0:
                        amps[j] = np.nan
                        continue
                    Nb = np.sum(boosted > F)
                    amps[j] = (Nb * delta**2 - Ni) / Ni
                ax.plot(Fs_full, amps, alpha=0.5)
            x_lo = np.ceil(Fs_full[0])
            x_hi = np.floor(Fs_full[-1])
            for x in np.arange(x_lo, x_hi + 0.5, 1.0):
                ax.axvline(x, ls='--', color='grey', alpha=0.3, lw=0.8)
            ax.set_xlabel(r'Flux limit $S_0$')
            ax.set_ylabel(r'Expected amplitude $\mathcal{D}_\mathrm{CMB}$')
            ax.set_title('What should the lower cutoff flux be?')
            plt.show()
            return None

        # Full mode.
        amplitudes = np.empty(n_mc)
        # Stash one trial for plotting.
        trial_idx = int(rng.integers(0, n_mc))
        trial_Fs = trial_amps = trial_slope = trial_intercept = None
        for i in range(n_mc):
            jitter, boosted, Fs_full = _one_pass()
            Fs = Fs_full[Fs_full > fluxcut + cutoff]
            amps = np.empty(len(Fs))
            for j, F in enumerate(Fs):
                Ni = np.sum(jitter > F)
                Nb = np.sum(boosted > F)
                amps[j] = (Nb * delta**2 - Ni) / Ni
            slope, intercept = np.polyfit(Fs, amps, 1)
            amplitudes[i] = slope * fluxcut + intercept
            if i == trial_idx:
                # Compute amp curve over the full flux range for plotting.
                amps_full = np.empty(len(Fs_full))
                for j, F in enumerate(Fs_full):
                    Ni = np.sum(jitter > F)
                    if Ni == 0:
                        amps_full[j] = np.nan
                        continue
                    Nb = np.sum(boosted > F)
                    amps_full[j] = (Nb * delta**2 - Ni) / Ni
                trial_Fs = Fs_full
                trial_amps = amps_full
                trial_slope = slope
                trial_intercept = intercept

        mean_amp = float(np.mean(amplitudes))
        std_amp = float(np.std(amplitudes))

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(trial_Fs, trial_amps, color='C0', alpha=0.7)
            xs = np.array([fluxcut, trial_Fs[-1]])
            ax.plot(xs, trial_slope * xs + trial_intercept,
                    color='black', ls='--')
            ax.axvline(fluxcut, ls='--', color='grey', alpha=0.5)
            ax.set_xlabel(r'Flux limit $S_0$')
            ax.set_ylabel(r'Expected amplitude $\mathcal{D}_\mathrm{CMB}$')
            plt.show()

        if self._D is None:
            self._D = mean_amp
        if return_std:
            return mean_amp, std_amp
        return mean_amp

    @staticmethod
    def _catalogue_lonlat(cat, labels, mm):
        """Return (lon, lat) in degrees for the catalogue, in mm._coord_system."""
        lon_name, lat_name = lonlat_names(mm._coord_system)
        lon_col = labels.get(lon_name, lon_name)
        lat_col = labels.get(lat_name, lat_name)
        if lon_col not in cat.columns:
            for try_lon, try_lat in [('ra', 'dec'), ('l', 'b')]:
                t_lon = labels.get(try_lon, try_lon)
                t_lat = labels.get(try_lat, try_lat)
                if t_lon in cat.columns and t_lat in cat.columns:
                    lon_col, lat_col = t_lon, t_lat
                    break
        return (np.asarray(cat[lon_col], dtype=float),
                np.asarray(cat[lat_col], dtype=float))

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
        shared_parameters : list of str, optional
            Parameter names shared across datasets. For the compositional
            N-way path this replaces calling shared(); for the legacy
            two-dataset path it is equivalent to the shared_parameters
            argument of model2(). Re-runs prior reconciliation on any
            existing children. Defaults to ['v', 'theta', 'phi'] once
            the first child is added.
        **kwargs : dict
            Prior settings for parameters.
        """
        shared_params = shared_parameters
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

        # Update shared parameter list (works for both legacy and N-joint)
        if shared_params is not None:
            self._shared_parameters = list(shared_params)
            for cname, child in self._children.items():
                self._reconcile_shared_priors(child, cname, stacklevel=2)

        if not kwargs and shared_params is None:
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

    # ------------------------------------------------------------------
    # Compositional N-way API
    # ------------------------------------------------------------------
    def add(self, other: "Analyser", name: str | None = None) -> "Analyser":
        """Add a child Analyser to form a joint composite.

        Parameters
        ----------
        other : Analyser
            Single-dataset Analyser to add.
        name : str, optional
            Name for this child. Defaults to the variable name of *other* at
            the call site (introspected). Falls back to 'analyser_N' with a
            UserWarning if introspection fails.

        Returns
        -------
        self
        """
        if other is self:
            raise ValueError("Cannot add an Analyser to itself.")
        if self._map2 is not None:
            raise ValueError(
                "Cannot use add() on a legacy 2-dataset Analyser (constructed with "
                "Map2=). Use the compositional form instead."
            )
        for existing_name, existing_child in self._children.items():
            if existing_child is other:
                raise ValueError(
                    f"This Analyser is already registered as {existing_name!r}."
                )

        frame = inspect.currentframe().f_back

        # Handle nested composition: flatten with a warning
        if other._is_composite:
            warnings.warn(
                "The Analyser being added is itself composite. Flattening: "
                "its children will be added as siblings.",
                UserWarning,
                stacklevel=2,
            )
            other_name = name or _introspect_name(other, frame) or \
                f'analyser_{len(self._children) + 1}'
            if other_name in self._children:
                raise ValueError(f"Name {other_name!r} is already used.")
            # Add a shallow copy of other (without its children) as a sibling
            other_snap = copy.copy(other)
            other_snap._children = {}
            other_snap._is_composite = False
            self._children[other_name] = other_snap
            self._is_composite = True
            # Absorb other's children as flat siblings
            for child_name, child in other._children.items():
                if child_name in self._children:
                    raise ValueError(
                        f"Name collision during flattening: {child_name!r} "
                        "already exists in self."
                    )
                self._children[child_name] = child
            return self

        # Normal single-dataset add
        if name is None:
            name = _introspect_name(other, frame)
            if name is None:
                name = f'analyser_{len(self._children) + 1}'
                warnings.warn(
                    f"Could not introspect a variable name for the added Analyser. "
                    f"Registered as {name!r}. Pass name= explicitly to suppress this.",
                    UserWarning,
                    stacklevel=2,
                )
        if name in self._children:
            raise ValueError(f"Name {name!r} is already used.")

        if other._map_coords != self._map_coords:
            raise ValueError(
                f"Cannot add analyser with map_coords={other._map_coords!r}; "
                f"self has map_coords={self._map_coords!r}. "
                "Heterogeneous coordinate systems are not yet supported."
            )

        # Set default shared params on first add
        if not self._is_composite and not self._shared_parameters:
            self._shared_parameters = ['v', 'theta', 'phi']
        self._reconcile_shared_priors(other, name, stacklevel=2)
        self._children[name] = other
        self._is_composite = True
        return self

    def access(self, name: str) -> "Analyser":
        """Return the child Analyser registered under *name*.

        Mutations on the returned object propagate into the composite.

        Raises
        ------
        KeyError
            If *name* is not found. The error message lists available names.
        """
        if name not in self._children:
            available = list(self._children.keys())
            raise KeyError(
                f"No child named {name!r}. Available: {available}"
            )
        return self._children[name]

    def remove(self, name: str) -> "Analyser":
        """Remove a child Analyser by name.

        If this was the last child, the composite reverts to single-dataset mode.

        Raises
        ------
        KeyError
            If *name* is not found.
        """
        if name not in self._children:
            available = list(self._children.keys())
            raise KeyError(
                f"No child named {name!r}. Available: {available}"
            )
        del self._children[name]
        if len(self._children) == 0:
            self._is_composite = False
        return self

    def _reconcile_shared_priors(self, other: "Analyser", name: str,
                                  stacklevel: int = 2) -> None:
        """Warn and overwrite child priors for any shared param that differs from self."""
        if self._priors_config is None or other._priors_config is None:
            return
        for p in self._shared_parameters:
            if p not in other._priors_config:
                continue
            self_prior = self._priors_config.get(p)
            other_prior = other._priors_config.get(p)
            if self_prior is not None and other_prior is not None \
                    and self_prior != other_prior:
                warnings.warn(
                    f"Shared prior {p!r} on {name!r} differs from a1's prior; "
                    f"defaulting to a1's prior.",
                    UserWarning,
                    stacklevel=stacklevel + 1,
                )
                other._priors_config[p] = copy.deepcopy(self_prior)

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
    # Sampler dispatch
    # ------------------------------------------------------------------
    def ultranest(self, savedir=None, name=None, min_num_live_points=400,
                  dlogz=0.5, frac_remain=0.01, step=False, step_nsteps=None,
                  seed=None, **sampler_kwargs):
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
        step : bool
            If True, attach a ``ultranest.stepsampler.SliceSampler`` to the
            ReactiveNestedSampler. Strongly recommended for models with
            more than ~5 parameters.
        step_nsteps : int, optional
            Slice steps per new live point. Default ``2 * len(param_names)``.
        seed : int, optional
            Random seed for reproducibility.
        **sampler_kwargs
            Additional kwargs forwarded to ReactiveNestedSampler.run().
        """
        from . import _ultranest as _un
        return _un.run_ultranest(
            self, savedir=savedir, name=name,
            min_num_live_points=min_num_live_points, dlogz=dlogz,
            frac_remain=frac_remain, step=step, step_nsteps=step_nsteps,
            seed=seed, **sampler_kwargs,
        )

    def blackjax(self, savedir=None, name=None, seed=0,
                 n_live=500, n_delete=50, num_mcmc_steps=None,
                 dlogz=0.5, max_iterations=10000):
        """Run GPU-accelerated nested sampling with BlackJAX.

        Reduced scope compared to ``.ultranest(...)``: supports gaussian,
        poisson, general_poisson, and the ecliptic-bias correction; does not
        yet support ell>=2 or a second/additional dipole, and forces
        ``v, theta, phi`` shared (everything else unshared) in joint
        analyses. Use ``.ultranest(...)`` for those configurations.
        """
        from . import _blackjax as _bj
        return _bj.run_blackjax(
            self, savedir=savedir, name=name, seed=seed,
            n_live=n_live, n_delete=n_delete, num_mcmc_steps=num_mcmc_steps,
            dlogz=dlogz, max_iterations=max_iterations,
        )

    # Method shims so existing tests / external callers can keep doing
    # ``analyser._build_joint_n()`` etc. The implementations live in
    # ``_ultranest.py``.
    def _build_single(self):
        from . import _ultranest as _un
        return _un.build_single(self)

    def _build_joint(self):
        from . import _ultranest as _un
        return _un.build_joint(self)

    def _build_joint_n(self):
        from . import _ultranest as _un
        return _un.build_joint_n(self)

    def _make_loglike_dict_joint(self, *args, **kwargs):
        from . import _ultranest as _un
        return _un.make_loglike_dict_joint(*args, **kwargs)

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
