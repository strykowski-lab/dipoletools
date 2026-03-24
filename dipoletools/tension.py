"""Tension class: compute Bayesian tension statistics between datasets."""

import os
import numpy as np
import scipy as sp
from scipy.special import gammaincc

from ._utils import format_value_error, quantile_1sigma, r2d
from ._defaults import get_label


class Tension:
    """Compute tension statistics between two datasets from their nested sampling chains.

    Parameters
    ----------
    A : Analyser, Posterior, or str
        First dataset (run A).
    B : Analyser, Posterior, or str
        Second dataset (run B).
    AB : Analyser, Posterior, or str, optional
        Joint run. If not provided, inferred from A+B or B+A directory names.
    coords : str
        Coordinate system for the table function. Default 'C'.
    """

    def __init__(self, A, B, AB=None, coords='C'):
        from .posterior import Posterior

        self._coord_system = coords

        # Convert each input to a Posterior object
        self._post_A = self._to_posterior(A)
        self._post_B = self._to_posterior(B)

        if AB is not None:
            self._post_AB = self._to_posterior(AB)
        else:
            self._post_AB = self._infer_joint(self._post_A, self._post_B)

        # Compute or load tension statistics
        self._logR = None
        self._logR_err = None
        self._logI = None
        self._logI_err = None
        self._logS = None
        self._logS_err = None
        self._d = None
        self._d_err = None
        self._p = None
        self._p_err = None
        self._sigma = None
        self._sigma_err = None

        self._init_tension()

    @staticmethod
    def _to_posterior(obj):
        """Convert an input to a Posterior object."""
        from .posterior import Posterior
        from .analyser import Analyser

        if isinstance(obj, Posterior):
            return obj
        elif isinstance(obj, Analyser):
            return Posterior(obj)
        elif isinstance(obj, str):
            return Posterior(obj)
        else:
            raise ValueError(
                f"Expected Analyser, Posterior, or savedir string, got {type(obj)}"
            )

    @staticmethod
    def _infer_joint(post_A, post_B):
        """Try to find the joint run directory from A and B savedirs."""
        from .posterior import Posterior

        dir_A = post_A.savedir
        dir_B = post_B.savedir

        # Get the parent directory and run names
        parent_A = os.path.dirname(dir_A)
        parent_B = os.path.dirname(dir_B)
        name_A = os.path.basename(dir_A)
        name_B = os.path.basename(dir_B)

        # Try both orderings in the same parent directory
        for parent in [parent_A, parent_B]:
            for joint_name in [f'{name_A}+{name_B}', f'{name_B}+{name_A}']:
                joint_dir = os.path.join(parent, joint_name)
                if os.path.exists(joint_dir):
                    try:
                        return Posterior(joint_dir)
                    except Exception:
                        continue

        raise FileNotFoundError(
            f"Cannot find joint run. Tried:\n"
            f"  {os.path.join(parent_A, name_A + '+' + name_B)}\n"
            f"  {os.path.join(parent_A, name_B + '+' + name_A)}\n"
            "Pass the joint run explicitly as the third argument."
        )

    def _init_tension(self):
        """Compute or load tension statistics."""
        save_file = os.path.join(self._post_AB.savedir, 'tension_stats.npz')

        if os.path.exists(save_file):
            loaded = np.load(save_file)
            self._logR = float(loaded['logR'])
            self._logR_err = loaded['logR_err']
            self._logI = float(loaded['logI'])
            self._logI_err = loaded['logI_err']
            self._logS = float(loaded['logS'])
            self._logS_err = loaded['logS_err']
            self._d = float(loaded['d'])
            self._d_err = loaded['d_err']
            self._p = float(loaded['p'])
            self._p_err = loaded['p_err']
            self._sigma = float(loaded['sigma'])
            self._sigma_err = loaded['sigma_err']
            return

        # Compute from the three Posterior objects
        logZ_A = self._post_A.logZ
        logZ_A_err = self._post_A.logZ_err
        logZ_B = self._post_B.logZ
        logZ_B_err = self._post_B.logZ_err
        logZ_AB = self._post_AB.logZ
        logZ_AB_err = self._post_AB.logZ_err

        kl_A = self._post_A.kl
        kl_A_err = self._post_A.kl_err
        kl_B = self._post_B.kl
        kl_B_err = self._post_B.kl_err
        kl_AB = self._post_AB.kl
        kl_AB_err = self._post_AB.kl_err

        d_A = self._post_A.d
        d_A_err = self._post_A.d_err
        d_B = self._post_B.d
        d_B_err = self._post_B.d_err
        d_AB = self._post_AB.d
        d_AB_err = self._post_AB.d_err

        # logR = logZ_AB - logZ_A - logZ_B
        self._logR = logZ_AB - logZ_A - logZ_B
        self._logR_err = np.sqrt(logZ_A_err**2 + logZ_B_err**2 + logZ_AB_err**2)

        # logI = KL_A + KL_B - KL_AB
        self._logI = kl_A + kl_B - kl_AB
        self._logI_err = np.sqrt(kl_A_err**2 + kl_B_err**2 + kl_AB_err**2)

        # logS = logR - logI
        self._logS = self._logR - self._logI
        self._logS_err = np.sqrt(self._logR_err**2 + self._logI_err**2)

        # d = d_A + d_B - d_AB
        self._d = d_A + d_B - d_AB
        self._d_err = np.sqrt(d_A_err**2 + d_B_err**2 + d_AB_err**2)

        # p-value and sigma
        self._p, self._p_err = self._calc_p(
            self._d, self._d_err, self._logS, self._logS_err
        )
        self._sigma, self._sigma_err = self._calc_sigma(self._p, self._p_err)

        # Save
        try:
            np.savez(save_file,
                     logR=self._logR, logR_err=self._logR_err,
                     logI=self._logI, logI_err=self._logI_err,
                     logS=self._logS, logS_err=self._logS_err,
                     d=self._d, d_err=self._d_err,
                     p=self._p, p_err=self._p_err,
                     sigma=self._sigma, sigma_err=self._sigma_err)
        except Exception as e:
            print(f"Warning: Could not save tension stats: {e}")

    @staticmethod
    def _calc_p(d, d_err, logS, logS_err):
        """Compute the tension p-value from d and logS.

        p = gammaincc(d/2, (d - 2*logS)/2)
        Errors from extreme values of d±derr, logS±logSerr.
        """
        def _p(d_val, logS_val):
            if d_val <= 0:
                return 1.0
            arg1 = d_val / 2
            arg2 = (d_val - 2 * logS_val) / 2
            if arg2 < 0:
                return 1.0
            return gammaincc(arg1, arg2)

        d_err_plus, d_err_minus = d_err
        logS_err_plus, logS_err_minus = logS_err

        p_median = _p(d, logS)

        p_extremes = [
            _p(d + d_err_plus, logS + logS_err_plus),
            _p(d + d_err_plus, logS - logS_err_minus),
            _p(d - d_err_minus, logS + logS_err_plus),
            _p(d - d_err_minus, logS - logS_err_minus),
        ]

        p_upper = max(p_extremes) - p_median
        p_lower = p_median - min(p_extremes)

        return p_median, np.array([p_upper, p_lower])

    @staticmethod
    def _calc_sigma(p, p_err):
        """Convert p-value to sigma tension level.

        sigma = -ndtri(p/2) where ndtri is the inverse normal CDF.
        """
        def _sigma(p_val):
            if p_val <= 0:
                return np.inf
            if p_val >= 1:
                return 0.0
            log_half_p = np.log(p_val) - np.log(2)
            return -sp.special.ndtri(np.exp(log_half_p))

        sigma_median = _sigma(p)

        # For error propagation, compute sigma at extreme p values
        p_upper, p_lower = p_err
        sigma_at_low_p = _sigma(p - p_lower)  # lower p → higher sigma
        sigma_at_high_p = _sigma(p + p_upper)  # higher p → lower sigma

        sigma_err_plus = sigma_at_low_p - sigma_median
        sigma_err_minus = sigma_median - sigma_at_high_p

        return sigma_median, np.array([max(0, sigma_err_plus), max(0, sigma_err_minus)])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def logR(self):
        return self._logR

    @property
    def logR_err(self):
        return self._logR_err

    @property
    def logI(self):
        return self._logI

    @property
    def logI_err(self):
        return self._logI_err

    @property
    def logS(self):
        return self._logS

    @property
    def logS_err(self):
        return self._logS_err

    @property
    def d(self):
        return self._d

    @property
    def d_err(self):
        return self._d_err

    @property
    def p(self):
        return self._p

    @property
    def p_err(self):
        return self._p_err

    @property
    def sigma(self):
        return self._sigma

    @property
    def sigma_err(self):
        return self._sigma_err

    # ------------------------------------------------------------------
    # coords
    # ------------------------------------------------------------------
    def coords(self, *args):
        """Query or set the coordinate system for the table function."""
        if len(args) == 0:
            return self._coord_system
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (list, tuple)):
                self._coord_system = arg[-1]
            else:
                self._coord_system = arg
        elif len(args) == 2:
            self._coord_system = args[1]

    # ------------------------------------------------------------------
    # table
    # ------------------------------------------------------------------
    def table(self, parameters=None, decimals=None, velocity_scale=369.82):
        """Generate a formatted table row for the joint analysis.

        Loads the joint chains (AB) and computes posterior summaries,
        then appends tension statistics.

        Parameters
        ----------
        parameters : list of str, optional
            Parameter names to include, in order. Tension stats (logR, logI,
            logS, d, sigma, p) are also valid parameter names.
        decimals : int or list of int, optional
            Number of decimal places. If None, auto-determined from errors.
        velocity_scale : float
            Factor to convert v to km/s. Default 369.82.

        Returns
        -------
        str
            Formatted LaTeX table row.
        """
        # Get chain entries from the joint posterior
        all_entries = self._prepare_table_entries(velocity_scale)

        # Determine order
        if parameters is not None:
            ordered_keys = parameters
        else:
            ordered_keys = self._default_table_order()

        ordered_keys = [k for k in ordered_keys if k in all_entries]

        # Handle decimals
        if isinstance(decimals, (list, tuple)):
            if len(decimals) != len(ordered_keys):
                raise ValueError(
                    f"decimals list length ({len(decimals)}) must match "
                    f"number of table entries ({len(ordered_keys)})."
                )
        elif isinstance(decimals, int):
            decimals = [decimals] * len(ordered_keys)

        parts = []
        for i, key in enumerate(ordered_keys):
            med, err = all_entries[key]
            dec = decimals[i] if decimals is not None else None
            parts.append(format_value_error(med, err[0], err[1], dec))

        return ' & '.join(parts)

    def _prepare_table_entries(self, velocity_scale):
        """Compute all table entries: chain posteriors + tension stats."""
        entries = {}

        # Chain posteriors from the joint run
        post = self._post_AB
        for i, p in enumerate(post._param_names):
            samples = post._samples[:, i].copy()

            if p == 'v':
                samples *= velocity_scale

            if p in ('theta', 'phi', 'theta_a', 'phi_a', 'theta_b', 'phi_b',
                      'theta2', 'phi2'):
                if p.startswith('theta'):
                    samples = 90 - r2d(samples)
                else:
                    samples = r2d(samples)

            med, err = quantile_1sigma(samples)
            display_name = self._angular_display_name(p)
            entries[display_name] = (med, err)

        # Tension statistics
        entries['logR'] = (self._logR, self._logR_err)
        entries['logI'] = (self._logI, self._logI_err)
        entries['logS'] = (self._logS, self._logS_err)
        entries['d'] = (self._d, self._d_err)
        entries['sigma'] = (self._sigma, self._sigma_err)
        entries['p'] = (self._p, self._p_err)

        return entries

    def _angular_display_name(self, param_name):
        """Map angular param names to coordinate-appropriate display names."""
        coord_map = {
            'C': {'theta': 'dec', 'phi': 'ra'},
            'G': {'theta': 'b', 'phi': 'l'},
            'E': {'theta': 'lat_ecl', 'phi': 'lon_ecl'},
        }
        mapping = coord_map.get(self._coord_system, {})
        for base in ('theta', 'phi'):
            if param_name.startswith(base):
                suffix = param_name[len(base):]
                if base in mapping:
                    return mapping[base] + suffix
        return param_name

    def _default_table_order(self):
        """Default column order for the tension table."""
        coord_map = {
            'C': ('ra', 'dec'),
            'G': ('l', 'b'),
            'E': ('lon_ecl', 'lat_ecl'),
        }
        lon_name, lat_name = coord_map.get(self._coord_system, ('phi', 'theta'))

        order = ['v']
        order.append(lon_name)
        order.append(lat_name)

        # Other model parameters
        skip = {'v', 'theta', 'phi'}
        for p in self._post_AB._param_names:
            if p not in skip:
                display = self._angular_display_name(p)
                if display not in order:
                    order.append(display)

        order.extend(['logR', 'logI', 'logS', 'd', 'sigma'])
        return order
