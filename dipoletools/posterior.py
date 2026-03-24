"""Posterior class: chain analysis, plotting, and table generation."""

import os
import numpy as np

from ._utils import (
    healpy_quantile, quantile_1sigma, quantile_2sigma,
    convert_thetaphi, r2d, d2r, format_value_error
)
from ._defaults import get_label, PLOT_LABELS_ANGULAR


class Posterior:
    """Analyse nested sampling chains and produce plots and tables.

    Parameters
    ----------
    source : Analyser, str, or Posterior
        An Analyser object (uses its savedir), a savedir path string,
        or another Posterior object.
    coords : str
        Coordinate system for angular parameters. Default 'C'.
    """

    def __init__(self, source, coords='C'):
        self._coord_system = coords
        self._savedir = None
        self._param_names = None
        self._samples = None

        # Anesthetic statistics
        self._kl = None
        self._kl_err = None
        self._logZ = None
        self._logZ_err = None
        self._d = None
        self._d_err = None

        # Resolve the source
        if isinstance(source, str):
            self._savedir = source
        elif hasattr(source, 'savedir'):
            sd = source.savedir
            if sd is None:
                raise ValueError("Source object has no savedir. Run ultranest first.")
            self._savedir = sd
        elif hasattr(source, 'Savedir'):
            # Legacy support
            sd = source.Savedir
            if sd is None:
                raise ValueError("Source object has no savedir. Run ultranest first.")
            self._savedir = sd
        else:
            raise ValueError("source must be an Analyser, Posterior, or savedir string.")

        # Load chains
        self._load_chains()

        # Compute or load anesthetic statistics
        self._init_stats()

    # ------------------------------------------------------------------
    # Load chains
    # ------------------------------------------------------------------
    def _load_chains(self):
        """Load equal-weighted posterior samples from the savedir."""
        chains_file = os.path.join(self._savedir, 'chains', 'equal_weighted_post.txt')
        if not os.path.exists(chains_file):
            # Maybe savedir IS the parent, try run1 or look for chains dir
            for sub in ['run1', '']:
                test = os.path.join(self._savedir, sub, 'chains', 'equal_weighted_post.txt')
                if os.path.exists(test):
                    chains_file = test
                    if sub:
                        self._savedir = os.path.join(self._savedir, sub)
                    break

        if not os.path.exists(chains_file):
            raise FileNotFoundError(
                f"Cannot find chains at {chains_file}. "
                "Ensure nested sampling has been run."
            )

        with open(chains_file, 'r') as f:
            lines = f.readlines()

        self._param_names = lines[0].strip().split()
        data = []
        for line in lines[1:]:
            vals = line.strip().split()
            data.append([float(v) for v in vals])

        self._samples = np.array(data)

    # ------------------------------------------------------------------
    # Anesthetic statistics
    # ------------------------------------------------------------------
    def _init_stats(self):
        """Compute or load kl, logZ, d from anesthetic."""
        stats_file = os.path.join(self._savedir, 'anesthetic_stats.npz')

        if os.path.exists(stats_file):
            loaded = np.load(stats_file)
            self._logZ = float(loaded['logZ'])
            self._logZ_err = loaded['logZ_err']
            self._kl = float(loaded['kl'])
            self._kl_err = loaded['kl_err']
            self._d = float(loaded['d'])
            self._d_err = loaded['d_err']
            return

        try:
            from anesthetic import read_chains
            nested_samples = read_chains(self._savedir)
            nsamples = nested_samples.shape[0]
            bayesian_stats = nested_samples.stats(nsamples)

            self._logZ, self._logZ_err = quantile_2sigma(bayesian_stats['logZ'])
            self._kl, self._kl_err = quantile_2sigma(bayesian_stats['D_KL'])
            self._d, self._d_err = quantile_2sigma(bayesian_stats['d_G'])

            # Save to disk
            np.savez(stats_file,
                     logZ=self._logZ, logZ_err=self._logZ_err,
                     kl=self._kl, kl_err=self._kl_err,
                     d=self._d, d_err=self._d_err)
        except Exception as e:
            print(f"Warning: Could not compute anesthetic stats: {e}")
            self._logZ = self._kl = self._d = 0.0
            self._logZ_err = self._kl_err = self._d_err = np.array([0.0, 0.0])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def savedir(self):
        return self._savedir

    @property
    def kl(self):
        return self._kl

    @property
    def kl_err(self):
        return self._kl_err

    @property
    def logZ(self):
        return self._logZ

    @property
    def logZ_err(self):
        return self._logZ_err

    @property
    def d(self):
        return self._d

    @property
    def d_err(self):
        return self._d_err

    # ------------------------------------------------------------------
    # coords
    # ------------------------------------------------------------------
    def coords(self, *args):
        """Query or set/convert the coordinate system for angular parameters.

        No args: returns current system.
        One arg: labels current system.
        Two args or list of two: converts from first to second.
        """
        if len(args) == 0:
            return self._coord_system

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (list, tuple)):
                if len(arg) == 1:
                    self._coord_system = arg[0]
                    return
                elif len(arg) == 2:
                    from_sys, to_sys = arg
                    self._convert_chains(from_sys, to_sys)
                    self._coord_system = to_sys
                    return
            else:
                self._coord_system = arg
                return
        elif len(args) == 2:
            from_sys, to_sys = args
            self._convert_chains(from_sys, to_sys)
            self._coord_system = to_sys
            return

    def _convert_chains(self, from_sys, to_sys):
        """Convert angular columns in the chains between coordinate systems."""
        if from_sys == to_sys:
            return

        # Find angular parameter indices
        angular_pairs = self._find_angular_pairs()

        for theta_idx, phi_idx in angular_pairs:
            theta_rad = self._samples[:, theta_idx]
            phi_rad = self._samples[:, phi_idx]
            new_theta, new_phi = convert_thetaphi(theta_rad, phi_rad, from_sys, to_sys)
            self._samples[:, theta_idx] = new_theta
            self._samples[:, phi_idx] = new_phi

    def _find_angular_pairs(self):
        """Find (theta_index, phi_index) pairs in the parameter names."""
        pairs = []
        names = self._param_names

        # Standard pair
        if 'theta' in names and 'phi' in names:
            pairs.append((names.index('theta'), names.index('phi')))

        # Quadrupole pairs
        if 'theta_a' in names and 'phi_a' in names:
            pairs.append((names.index('theta_a'), names.index('phi_a')))
        if 'theta_b' in names and 'phi_b' in names:
            pairs.append((names.index('theta_b'), names.index('phi_b')))

        # Joint analysis pairs (with '2' suffix)
        if 'theta2' in names and 'phi2' in names:
            pairs.append((names.index('theta2'), names.index('phi2')))

        return pairs

    # ------------------------------------------------------------------
    # chains
    # ------------------------------------------------------------------
    def chains(self):
        """Return the raw chain samples as a numpy array.

        Columns correspond to self._param_names.
        """
        return self._samples.copy()

    # ------------------------------------------------------------------
    # corner plot
    # ------------------------------------------------------------------
    def corner(self, parameters=None, labels=None, **kwargs):
        """Plot a GetDist corner plot of the posterior chains.

        Parameters
        ----------
        parameters : list of str, optional
            Parameter names to plot. Default: all.
        labels : dict, optional
            Override labels for parameters. E.g. {'v': r'$v_{\\rm km/s}$'}.
        **kwargs
            Additional kwargs passed to getdist plotting.
        """
        from getdist import MCSamples, plots as gdplots

        names = self._param_names
        samples = self._samples

        if parameters is not None:
            indices = [names.index(p) for p in parameters if p in names]
            plot_names = [names[i] for i in indices]
            plot_samples = samples[:, indices]
        else:
            plot_names = names
            plot_samples = samples

        plot_labels = [get_label(p, self._coord_system, labels) for p in plot_names]

        mc = MCSamples(samples=plot_samples, names=plot_names, labels=plot_labels)

        g = gdplots.get_subplot_plotter()
        g.triangle_plot(mc, filled=True, **kwargs)

        import matplotlib.pyplot as plt
        plt.show()
        return g

    # ------------------------------------------------------------------
    # sky plot
    # ------------------------------------------------------------------
    def sky(self, levels=None, smooth=0.04, color='cornflowerblue', **kwargs):
        """Plot posterior contours on a healpy Mollweide projection.

        Automatically detects angular parameters and converts to
        the appropriate coordinate system for plotting.

        Parameters
        ----------
        levels : array-like, optional
            Sigma levels for contours. Default [0.5, 1, 1.5, 2].
        smooth : float
            Smoothing factor for contours. Default 0.04.
        color : str
            Contour color. Default 'cornflowerblue'.
        """
        import healpy as hp
        import matplotlib.pyplot as plt

        if levels is None:
            levels = np.array([0.5, 1, 1.5, 2])
        levels = 1 - np.exp(-0.5 * np.asarray(levels)**2)

        # Find angular pairs
        angular_pairs = self._find_angular_pairs()
        if not angular_pairs:
            raise ValueError("No angular parameters found in the chains.")

        for theta_idx, phi_idx in angular_pairs:
            theta_samples = self._samples[:, theta_idx]
            phi_samples = self._samples[:, phi_idx]

            # Convert to (latitude, longitude) for plotting
            if self._coord_system == 'G':
                # Galactic: theta=colatitude, phi=longitude
                b_rad = d2r(90) - theta_samples  # latitude in radians
                l_rad = phi_samples  # longitude in radians
                # Remap l for healpy mollweide: [0,pi]->[-pi,0], [pi,2pi]->[0,pi]
                l_plot = np.where(l_rad <= np.pi, -l_rad, 2 * np.pi - l_rad)
                coord_label = 'G'
            elif self._coord_system == 'C':
                # Equatorial: theta=colatitude, phi=longitude
                dec_rad = np.pi / 2 - theta_samples
                ra_rad = phi_samples
                # Convert to galactic for standard mollview display
                from astropy.coordinates import SkyCoord
                from astropy import units as u
                sc = SkyCoord(ra=r2d(ra_rad) * u.deg, dec=r2d(dec_rad) * u.deg, frame='icrs')
                gc = sc.galactic
                b_rad = gc.b.rad
                l_rad = gc.l.rad
                l_plot = np.where(l_rad <= np.pi, -l_rad, 2 * np.pi - l_rad)
                coord_label = 'G'
            else:
                b_rad = np.pi / 2 - theta_samples
                l_plot = phi_samples
                coord_label = self._coord_system

            # Plot
            hp.projview(np.zeros(hp.nside2npix(1)), cmap='Greys', min=0, max=1,
                        graticule=True, graticule_labels=False, cbar=False)

            try:
                from dynesty import plotting as dyplot
                dyplot._hist2d(l_plot, b_rad, levels=levels, smooth=smooth,
                               color=color, no_fill_contours=True,
                               contour_kwargs={'zorder': 1, 'linewidths': 2},
                               contourf_kwargs={'zorder': 1})
            except (ImportError, TypeError):
                pass

            # Add label for which angular pair this is
            theta_name = self._param_names[theta_idx]
            if theta_name != 'theta':
                plt.title(f"Angular pair: {theta_name}, {self._param_names[phi_idx]}")

        plt.show()

    # ------------------------------------------------------------------
    # table
    # ------------------------------------------------------------------
    def table(self, parameters=None, decimals=None, velocity_scale=369.82):
        """Generate a formatted table row for this analysis.

        Parameters
        ----------
        parameters : list of str, optional
            Parameter names to include, in order. If None, uses default order:
            v, l/phi, b/theta, [other model params], logZ, d, kl.
        decimals : int or list of int, optional
            Number of decimal places. If None, auto-determined from errors.
        velocity_scale : float
            Factor to convert v to km/s. Default 369.82 (v_CMB).

        Returns
        -------
        str
            Formatted LaTeX table row.
        """
        # Get parameter samples and compute statistics
        all_entries = self._prepare_table_entries(velocity_scale)

        # Determine the order of entries
        if parameters is not None:
            ordered_keys = parameters
        else:
            ordered_keys = self._default_table_order()

        # Filter to available entries
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

        # Format each entry
        parts = []
        for i, key in enumerate(ordered_keys):
            med, err = all_entries[key]
            dec = decimals[i] if decimals is not None else None
            parts.append(format_value_error(med, err[0], err[1], dec))

        return ' & '.join(parts)

    def _prepare_table_entries(self, velocity_scale):
        """Compute median and errors for all table entries."""
        entries = {}

        for i, p in enumerate(self._param_names):
            samples = self._samples[:, i].copy()

            # Scale v to km/s
            if p == 'v':
                samples *= velocity_scale

            # Convert angular params to degrees if still in radians
            if p in ('theta', 'phi', 'theta_a', 'phi_a', 'theta_b', 'phi_b',
                      'theta2', 'phi2'):
                # Convert colatitude to latitude for theta
                if p.startswith('theta'):
                    samples = 90 - r2d(samples)
                else:
                    samples = r2d(samples)

            med, err = quantile_1sigma(samples)

            # Map angular param names to coordinate-system names
            display_name = self._angular_display_name(p)
            entries[display_name] = (med, err)

        # Add statistical quantities
        entries['logZ'] = (self._logZ, self._logZ_err)
        entries['d'] = (self._d, self._d_err)
        entries['kl'] = (self._kl, self._kl_err)

        return entries

    def _angular_display_name(self, param_name):
        """Map internal angular param names to coordinate-system-appropriate names."""
        coord_map = {
            'C': {'theta': 'dec', 'phi': 'ra'},
            'G': {'theta': 'b', 'phi': 'l'},
            'E': {'theta': 'lat_ecl', 'phi': 'lon_ecl'},
        }
        mapping = coord_map.get(self._coord_system, {})

        # Handle suffixed versions (theta_a, theta2, etc.)
        for base in ('theta', 'phi'):
            if param_name.startswith(base):
                suffix = param_name[len(base):]
                if base in mapping:
                    return mapping[base] + suffix
        return param_name

    def _default_table_order(self):
        """Return the default order for table columns."""
        coord_map = {
            'C': ('ra', 'dec'),
            'G': ('l', 'b'),
            'E': ('lon_ecl', 'lat_ecl'),
        }
        lon_name, lat_name = coord_map.get(self._coord_system, ('phi', 'theta'))

        order = ['v']
        order.append(lon_name)
        order.append(lat_name)

        # Add other model parameters
        skip = {'v', 'theta', 'phi'}
        for p in self._param_names:
            if p not in skip:
                display = self._angular_display_name(p)
                if display not in order:
                    order.append(display)

        order.extend(['logZ', 'd', 'kl'])
        return order
