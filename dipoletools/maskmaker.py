"""MaskMaker class: create and combine masks for HEALPix maps."""

import os
import re
import numpy as np
import healpy as hp

from ._utils import convert_lonlat, d2r, r2d, _CallableArray
from ._defaults import SHORTHAND_MASKS, DATASTORE_PATH


class MaskMaker:
    """Create and manage HEALPix masks.

    Masks use the convention: True/1 = keep (unmasked), False/0 = masked out.

    Parameters
    ----------
    mask : str, optional
        Path to a mask file (.npy, .pkl, .fits) or a shorthand name
        ('racs-low1', 'nvss', 'catwise'). If provided, the mask is loaded
        and nside/coords are inferred. If omitted, starts with all pixels
        unmasked (ready for manual masking with slices/discs).
    """

    def __init__(self, mask=None):
        self._nside = 64
        self._coord_system = 'G'
        self._mask = np.ones(hp.nside2npix(self._nside), dtype=bool)

        if mask is not None:
            self._load(mask)

    # ------------------------------------------------------------------
    # nside property
    # ------------------------------------------------------------------
    @property
    def nside(self):
        return self._nside

    @nside.setter
    def nside(self, value):
        old_nside = self._nside
        self._nside = int(value)
        if self._nside != old_nside:
            # Resize mask
            self._mask = np.ones(hp.nside2npix(self._nside), dtype=bool)

    # ------------------------------------------------------------------
    # mask property (also callable)
    # ------------------------------------------------------------------
    @property
    def mask(self):
        """Return the current mask array (True = keep, False = masked).

        Can be used as ``mm.mask`` or ``mm.mask()`` — both return the same
        boolean array.
        """
        result = self._mask.copy()
        return _CallableArray(result, lambda: self._mask.copy())

    # ------------------------------------------------------------------
    # show
    # ------------------------------------------------------------------
    def show(self, **kwargs):
        """Display the mask with healpy.projview.

        Unmasked pixels (True) appear as white (background).
        Masked pixels (False) appear as gray.
        Default title: none. Any keyword arguments are forwarded to hp.projview.
        """
        import matplotlib.pyplot as plt
        display = np.full(len(self._mask), np.nan)
        display[~self._mask] = 0.5  # masked pixels → mid-gray
        kwargs.setdefault('title', '')
        hp.projview(display, cmap='Greys', min=0, max=1, **kwargs)
        plt.show()

    # ------------------------------------------------------------------
    # _load (internal)
    # ------------------------------------------------------------------
    def _load(self, filepath):
        """Load a mask from a file or shorthand name."""
        filepath = str(filepath)

        # Resolve shorthand mask names
        if filepath in SHORTHAND_MASKS:
            filepath = os.path.join(DATASTORE_PATH, SHORTHAND_MASKS[filepath])

        if filepath.endswith('.npy'):
            loaded = np.load(filepath)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            import pickle
            with open(filepath, 'rb') as f:
                loaded = pickle.load(f)
        elif filepath.endswith('.fits') or filepath.endswith('.fit'):
            loaded = hp.read_map(filepath)
        else:
            # Try numpy
            loaded = np.load(filepath)

        loaded = np.asarray(loaded)

        # Infer nside from the loaded mask
        if loaded.size == hp.nside2npix(hp.npix2nside(loaded.size)):
            inferred_nside = hp.npix2nside(loaded.size)
            if inferred_nside != self._nside:
                self._nside = inferred_nside
                self._mask = np.ones(loaded.size, dtype=bool)

        # Convert to boolean (treat >0 and True as keep)
        mask_bool = np.asarray(loaded, dtype=bool)

        # Combine with existing mask (AND)
        self._mask = self._mask & mask_bool

    # ------------------------------------------------------------------
    # discs
    # ------------------------------------------------------------------
    def discs(self, locations, radii):
        """Mask circular discs around specified locations.

        Parameters
        ----------
        locations : list of tuples
            Each tuple is (longitude, latitude) in degrees in the current
            coordinate system. E.g. [(l1, b1), (l2, b2)] for galactic.
        radii : float or list of float
            Radius/radii of the disc masks in degrees. If a single value,
            the same radius is used for all locations.
        """
        nside = self._nside

        if isinstance(radii, (int, float)):
            radii = [radii] * len(locations)

        for (lon, lat), radius in zip(locations, radii):
            # Convert to HEALPix theta, phi
            theta = d2r(90 - lat)
            phi = d2r(lon) % (2 * np.pi)
            vec = hp.ang2vec(theta, phi)
            radius_rad = d2r(radius)
            disc_pixels = hp.query_disc(nside, vec, radius_rad)
            self._mask[disc_pixels] = False

    # ------------------------------------------------------------------
    # slices
    # ------------------------------------------------------------------
    def slices(self, *expressions):
        """Mask regions defined by coordinate expressions.

        Areas specified are masked OUT (set to False).

        Supports flexible expressions using l, b, ra, dec:
        - One-sided: 'b > 10', '10 < b', 'dec < -40'
        - Two-sided: '60 < ra < 80', '|b| < 10'
        - Multiple expressions can be passed as separate arguments.

        The masker uses coords to know the current coordinate system
        and converts as necessary.

        Parameters
        ----------
        *expressions : str
            One or more inequality expressions.
        """
        nside = self._nside
        npix = hp.nside2npix(nside)

        # Get pixel coordinates in all needed systems
        theta_pix, phi_pix = hp.pix2ang(nside, np.arange(npix))

        # Compute pixel lat/lon in the current coordinate system
        lat_pix = 90 - r2d(theta_pix)  # latitude in degrees
        lon_pix = r2d(phi_pix)  # longitude in degrees

        for expr in expressions:
            mask_out = self._evaluate_expression(expr, lon_pix, lat_pix, nside)
            self._mask[mask_out] = False

    def _evaluate_expression(self, expr, lon_pix, lat_pix, nside):
        """Parse and evaluate a masking expression. Returns indices to mask out."""
        expr = expr.strip()

        coord_maps = {
            'C': {'lon': 'ra', 'lat': 'dec'},
            'G': {'lon': 'l', 'lat': 'b'},
            'E': {'lon': 'lon_ecl', 'lat': 'lat_ecl'},
        }

        # Determine which coordinate variable is used in the expression
        used_var = None
        var_type = None
        var_system = None

        for sys_name, sys_coords in coord_maps.items():
            for vtype, vname in sys_coords.items():
                if vname in expr:
                    used_var = vname
                    var_type = vtype
                    var_system = sys_name
                    break
            if used_var:
                break

        if used_var is None:
            raise ValueError(f"Could not identify coordinate variable in expression: {expr}")

        # Get pixel values in the appropriate coordinate system
        if var_system == self._coord_system:
            if var_type == 'lat':
                values = lat_pix
            else:
                values = lon_pix
        else:
            new_lon, new_lat = convert_lonlat(lon_pix, lat_pix,
                                              self._coord_system, var_system)
            if var_type == 'lat':
                values = new_lat
            else:
                values = new_lon

        # Handle absolute value notation: |b| < 10 → abs(b) < 10
        abs_expr = re.sub(r'\|(\w+)\|', r'abs(\1)', expr)

        return self._parse_inequality(abs_expr, used_var, values)

    def _parse_inequality(self, expr, var_name, values):
        """Parse an inequality expression and return pixel indices that satisfy it."""
        expr = expr.strip()

        two_sided = re.match(
            r'(-?[\d.]+)\s*([<>]=?)\s*(?:abs\()?(\w+)(?:\))?\s*([<>]=?)\s*(-?[\d.]+)',
            expr
        )
        if two_sided:
            left_val = float(two_sided.group(1))
            left_op = two_sided.group(2)
            right_op = two_sided.group(4)
            right_val = float(two_sided.group(5))

            use_abs = 'abs(' in expr
            v = np.abs(values) if use_abs else values

            left_cond = self._compare(left_val, left_op, v)
            right_cond = self._compare(v, right_op, right_val)
            return np.where(left_cond & right_cond)[0]

        one_sided_var_first = re.match(
            r'(?:abs\()?(\w+)(?:\))?\s*([<>]=?)\s*(-?[\d.]+)', expr
        )
        one_sided_num_first = re.match(
            r'(-?[\d.]+)\s*([<>]=?)\s*(?:abs\()?(\w+)(?:\))?', expr
        )

        use_abs = 'abs(' in expr
        v = np.abs(values) if use_abs else values

        if one_sided_var_first:
            op = one_sided_var_first.group(2)
            num = float(one_sided_var_first.group(3))
            return np.where(self._compare(v, op, num))[0]
        elif one_sided_num_first:
            num = float(one_sided_num_first.group(1))
            op = one_sided_num_first.group(2)
            return np.where(self._compare(num, op, v))[0]
        else:
            raise ValueError(f"Could not parse expression: {expr}")

    @staticmethod
    def _compare(left, op, right):
        """Apply a comparison operator."""
        if op == '<':
            return left < right
        elif op == '<=':
            return left <= right
        elif op == '>':
            return left > right
        elif op == '>=':
            return left >= right
        else:
            raise ValueError(f"Unknown operator: {op}")

    # ------------------------------------------------------------------
    # coords
    # ------------------------------------------------------------------
    def coords(self, *args):
        """Query or set/convert the coordinate system.

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
                elif len(arg) == 2:
                    self._coord_system = arg[1]
                return
            self._coord_system = arg
            return

        if len(args) == 2:
            self._coord_system = args[1]
            return

        raise ValueError("Pass one or two coordinate system names.")
