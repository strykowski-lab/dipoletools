import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u


class _CallableArray(np.ndarray):
    """A numpy array subclass that is also callable.

    Enables dual ``.attr`` / ``.attr()`` syntax for properties that compute
    arrays, e.g. ``mm.map`` and ``mm.map()`` both work.
    """
    _fn = None

    def __new__(cls, array, fn=None):
        obj = np.asarray(array).view(cls)
        obj._fn = fn
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._fn = getattr(obj, '_fn', None)

    def __call__(self, *args, **kwargs):
        if self._fn is not None:
            return self._fn(*args, **kwargs)
        raise TypeError("This array is not callable")


def ang2vec(theta, phi):
    """Convert (theta, phi) in radians to Cartesian unit vector(s)."""
    sintheta = np.sin(theta)
    return np.array([np.cos(phi) * sintheta, np.sin(phi) * sintheta, np.cos(theta)])


def r2d(rad):
    return rad * 180 / np.pi


def d2r(deg):
    return deg * np.pi / 180


def healpy_quantile(x, q=None, weights=None):
    if q is None:
        q = [0.16, 0.5, 0.84]
    if weights is None:
        weights = np.ones(x.shape)
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")
    weights = np.atleast_1d(weights)
    idx = np.argsort(x)
    sw = weights[idx]
    cdf = np.cumsum(sw)[:-1]
    cdf /= cdf[-1]
    cdf = np.append(0, cdf)
    quantiles = np.interp(q, cdf, x[idx]).tolist()
    return quantiles


def quantile_1sigma(samples):
    """Return (median, [upper_err, lower_err]) at 1-sigma."""
    ql, qm, qh = healpy_quantile(samples, [0.16, 0.5, 0.84])
    return qm, np.array([qh - qm, qm - ql])


def quantile_2sigma(samples):
    """Return (median, [upper_err, lower_err]) at 2-sigma (for bootstrap stats)."""
    ql, qm, qh = healpy_quantile(samples, [0.025, 0.5, 0.975])
    return qm, np.array([qh - qm, qm - ql])


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------
# Supported systems: 'C' (celestial/equatorial/ICRS), 'G' (galactic), 'E' (ecliptic)

def _to_skycoord(lon_deg, lat_deg, system):
    """Create a SkyCoord from longitude/latitude in degrees in the given system."""
    if system == 'C':
        return SkyCoord(ra=lon_deg * u.deg, dec=lat_deg * u.deg, frame='icrs')
    elif system == 'G':
        return SkyCoord(l=lon_deg * u.deg, b=lat_deg * u.deg, frame='galactic')
    elif system == 'E':
        return SkyCoord(lon=lon_deg * u.deg, lat=lat_deg * u.deg,
                        frame='barycentricmeanecliptic')
    else:
        raise ValueError(f"Unknown coordinate system: {system}. Use 'C', 'G', or 'E'.")


def _from_skycoord(sc, system):
    """Extract (longitude_deg, latitude_deg) arrays from a SkyCoord in the target system."""
    if system == 'C':
        c = sc.icrs
        return c.ra.deg, c.dec.deg
    elif system == 'G':
        c = sc.galactic
        return c.l.deg, c.b.deg
    elif system == 'E':
        c = sc.transform_to('barycentricmeanecliptic')
        return c.lon.deg, c.lat.deg
    else:
        raise ValueError(f"Unknown coordinate system: {system}. Use 'C', 'G', or 'E'.")


def convert_lonlat(lon_deg, lat_deg, from_sys, to_sys):
    """Convert arrays of (longitude, latitude) in degrees between coordinate systems."""
    if from_sys == to_sys:
        return lon_deg.copy(), lat_deg.copy()
    sc = _to_skycoord(lon_deg, lat_deg, from_sys)
    return _from_skycoord(sc, to_sys)


def convert_thetaphi(theta_rad, phi_rad, from_sys, to_sys):
    """Convert colatitude/longitude (radians) between coordinate systems.

    theta = colatitude [0, pi], phi = longitude [0, 2pi].
    """
    if from_sys == to_sys:
        return theta_rad.copy(), phi_rad.copy()
    lat_deg = 90 - r2d(theta_rad)
    lon_deg = r2d(phi_rad)
    new_lon, new_lat = convert_lonlat(lon_deg, lat_deg, from_sys, to_sys)
    new_theta = d2r(90 - new_lat)
    new_phi = d2r(new_lon) % (2 * np.pi)
    return new_theta, new_phi


def lonlat_names(system):
    """Return (lon_name, lat_name) for a coordinate system."""
    if system == 'C':
        return 'ra', 'dec'
    elif system == 'G':
        return 'l', 'b'
    elif system == 'E':
        return 'lon_ecl', 'lat_ecl'
    else:
        raise ValueError(f"Unknown coordinate system: {system}")


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------

def _auto_decimals(error):
    """Determine number of decimal places from the error value.

    Rules:
    - Count decimal places to reach the first non-zero digit.
    - If that digit is 1, add one more decimal place.
    """
    if error == 0 or not np.isfinite(error):
        return 2
    abs_err = abs(error)
    if abs_err >= 1:
        # For errors >= 1, find first significant digit
        magnitude = int(np.floor(np.log10(abs_err)))
        first_digit = int(abs_err / 10**magnitude)
        if first_digit == 1:
            return max(0, -magnitude + 1)
        else:
            return max(0, -magnitude)
    else:
        # For errors < 1, count leading zeros after decimal
        magnitude = int(np.floor(np.log10(abs_err)))  # negative
        decimals = -magnitude  # number of decimal places to reach first non-zero
        first_digit = int(abs_err * 10**decimals)
        if first_digit == 1:
            decimals += 1
        return decimals


def format_value_error(median, upper, lower, decimals=None):
    """Format a value with asymmetric errors for a table.

    If upper and lower errors are equal when rounded, uses ± format.
    Otherwise uses median^{+upper}_{-lower} format.
    """
    if decimals is None:
        # Use the larger error to determine sig figs
        ref_err = max(abs(upper), abs(lower))
        decimals = _auto_decimals(ref_err)

    fmt_str = f'.{decimals}f'
    med_str = f'{median:{fmt_str}}'
    up_str = f'{upper:{fmt_str}}'
    lo_str = f'{lower:{fmt_str}}'

    if up_str == lo_str:
        return f'${med_str} \\pm {up_str}$'
    else:
        return f'${{{med_str}}}_{{-{lo_str}}}^{{+{up_str}}}$'
