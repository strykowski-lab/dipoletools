"""Default configurations for known catalogues, priors, and plot labels."""

import numpy as np
import healpy as hp

# ---------------------------------------------------------------------------
# Known catalogue configurations (auto-detection from filepath)
# ---------------------------------------------------------------------------
# Each entry maps filename patterns to default column label mappings.
# 'coord_system' indicates what coordinate system the catalogue is natively in.

KNOWN_CATALOGUES = {
    'racs': {
        'patterns': ['racs', 'RACS', 'AS110'],
        'labels': {'ra': 'ra', 'dec': 'dec', 'flux': 'total_flux_source',
                   'id': 'source_id'},
        'coord_system': 'C',
    },
    'nvss': {
        'patterns': ['nvss', 'NVSS'],
        'labels': {'ra': 'ra', 'dec': 'dec', 'flux': 'flux', 'id': 'id'},
        'coord_system': 'C',
    },
    'catwise': {
        'patterns': ['catwise', 'CatWISE'],
        'labels': {'ra': 'ra', 'dec': 'dec', 'flux': 'w1', 'id': 'source_id'},
        'coord_system': 'C',
    },
    'local_sources': {
        'patterns': ['local_sources', 'ned_2mrs'],
        'labels': {'ra': 'ra', 'dec': 'dec', 'id': 'name'},
        'coord_system': 'C',
    },
}


def detect_catalogue(filepath):
    """Try to identify a known catalogue from the filepath.

    Returns the catalogue config dict if recognized, else None.
    """
    for name, config in KNOWN_CATALOGUES.items():
        for pattern in config['patterns']:
            if pattern in str(filepath):
                return config
    return None


# ---------------------------------------------------------------------------
# Shorthand catalogue names
# ---------------------------------------------------------------------------
# Base path for all shorthand catalogues
DATASTORE_PATH = '/Users/mali/repos/datastore'

def _load_planck_map(nside=64, freq=30):
    """Load Planck foreground-subtracted temperature map."""
    from astropy.io import fits
    lead = 'LFI' if freq < 100 else 'HFI'
    end = '0512' if freq < 70 else '1024'
    freq_str = str(freq).zfill(3)
    T_cmb = 2.7255

    path = f'{DATASTORE_PATH}/BP_{freq_str}_IQU_n{end}_v2.fits'
    hdul = fits.open(path)
    temp = hdul[1].data['I_MEAN'].flatten() * 1e-6  # μK → K

    fg_path = f'{DATASTORE_PATH}/{lead}_CompMap_Foregrounds-smica-{freq_str}_R3.00.fits'
    hdul = fits.open(fg_path)
    fg = hdul[1].data['INTENSITY'].flatten()
    fg = hp.ud_grade(fg, nside_out=hp.get_nside(temp))

    result = temp - fg + T_cmb
    return hp.ud_grade(result, nside_out=nside)


SHORTHAND_CATALOGUES = {
    'planck': {
        'type': 'map',
        'loader': _load_planck_map,
    },
    'catwise': {
        'file': 'catwise_agns.fits',
        'labels': {'ra': 'ra', 'dec': 'dec', 'flux': 'w1', 'id': 'source_id'},
        'cuts': [{'col': 'w1cov', 'min': 80}, {'col': 'w1', 'max': 16.5, 'strict': True}],
        'map_coords': 'G',
    },
    'nvss': {
        'file': 'full_NVSS_combined_named.dat',
        'labels': {'ra': 'ra', 'dec': 'dec', 'flux': 'integrated_flux', 'id': 'source_name', 'flux_err': 'integrated_flux_unc'},
        'cuts': [{'col': 'integrated_flux', 'min': 15, 'max': 1000}],
    },
    'racs-low1': {
        'file': 'RACS-low1_sources_25arcsec_allsources.fits',
        'labels': {'ra': 'ra', 'dec': 'dec', 'flux': 'total_flux_source',
                   'flux_err': 'e_total_flux_source', 'id': 'source_id'},
        'cuts': [{'col': 'total_flux_source', 'min': 15, 'max': 1000}],
    },
    'racs-low2': {
        'file': 'RACS-low2_sources_patched.fits',
        'labels': {'ra': 'RA', 'dec': 'Dec', 'flux': 'Total_flux',
                   'flux_err': 'E_Total_flux', 'id': 'Source_ID'},
        'cuts': [{'col': 'Total_flux', 'min': 15, 'max': 1000}],
    },
    'racs-low2-25': {
        'file': 'RACS-low2_sources_25arcsec_patched.fits',
        'labels': {'ra': 'RA', 'dec': 'Dec', 'flux': 'Total_flux',
                   'flux_err': 'E_Total_flux', 'id': 'Source_ID'},
        'cuts': [{'col': 'Total_flux', 'min': 15, 'max': 1000}],
    },
    'racs-low2-45': {
        'file': 'RACS-low2_sources_45arcsec_patched.fits',
        'labels': {'ra': 'RA', 'dec': 'Dec', 'flux': 'Total_flux',
                   'flux_err': 'E_Total_flux', 'id': 'Source_ID'},
        'cuts': [{'col': 'Total_flux', 'min': 15, 'max': 1000}],
    },
    'racs-low3': {
        'file': 'RACS-low3_sources.fits',
        'labels': {'ra': 'RA', 'dec': 'Dec', 'flux': 'Total_flux',
                   'flux_err': 'E_Total_flux', 'id': 'Source_ID'},
        'cuts': [{'col': 'Total_flux', 'min': 15, 'max': 1000}],
    },
    'racs-low3-scaled': {
        'file': 'RACS-low3_sources_scaled.fits',
        'labels': {'ra': 'RA', 'dec': 'Dec', 'flux': 'Total_flux',
                   'flux_err': 'E_Total_flux', 'id': 'Source_ID'},
        'cuts': [{'col': 'Total_flux', 'min': 15, 'max': 1000}],
    },
    'racs-mid1': {
        'file': 'RACS-mid_sources.fits',
        'labels': {'ra': 'ra', 'dec': 'dec', 'flux': 'total_flux',
                   'flux_err': 'e_total_flux', 'id': 'id'},
        'cuts': [{'col': 'total_flux', 'min': 15, 'max': 1000}],
    },
    'racs-mid1-25': {
        'file': 'RACS-mid_sources_25arcsec.fits',
        'labels': {'ra': 'ra', 'dec': 'dec', 'flux': 'total_flux',
                   'flux_err': 'e_total_flux', 'id': 'id'},
        'cuts': [{'col': 'total_flux', 'min': 15, 'max': 1000}],
    },
    'racs-mid1-45': {
        'file': 'RACS-mid_sources_45arcsec.fits',
        'labels': {'ra': 'RA', 'dec': 'Dec', 'flux': 'Total_flux',
                   'flux_err': 'E_Total_flux', 'id': 'Source_ID'},
        'cuts': [{'col': 'Total_flux', 'min': 15, 'max': 1000}],
    },
    'racs-high': {
        'file': 'RACS-high_sources.fits',
        'labels': {'ra': 'ra', 'dec': 'dec', 'flux': 'total_flux',
                   'flux_err': 'e_total_flux', 'id': 'id', 'noise': 'noise'},
        'cuts': [{'col': 'total_flux', 'min': 15, 'max': 1000}],
    },
    'local': {
        'file': 'local_sources_ned_2mrs.csv',
        'labels': {'ra': 'ra', 'dec': 'dec', 'id': 'LS_id', 'z': 'z'},
    },
}


# ---------------------------------------------------------------------------
# Shorthand mask names
# ---------------------------------------------------------------------------
SHORTHAND_MASKS = {
    'racs-low1': 'racs_galmask.npy',
    'nvss': 'nvss_galmask.npy',
    'catwise': 'catwise_mask.pkl',
}


# ---------------------------------------------------------------------------
# Default prior ranges
# ---------------------------------------------------------------------------
# 'type' can be 'uniform' or 'polar'.
# For 'uniform': value is (low, high).
# For 'polar': the prior transform is theta = arccos(2u - 1).
# For 'auto': computed at runtime from the data.

DEFAULT_PRIORS = {
    'v': {'type': 'uniform', 'low': 0.0, 'high': 20.0},
    'theta': {'type': 'polar'},
    'phi': {'type': 'uniform', 'low': 0.0, 'high': 2 * np.pi},
    'N': {'type': 'auto'},  # ±10% of mean of unmasked pixels
    'Q': {'type': 'uniform', 'low': 0.0, 'high': 1.0},
    'theta_a': {'type': 'polar'},
    'phi_a': {'type': 'uniform', 'low': 0.0, 'high': 2 * np.pi},
    'theta_b': {'type': 'polar'},
    'phi_b': {'type': 'uniform', 'low': 0.0, 'high': 2 * np.pi},
    'bias': {'type': 'uniform', 'low': -2.0, 'high': 2.0},
    'rms_slope': {'type': 'auto'},  # ±25% of power-law fit slope
    'gp_dispersion': {'type': 'uniform', 'low': 0.0, 'high': 1.0},
    # Secondary ("second") dipole parameters for two-dipole models.
    'v_sd': {'type': 'uniform', 'low': 0.0, 'high': 20.0},
    'theta_sd': {'type': 'polar'},
    'phi_sd': {'type': 'uniform', 'low': 0.0, 'high': 2 * np.pi},
}

# ---------------------------------------------------------------------------
# Default plot labels
# ---------------------------------------------------------------------------
# Labels are coordinate-system dependent for angular parameters.

PLOT_LABELS_BASE = {
    'v': r'\tilde{v}',
    'v_sd': r'\tilde{v}_{\rm sd}',
    'N': r'\bar{N}',
    'Q': r'Q',
    'bias': r'b_{\rm ecl}',
    'rms_slope': r'x_{\rm rms}',
    'gp_dispersion': r'b_{\rm GP}',
    'logZ': r'\ln\mathcal{Z}',
    'kl': r'\mathcal{D}_{\rm KL}',
    'd': r'd_{\rm G}',
    'logR': r'\ln\mathcal{R}',
    'logI': r'\ln\mathcal{I}',
    'logS': r'\ln\mathcal{S}',
    'sigma': r'\sigma',
    'p': r'p',
}

PLOT_LABELS_ANGULAR = {
    'C': {
        'theta': r'\delta', 'phi': r'\alpha',
        'theta_a': r'\delta_a', 'phi_a': r'\alpha_a',
        'theta_b': r'\delta_b', 'phi_b': r'\alpha_b',
        'theta_sd': r'\delta_{\rm sd}', 'phi_sd': r'\alpha_{\rm sd}',
    },
    'G': {
        'theta': r'b', 'phi': r'l',
        'theta_a': r'b_a', 'phi_a': r'l_a',
        'theta_b': r'b_b', 'phi_b': r'l_b',
        'theta_sd': r'b_{\rm sd}', 'phi_sd': r'l_{\rm sd}',
    },
    'E': {
        'theta': r'\beta', 'phi': r'\lambda',
        'theta_a': r'\beta_a', 'phi_a': r'\lambda_a',
        'theta_b': r'\beta_b', 'phi_b': r'\lambda_b',
        'theta_sd': r'\beta_{\rm sd}', 'phi_sd': r'\lambda_{\rm sd}',
    },
}


def get_label(param_name, coord_system='C', user_labels=None):
    """Get the plot label for a parameter name."""
    if user_labels and param_name in user_labels:
        return user_labels[param_name]
    if coord_system in PLOT_LABELS_ANGULAR:
        if param_name in PLOT_LABELS_ANGULAR[coord_system]:
            return PLOT_LABELS_ANGULAR[coord_system][param_name]
    if param_name in PLOT_LABELS_BASE:
        return PLOT_LABELS_BASE[param_name]
    # Strip trailing '2' for joint analysis parameters
    base_name = param_name.rstrip('2')
    if base_name != param_name:
        base_label = get_label(base_name, coord_system, user_labels)
        if base_label != base_name:
            return base_label + r'_2'
    return param_name
