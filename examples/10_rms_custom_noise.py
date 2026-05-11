"""Example 10: Analyser — RMS likelihood with a custom noise map.

Demonstrates passing a user-supplied HEALPix noise/RMS map directly to
``Analyser.model(rms=...)``. The argument accepts:

- ``rms=True``  — derive the noise map from the MapMaker's 'noise' column
- ``rms=<np.ndarray>`` — use this RING-ordered HEALPix array directly
- ``rms='/path/to/file.npy'`` (or .fits / .hpx) — load from disk

In all custom-map cases the noise map must be in the *same coordinate
system and nside* as the Analyser's count map. NaN pixels in the noise
map are added to the count-map mask automatically (with a warning).
"""

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dipoletools import Analyser, Posterior
from dipoletools._utils import ang2vec, d2r

# -------------------------------------------------------------------------
# 1. Synthetic dipole map with spatially varying noise
# -------------------------------------------------------------------------
np.random.seed(7)
nside = 64
npix = hp.nside2npix(nside)

v_true = 3.0
D_survey = 0.005
theta_true = d2r(90 - 40)
phi_true = d2r(260)
N_true = 20.0
rms_slope_true = 0.5

pos = np.array(hp.pix2vec(nside, np.arange(npix))).T
dipole_vec = ang2vec(theta_true, phi_true)
cos_angle = np.sum(dipole_vec * pos, axis=1)
D = v_true * D_survey

# Construct a spatially varying noise (RMS) map: e.g. higher noise near
# the equator. The Analyser uses this map to scale per-pixel expected
# counts as (rms / median(rms))**(-rms_slope).
theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
lat_pix = 90 - np.degrees(theta_pix)
rms_map = 1.0 + 0.5 * np.cos(np.radians(lat_pix))     # ~[0.5, 1.5]
rms_ref = np.median(rms_map)
rms_factor = (rms_map / rms_ref) ** (-rms_slope_true)

expected = N_true * (1 + D * cos_angle) * rms_factor
counts = np.random.poisson(expected).astype(float)

# Mask the galactic plane in the count map.
mask = np.abs(lat_pix) > 10
counts[~mask] = np.nan

# Introduce a few NaN pixels in the noise map to demonstrate the auto-mask
# extension behaviour.
bad_pix = np.random.choice(np.where(mask)[0], size=20, replace=False)
rms_map[bad_pix] = np.nan

print(f"Generated map: {int(np.sum(mask))} unmasked pixels "
      f"(20 of which have NaN noise and will be auto-masked).")

# -------------------------------------------------------------------------
# 2. Option A: pass the noise map directly as an array
# -------------------------------------------------------------------------
a = Analyser(map=counts, D=D_survey, map_coords='G')
print("\n" + a.model(type='poisson', rms=rms_map))
print("\n" + a.priors())

# -------------------------------------------------------------------------
# 3. Option B (commented): load from a file. The path can be .npy/.fits/.hpx.
# -------------------------------------------------------------------------
# np.save('/tmp/my_rms_map.npy', rms_map)
# a = Analyser(map=counts, D=D_survey, map_coords='G')
# a.model(type='poisson', rms='/tmp/my_rms_map.npy')

# -------------------------------------------------------------------------
# 4. Run nested sampling and check recovery
# -------------------------------------------------------------------------
savedir = '/tmp/example_rms_custom'
results = a.ultranest(savedir=savedir, name='rms_custom',
                      min_num_live_points=200, dlogz=1.0)

p = Posterior(a.savedir, coords='G')
print(f"\nlogZ = {p.logZ:.2f}")
print(f"Truth: v={v_true}, rms_slope={rms_slope_true}")
print(f"Recovered: {p.table()}")

try:
    p.corner()
    plt.savefig('example_outputs/10_corner.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 10_corner.png")
except Exception as e:
    print(f"Corner plot skipped: {e}")

print("\nDone!")
