"""Example 8: Analyser — Two-dipole model (primary + secondary).

The ``second_dipole`` kwarg on ``Analyser.model`` adds a second dipole
component to the expected counts:

    mu_i = N * (1 + D * v * (n . p_i) + D * v_sd * (n_sd . p_i))

where (v, n) are the primary dipole's amplitude and direction, and
(v_sd, n_sd) are the secondary's. Either v_sd or n_sd (or both) can be
fixed by the user; anything left unfixed is sampled with the same default
priors as the primary dipole.

This example shows the three supported modes:
1. Fixed direction (e.g. pinned at the south equatorial pole), free v_sd.
2. Fixed amplitude, free direction.
3. Both fixed — the secondary contributes a constant offset pattern.

The fixed direction is specified in any of 'C' (equatorial/ICRS),
'G' (galactic), or 'E' (ecliptic); dipoletools converts it internally to
the map's coord system, so you can keep working in whatever frame the
map is in.
"""

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dipoletools import Analyser, Posterior
from dipoletools._utils import ang2vec, d2r, convert_lonlat

np.random.seed(42)
nside = 32
npix = hp.nside2npix(nside)
D_survey = 5e-3

# ---------------------------------------------------------------------
# Generate a synthetic map with TWO dipoles: one toward (l=260, b=40) in
# galactic coords (primary), one toward the south equatorial pole
# (secondary, specified in 'C', converted to 'G' for map generation).
# ---------------------------------------------------------------------
v1_true, theta1_true, phi1_true = 3.0, d2r(90 - 40), d2r(260)
v2_true = 1.5

# Convert south equatorial pole (ra=0, dec=-90) into galactic coords.
lon_g, lat_g = convert_lonlat(np.array([0.0]), np.array([-90.0]), 'C', 'G')
theta2_g = d2r(90 - lat_g[0])
phi2_g = d2r(lon_g[0]) % (2 * np.pi)

pos = np.array(hp.pix2vec(nside, np.arange(npix))).T
n1 = ang2vec(theta1_true, phi1_true)
n2 = ang2vec(theta2_g, phi2_g)
N_true = 40.0
expected = N_true * (
    1 + v1_true * D_survey * (pos @ n1) + v2_true * D_survey * (pos @ n2)
)
counts = np.random.poisson(expected).astype(float)
theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
mask = np.abs(90 - np.degrees(theta_pix)) > 10
counts[~mask] = np.nan

# ---------------------------------------------------------------------
# Mode 1 — fix direction (south equatorial pole), free v_sd
# ---------------------------------------------------------------------
a = Analyser(map=counts, D=D_survey, map_coords='G')
print(a.model(type='poisson', second_dipole={
    'fix_direction': (0.0, -90.0),  # ra, dec of south equatorial pole
    'direction_coords': 'C',        # interpreted as equatorial, converted
    'fix_v': None,                  # free amplitude
}))
print(a.priors())
a.ultranest(savedir='/tmp/example_two_dipoles_mode1',
            min_num_live_points=200, dlogz=1.0)

p = Posterior(a.savedir, coords='G')
print("Mode 1 params:", p._param_names)
print(f"Mode 1: recovered v={v1_true} (true), v_sd={v2_true} (true)")
print(p.table())

# ---------------------------------------------------------------------
# Mode 2 — fix amplitude, free secondary direction
# ---------------------------------------------------------------------
a2 = Analyser(map=counts, D=D_survey, map_coords='G')
print(a2.model(type='poisson', second_dipole={
    'fix_direction': None,
    'fix_v': v2_true,
}))
a2.ultranest(savedir='/tmp/example_two_dipoles_mode2',
             min_num_live_points=200, dlogz=1.0)
p2 = Posterior(a2.savedir, coords='G')
print("Mode 2 params:", p2._param_names)

# ---------------------------------------------------------------------
# Mode 3 — fix both (secondary is a known constant offset pattern)
# ---------------------------------------------------------------------
a3 = Analyser(map=counts, D=D_survey, map_coords='G')
print(a3.model(type='poisson', second_dipole={
    'fix_direction': (0.0, -90.0),
    'direction_coords': 'C',
    'fix_v': v2_true,
}))
a3.ultranest(savedir='/tmp/example_two_dipoles_mode3',
             min_num_live_points=200, dlogz=1.0)
p3 = Posterior(a3.savedir, coords='G')
print("Mode 3 params:", p3._param_names)

print("\nDone — see /tmp/example_two_dipoles_mode{1,2,3}/ for chains.")
