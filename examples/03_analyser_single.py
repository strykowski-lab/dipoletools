"""Example 3: Analyser — Single dataset analysis with nested sampling.

Demonstrates:
- Generating a synthetic dipole map
- Configuring model types: 'poisson', 'general_poisson', 'gaussian'
- Setting priors
- Running UltraNest nested sampling
- Loading results into a Posterior for plotting and tables

This example generates a Poisson count map with a known dipole signal,
runs nested sampling to recover the input parameters, then produces
a corner plot, sky plot, and formatted table.
"""

import numpy as np
import healpy as hp
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dipoletools import Analyser, Posterior
from dipoletools._utils import ang2vec, d2r

# -------------------------------------------------------------------------
# 1. Generate a synthetic dipole map
# -------------------------------------------------------------------------
np.random.seed(123)
nside = 64
npix = hp.nside2npix(nside)

# True parameters
v_true = 3.0        # v/v_CMB
D_survey = 0.005    # kinematic dipole amplitude
theta_true = d2r(90 - 40)   # colatitude for b=40 deg
phi_true = d2r(260)          # longitude l=260 deg
N_true = 20.0                # mean counts per pixel

# Build the expected count map
pos = np.array(hp.pix2vec(nside, np.arange(npix))).T
dipole_vec = ang2vec(theta_true, phi_true)
cos_angle = np.sum(dipole_vec * pos, axis=1)
D = v_true * D_survey
expected = N_true * (1 + D * cos_angle)

# Draw Poisson realisation
counts = np.random.poisson(expected).astype(float)

# Apply a simple galactic plane mask (|b| < 10 degrees)
theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
lat_pix = 90 - np.degrees(theta_pix)
mask = np.abs(lat_pix) > 10
counts[~mask] = np.nan

print(f"Generated map: {np.sum(mask)} unmasked pixels, mean = {np.nanmean(counts[mask]):.1f}")
print(f"True parameters: v={v_true}, b={40}°, l={260}°, N={N_true}")

# Visualise the input map
hp.mollview(counts, title='Synthetic dipole map', cmap='viridis')
hp.graticule()
plt.savefig('example_outputs/03_input_map.png', dpi=100, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# 2. Set up and run the Analyser
# -------------------------------------------------------------------------
a = Analyser(Map=counts, D=D_survey, map_coords='G')

# Configure model — default is poisson with ell=[0,1]
print("\n" + a.model(type='poisson'))
print("\n" + a.priors())

# Run nested sampling (quick settings for this example)
savedir = '/tmp/example_single'
results = a.ultranest(savedir=savedir, name='synthetic_dipole',
                      min_num_live_points=200, dlogz=1.0)

print(f"\nSampling complete. logZ = {results['logz']:.1f} +/- {results['logzerr']:.1f}")

# -------------------------------------------------------------------------
# 3. Load the Posterior and inspect results
# -------------------------------------------------------------------------
p = Posterior(a.savedir, coords='G')

print(f"\nPosterior loaded: {p._samples.shape[0]} samples")
print(f"logZ = {p.logZ:.2f} +{p.logZ_err[0]:.2f} -{p.logZ_err[1]:.2f}")
print(f"KL divergence = {p.kl:.2f}")
print(f"Bayesian dimensionality = {p.d:.2f}")

# -------------------------------------------------------------------------
# 4. Table output
# -------------------------------------------------------------------------
print(f"\nTable row:\n  {p.table()}")

# -------------------------------------------------------------------------
# 5. Corner plot
# -------------------------------------------------------------------------
try:
    g = p.corner()
    plt.savefig('example_outputs/03_corner.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\nSaved 03_corner.png")
except Exception as e:
    print(f"\nCorner plot skipped: {e}")

# -------------------------------------------------------------------------
# 6. Sky plot
# -------------------------------------------------------------------------
try:
    p.sky(color='cornflowerblue')
    plt.savefig('example_outputs/03_sky.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 03_sky.png")
except Exception as e:
    print(f"Sky plot skipped: {e}")

# -------------------------------------------------------------------------
# 7. Smoothed map
# -------------------------------------------------------------------------
try:
    a.smooth(steradians=0.5, cbar_units='counts/pixel', plot=True)
    plt.savefig('example_outputs/03_smooth.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 03_smooth.png")
except Exception as e:
    print(f"Smooth map skipped: {e}")

print("\nDone!")
