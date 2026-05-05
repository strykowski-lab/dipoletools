"""Example 5: Model types — Comparing poisson, general_poisson, and gaussian.

Demonstrates:
- Setting up each of the three likelihood types
- The bias=True option for ecliptic latitude correction
- How general_poisson adds a gp_dispersion parameter
- Custom prior overrides
"""

import numpy as np
import healpy as hp
from dipoletools import Analyser
from dipoletools._utils import ang2vec, d2r

# -------------------------------------------------------------------------
# Generate a simple test map
# -------------------------------------------------------------------------
np.random.seed(42)
nside = 32
npix = hp.nside2npix(nside)
pos = np.array(hp.pix2vec(nside, np.arange(npix))).T

D_survey = 0.005
dipole_vec = ang2vec(d2r(90 - 40), d2r(260))
cos_angle = np.sum(dipole_vec * pos, axis=1)
N_true = 30.0
expected = N_true * (1 + 3.0 * D_survey * cos_angle)
counts = np.random.poisson(expected).astype(float)

# -------------------------------------------------------------------------
# 1. Standard Poisson model
# -------------------------------------------------------------------------
print("=" * 50)
print("1. POISSON MODEL")
print("=" * 50)
a = Analyser(Map=counts, D=D_survey, map_coords='G')
print(a.model(type='poisson'))
print(a.priors())

# -------------------------------------------------------------------------
# 2. General Poisson model (adds gp_dispersion parameter)
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("2. GENERAL POISSON MODEL")
print("=" * 50)
a2 = Analyser(Map=counts, D=D_survey, map_coords='G')
print(a2.model(type='general_poisson'))
print(a2.priors())

# -------------------------------------------------------------------------
# 3. Gaussian model
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("3. GAUSSIAN MODEL")
print("=" * 50)
# Gaussian is used for continuous data like CMB temperature
temp_map = 2.7255 + 0.003 * cos_angle + np.random.normal(0, 0.001, npix)
a3 = Analyser(Map=temp_map, D=1.0, map_coords='G')
print(a3.model(type='gaussian'))
print(a3.priors())

# -------------------------------------------------------------------------
# 4. Poisson with ecliptic bias
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("4. POISSON WITH ECLIPTIC BIAS")
print("=" * 50)
a4 = Analyser(Map=counts, D=D_survey, map_coords='G')
print(a4.model(type='poisson', bias=True, bias_cecl=9.15e-4))
print(a4.priors())

# -------------------------------------------------------------------------
# 5. Custom prior overrides
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("5. CUSTOM PRIORS")
print("=" * 50)
a5 = Analyser(Map=counts, D=D_survey, map_coords='G')
a5.model(type='poisson')
# Override v prior to a narrower range
a5.priors(v=[0, 10])
# Override N prior with explicit bounds
a5.priors(N=[25.0, 35.0])
print(a5.priors())

# -------------------------------------------------------------------------
# 6. Higher multipoles (ell=[0,1,2] for quadrupole)
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
print("6. QUADRUPOLE MODEL (ell=[0,1,2])")
print("=" * 50)
a6 = Analyser(Map=counts, D=D_survey, map_coords='G')
print(a6.model(type='poisson', ell=[0, 1, 2]))
print(a6.priors())

print("\nDone!")
