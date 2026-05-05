import numpy as np
import healpy as hp
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dipoletools import Analyser, Posterior, Tension
from dipoletools._utils import ang2vec, d2r

np.random.seed(2026)
nside = 64
npix = hp.nside2npix(nside)
pos = np.array(hp.pix2vec(nside, np.arange(npix))).T

D_survey = 0.005

# Dataset A: dipole at (l=260, b=40)
theta_A = d2r(90 - 40)
phi_A = d2r(260)
v_A = 4.0
N_A = 25.0
expected_A = N_A * (1 + v_A * D_survey * np.sum(ang2vec(theta_A, phi_A) * pos, axis=1))
counts_A = np.random.poisson(expected_A).astype(float)

# Dataset B: dipole at (l=160, b=-20) — significantly offset
theta_B = d2r(90 - (-20))
phi_B = d2r(160)
v_B = 4.0
N_B = 18.0
expected_B = N_B * (1 + v_B * D_survey * np.sum(ang2vec(theta_B, phi_B) * pos, axis=1))
counts_B = np.random.poisson(expected_B).astype(float)

# Apply galactic plane mask to both
theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
lat_pix = 90 - np.degrees(theta_pix)
gal_mask = np.abs(lat_pix) > 10
counts_A[~gal_mask] = np.nan
counts_B[~gal_mask] = np.nan

print(f"Dataset A: N={N_A}, v={v_A}, direction=(l=260, b=40)")
print(f"Dataset B: N={N_B}, v={v_B}, direction=(l=160, b=-20)")
print(f"Unmasked pixels: {np.sum(gal_mask)}")

savedir = '/tmp/example_tension'

print("\n--- Running Dataset A ---")
a_A = Analyser(Map=counts_A, D=D_survey, map_coords='G')
a_A.model(type='poisson')

print("\n--- Running Dataset B ---")
a_B = Analyser(Map=counts_B, D=D_survey, map_coords='G')
a_B.model(type='poisson')

# Joint model test
print(a_A._shared_parameters) # should print [] (empty before add)
a_A.add(a_B)
print(a_A._shared_parameters) # should print ['v', 'theta', 'phi'] by default
a_A.priors(shared_parameters=['v'])
print(a_A._shared_parameters) # should print ['v']
exit()