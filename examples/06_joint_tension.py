"""Example 6: Joint analysis — Two datasets in TENSION.

Generates two synthetic Poisson dipole maps that point in different
directions on the sky, runs individual and joint nested sampling,
then computes tension statistics and produces plots.

Dataset A: dipole pointing toward (l=260, b=40)   — like radio surveys
Dataset B: dipole pointing toward (l=160, b=-20)  — deliberately offset

The shared parameters (v, theta, phi) are forced to agree in the joint
analysis, which should produce significant tension.
"""

import numpy as np
import healpy as hp
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dipoletools import Analyser, Posterior, Tension
from dipoletools._utils import ang2vec, d2r

# -------------------------------------------------------------------------
# 1. Generate two discrepant dipole maps
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 2. Run individual analyses
# -------------------------------------------------------------------------
savedir = '/tmp/example_tension'

print("\n--- Running Dataset A ---")
a_A = Analyser(Map=counts_A, D=D_survey, map_coords='G')
a_A.model(type='poisson')
a_A.ultranest(savedir=savedir, name='dataset_A', min_num_live_points=200, dlogz=1.0)

print("\n--- Running Dataset B ---")
a_B = Analyser(Map=counts_B, D=D_survey, map_coords='G')
a_B.model(type='poisson')
a_B.ultranest(savedir=savedir, name='dataset_B', min_num_live_points=200, dlogz=1.0)

# -------------------------------------------------------------------------
# 3. Run joint analysis (shared v, theta, phi)
# -------------------------------------------------------------------------
print("\n--- Running Joint A+B ---")
a_A_joint = Analyser(map=counts_A, D=D_survey, map_coords='G')
a_A_joint.model(type='poisson')
a_B_joint = Analyser(map=counts_B, D=D_survey, map_coords='G')
a_B_joint.model(type='poisson')
a_A_joint.add(a_B_joint)
a_A_joint.ultranest(savedir=savedir, name='dataset_A+dataset_B',
                    min_num_live_points=200, dlogz=1.0)

# -------------------------------------------------------------------------
# 4. Compute tension statistics
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TENSION STATISTICS")
print("=" * 60)
t = Tension(
    f'{savedir}/dataset_A',
    f'{savedir}/dataset_B',
    AB=f'{savedir}/dataset_A+dataset_B',
    coords='G'
)

print(f"logR  = {t.logR:7.2f} +{t.logR_err[0]:.2f} -{t.logR_err[1]:.2f}")
print(f"logI  = {t.logI:7.2f} +{t.logI_err[0]:.2f} -{t.logI_err[1]:.2f}")
print(f"logS  = {t.logS:7.2f} +{t.logS_err[0]:.2f} -{t.logS_err[1]:.2f}")
print(f"d     = {t.d:7.2f} +{t.d_err[0]:.2f} -{t.d_err[1]:.2f}")
print(f"p     = {t.p:7.4f}")
print(f"sigma = {t.sigma:7.2f} +{t.sigma_err[0]:.2f} -{t.sigma_err[1]:.2f}")
print(f"\nExpect: significant tension (logS << 0, sigma >> 2)")

# -------------------------------------------------------------------------
# 5. Posterior plots: individual corner plots
# -------------------------------------------------------------------------
p_A = Posterior(f'{savedir}/dataset_A', coords='G')
p_B = Posterior(f'{savedir}/dataset_B', coords='G')
p_AB = Posterior(f'{savedir}/dataset_A+dataset_B', coords='G')

try:
    p_A.corner()
    plt.suptitle('Dataset A posterior', y=1.02)
    plt.savefig('example_outputs/06_corner_A.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\nSaved 06_corner_A.png")
except Exception as e:
    print(f"Corner A skipped: {e}")

try:
    p_B.corner()
    plt.suptitle('Dataset B posterior', y=1.02)
    plt.savefig('example_outputs/06_corner_B.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 06_corner_B.png")
except Exception as e:
    print(f"Corner B skipped: {e}")

try:
    p_AB.corner()
    plt.suptitle('Joint A+B posterior (forced agreement)', y=1.02)
    plt.savefig('example_outputs/06_corner_joint.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 06_corner_joint.png")
except Exception as e:
    print(f"Joint corner skipped: {e}")

# -------------------------------------------------------------------------
# 6. Sky plot: overlay both individual posteriors
# -------------------------------------------------------------------------
try:
    p_A.sky(color='cornflowerblue')
    plt.title('Dataset A sky posterior')
    plt.savefig('example_outputs/06_sky_A.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 06_sky_A.png")
except Exception as e:
    print(f"Sky A skipped: {e}")

try:
    p_B.sky(color='orange')
    plt.title('Dataset B sky posterior')
    plt.savefig('example_outputs/06_sky_B.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 06_sky_B.png")
except Exception as e:
    print(f"Sky B skipped: {e}")

# -------------------------------------------------------------------------
# 7. Table output
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TABLE OUTPUT")
print("=" * 60)
print(f"A:     {p_A.table()}")
print(f"B:     {p_B.table()}")
print(f"A+B:   {t.table()}")

print("\nDone!")

# -------------------------------------------------------------------------
# Legacy API reference (kept for context; not run)
# -------------------------------------------------------------------------
# a_joint = Analyser(Map=counts_A, D=D_survey, Map2=counts_B, D2=D_survey, map_coords='G')
# a_joint.model(type='poisson')
# a_joint.model2(type='poisson', shared_parameters=['v', 'theta', 'phi'])
# a_joint.ultranest(savedir=savedir, name='dataset_A+dataset_B', ...)
