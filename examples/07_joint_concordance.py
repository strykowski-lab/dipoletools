"""Example 7: Joint analysis — Two datasets in CONCORDANCE.

Generates two synthetic Poisson dipole maps that both point in the same
direction on the sky (with independent noise realisations), runs individual
and joint nested sampling, then computes tension statistics.

Both datasets share the same underlying dipole direction (l=264, b=48),
mimicking the CMB dipole. The tension statistics should show:
- logR > 0 (data prefer the joint model)
- sigma << 2 (no significant tension)
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
# 1. Generate two concordant dipole maps
# -------------------------------------------------------------------------
np.random.seed(2026)
nside = 64
npix = hp.nside2npix(nside)
pos = np.array(hp.pix2vec(nside, np.arange(npix))).T

D_survey = 0.005

# Both datasets share the same dipole direction: (l=264, b=48)
theta_true = d2r(90 - 48)
phi_true = d2r(264)
v_true = 3.5

# Dataset A
N_A = 30.0
expected_A = N_A * (1 + v_true * D_survey * np.sum(ang2vec(theta_true, phi_true) * pos, axis=1))
counts_A = np.random.poisson(expected_A).astype(float)

# Dataset B: same direction, different N (independent realisation)
N_B = 22.0
expected_B = N_B * (1 + v_true * D_survey * np.sum(ang2vec(theta_true, phi_true) * pos, axis=1))
counts_B = np.random.poisson(expected_B).astype(float)

# Apply galactic plane mask
theta_pix, _ = hp.pix2ang(nside, np.arange(npix))
lat_pix = 90 - np.degrees(theta_pix)
gal_mask = np.abs(lat_pix) > 10
counts_A[~gal_mask] = np.nan
counts_B[~gal_mask] = np.nan

print(f"Both datasets: v={v_true}, direction=(l=264, b=48)")
print(f"  Dataset A: N={N_A}")
print(f"  Dataset B: N={N_B}")
print(f"  Unmasked pixels: {np.sum(gal_mask)}")

# -------------------------------------------------------------------------
# 2. Run individual analyses
# -------------------------------------------------------------------------
savedir = '/tmp/example_concordance'

print("\n--- Running Dataset A ---")
a_A = Analyser(Map=counts_A, D=D_survey, map_coords='G')
a_A.model(type='poisson')
a_A.ultranest(savedir=savedir, name='concordant_A', min_num_live_points=200, dlogz=1.0)

print("\n--- Running Dataset B ---")
a_B = Analyser(Map=counts_B, D=D_survey, map_coords='G')
a_B.model(type='poisson')
a_B.ultranest(savedir=savedir, name='concordant_B', min_num_live_points=200, dlogz=1.0)

# -------------------------------------------------------------------------
# 3. Run joint analysis (shared v, theta, phi)
# -------------------------------------------------------------------------
print("\n--- Running Joint A+B ---")
a_joint = Analyser(Map=counts_A, D=D_survey, Map2=counts_B, D2=D_survey, map_coords='G')
a_joint.model(type='poisson')
a_joint.model2(type='poisson', shared_parameters=['v', 'theta', 'phi'])
a_joint.ultranest(savedir=savedir, name='concordant_A+concordant_B',
                  min_num_live_points=200, dlogz=1.0)

# -------------------------------------------------------------------------
# 4. Compute tension statistics
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TENSION STATISTICS (CONCORDANT)")
print("=" * 60)
t = Tension(
    f'{savedir}/concordant_A',
    f'{savedir}/concordant_B',
    AB=f'{savedir}/concordant_A+concordant_B',
    coords='G'
)

print(f"logR  = {t.logR:7.2f} +{t.logR_err[0]:.2f} -{t.logR_err[1]:.2f}")
print(f"logI  = {t.logI:7.2f} +{t.logI_err[0]:.2f} -{t.logI_err[1]:.2f}")
print(f"logS  = {t.logS:7.2f} +{t.logS_err[0]:.2f} -{t.logS_err[1]:.2f}")
print(f"d     = {t.d:7.2f} +{t.d_err[0]:.2f} -{t.d_err[1]:.2f}")
print(f"p     = {t.p:7.4f}")
print(f"sigma = {t.sigma:7.2f} +{t.sigma_err[0]:.2f} -{t.sigma_err[1]:.2f}")
print(f"\nExpect: no tension (logS ~ 0, sigma < 2, logR > 0)")

# -------------------------------------------------------------------------
# 5. Posterior plots
# -------------------------------------------------------------------------
p_A = Posterior(f'{savedir}/concordant_A', coords='G')
p_B = Posterior(f'{savedir}/concordant_B', coords='G')
p_AB = Posterior(f'{savedir}/concordant_A+concordant_B', coords='G')

try:
    p_A.corner()
    plt.suptitle('Dataset A posterior', y=1.02)
    plt.savefig('example_outputs/07_corner_A.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\nSaved 07_corner_A.png")
except Exception as e:
    print(f"Corner A skipped: {e}")

try:
    p_B.corner()
    plt.suptitle('Dataset B posterior', y=1.02)
    plt.savefig('example_outputs/07_corner_B.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 07_corner_B.png")
except Exception as e:
    print(f"Corner B skipped: {e}")

try:
    p_AB.corner()
    plt.suptitle('Joint A+B posterior (concordant)', y=1.02)
    plt.savefig('example_outputs/07_corner_joint.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 07_corner_joint.png")
except Exception as e:
    print(f"Joint corner skipped: {e}")

# -------------------------------------------------------------------------
# 6. Sky plots
# -------------------------------------------------------------------------
try:
    p_A.sky(color='cornflowerblue')
    plt.title('Dataset A sky posterior')
    plt.savefig('example_outputs/07_sky_A.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 07_sky_A.png")
except Exception as e:
    print(f"Sky A skipped: {e}")

try:
    p_B.sky(color='orange')
    plt.title('Dataset B sky posterior')
    plt.savefig('example_outputs/07_sky_B.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved 07_sky_B.png")
except Exception as e:
    print(f"Sky B skipped: {e}")

# -------------------------------------------------------------------------
# 7. Tables
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TABLE OUTPUT")
print("=" * 60)
print(f"A:     {p_A.table()}")
print(f"B:     {p_B.table()}")
print(f"A+B:   {t.table()}")

# -------------------------------------------------------------------------
# 8. Comparison summary
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY: CONCORDANCE vs TENSION")
print("=" * 60)
print("For concordant data (same underlying dipole):")
print(f"  sigma = {t.sigma:.2f} (expect < 2)")
print(f"  logR  = {t.logR:.2f} (expect > 0, data prefer shared model)")
print(f"  logS  = {t.logS:.2f} (expect ~ 0, no suspiciousness)")
print("\nCompare with example 06 (discrepant data) for contrast.")

print("\nDone!")
