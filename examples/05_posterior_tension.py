"""Example 5: Posterior and Tension — Loading chains and computing statistics.

Demonstrates:
- Loading a Posterior from a chain directory
- Accessing logZ, KL divergence, Bayesian dimensionality
- Coordinate conversions on chains
- Table formatting
- Computing Tension statistics between two datasets
- All tension quantities: logR, logI, logS, d, p, sigma
"""

from dipoletools import Posterior, Tension

# -------------------------------------------------------------------------
# Use existing reference chains
# -------------------------------------------------------------------------
ref_dir = '/Users/mali/repos/dipole_tensions/un_nvssracs_referee3_new2026'

# -------------------------------------------------------------------------
# 1. Load a single Posterior
# -------------------------------------------------------------------------
print("=" * 60)
print("POSTERIOR: RACS A (equatorial)")
print("=" * 60)
p = Posterior(f'{ref_dir}/racsa_equatorial', coords='C')

print(f"Coordinate system: {p.coords()}")
print(f"Parameters: {p._param_names}")
print(f"Samples: {p._samples.shape}")
print(f"logZ = {p.logZ:.2f} +{p.logZ_err[0]:.2f} -{p.logZ_err[1]:.2f}")
print(f"KL divergence = {p.kl:.2f} +{p.kl_err[0]:.2f} -{p.kl_err[1]:.2f}")
print(f"Bayesian dim. = {p.d:.2f} +{p.d_err[0]:.2f} -{p.d_err[1]:.2f}")
print(f"\nTable (celestial): {p.table()}")

# Convert to galactic and re-display
p.coords('C', 'G')
print(f"\nCoords after conversion: {p.coords()}")
print(f"Table (galactic): {p.table()}")

# -------------------------------------------------------------------------
# 2. Compare multiple posteriors
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("INDIVIDUAL POSTERIORS SUMMARY")
print("=" * 60)
for name in ['racsa_equatorial', 'racsb_equatorial',
             'nvssa_equatorial', 'nvssb_equatorial', 'planck', 'catwise']:
    pi = Posterior(f'{ref_dir}/{name}')
    print(f"  {name:25s}  logZ={pi.logZ:12.2f}  d={pi.d:.2f}  KL={pi.kl:.2f}")

# -------------------------------------------------------------------------
# 3. Tension between two datasets
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TENSION: NVSS A vs RACS A")
print("=" * 60)
t = Tension(f'{ref_dir}/nvssa_equatorial', f'{ref_dir}/racsa_equatorial')

print(f"logR  = {t.logR:6.2f} +{t.logR_err[0]:.2f} -{t.logR_err[1]:.2f}")
print(f"logI  = {t.logI:6.2f} +{t.logI_err[0]:.2f} -{t.logI_err[1]:.2f}")
print(f"logS  = {t.logS:6.2f} +{t.logS_err[0]:.2f} -{t.logS_err[1]:.2f}")
print(f"d     = {t.d:6.2f} +{t.d_err[0]:.2f} -{t.d_err[1]:.2f}")
print(f"p     = {t.p:6.4f} +{t.p_err[0]:.4f} -{t.p_err[1]:.4f}")
print(f"sigma = {t.sigma:6.2f} +{t.sigma_err[0]:.2f} -{t.sigma_err[1]:.2f}")
print(f"\nTension table: {t.table()}")

# -------------------------------------------------------------------------
# 4. All tension pairs
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL TENSION PAIRS")
print("=" * 60)
pairs = [
    ('planck', 'racsa_equatorial'),
    ('planck', 'nvssa_equatorial'),
    ('nvssa_equatorial', 'racsa_equatorial'),
    ('catwise', 'racsa_equatorial'),
    ('catwise', 'planck'),
]
for a, b in pairs:
    try:
        ti = Tension(f'{ref_dir}/{a}', f'{ref_dir}/{b}')
        print(f"  {a}+{b}: {ti.sigma:.2f} sigma")
    except Exception as e:
        print(f"  {a}+{b}: ERROR - {e}")

print("\nDone!")
