"""Example 1: MapMaker — Loading catalogues, cutting, and making HEALPix maps.

Demonstrates:
- Loading a catalogue (auto-detecting known formats)
- Applying flux cuts
- Binning sources into a HEALPix map
- Coordinate system handling

Note: This example uses dummy data since the real catalogues are large.
For real data, you would use paths like:
    m = MapMaker('/path/to/AS110_Derived_Catalogue_racs_dr1_sources_*.csv')
or use shorthand names (configure this in dipoletools/_defaults.py), like:
    m = MapMaker('racs-low1')
"""

import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from dipoletools import MapMaker

# -------------------------------------------------------------------------
# 1. Create a dummy catalogue
# -------------------------------------------------------------------------
np.random.seed(42)
n_sources = 50000

# Random positions on the sky (celestial coordinates)
ra = np.random.uniform(0, 360, n_sources)
dec = np.degrees(np.arcsin(np.random.uniform(-1, 1, n_sources)))
flux = np.random.lognormal(mean=2.0, sigma=1.0, size=n_sources)  # mJy

# Save as CSV
dummy_path = '/tmp/dummy_catalogue.csv'
pd.DataFrame({'ra': ra, 'dec': dec, 'flux': flux}).to_csv(dummy_path, index=False)
print(f"Created dummy catalogue with {n_sources} sources")

# -------------------------------------------------------------------------
# 2. Load catalogue into MapMaker
# -------------------------------------------------------------------------
mp = MapMaker()
mp.catalogue(dummy_path, labels={'ra': 'ra', 'dec': 'dec', 'flux': 'flux'})
cat = mp.catalogue()
print(f"\nLoaded catalogue:")
print(f"  Sources: {len(cat)}")
print(f"  Columns: {list(cat.columns)}")

# -------------------------------------------------------------------------
# 3. Apply a flux cut
# -------------------------------------------------------------------------
mp.cut('flux', min=5.0)
print(f"\nAfter flux > 5 mJy cut: {len(mp.catalogue())} sources")

# Cut can also use max:
mp.restore()
mp.cut('flux', min=5.0, max=100.0)
print(f"After 5 < flux < 100 mJy: {len(mp.catalogue())} sources")

# Restore original catalogue
mp.restore()
print(f"After restore: {len(mp.catalogue())} sources")

# Re-apply for the map
mp.cut('flux', min=5.0)

# -------------------------------------------------------------------------
# 4. Create a HEALPix map
# -------------------------------------------------------------------------
nside = 64
count_map = mp.map(nside=nside)
print(f"\nHEALPix map created:")
print(f"  Nside: {nside}")
print(f"  Non-zero pixels: {np.sum(count_map > 0)}")
print(f"  Mean counts/pixel: {np.nanmean(count_map[count_map > 0]):.1f}")

# -------------------------------------------------------------------------
# 5. Visualise
# -------------------------------------------------------------------------
hp.mollview(count_map, title='Source count map (nside=64)', cmap='viridis')
hp.graticule()
plt.savefig('example_outputs/01_mapmaker_output.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nSaved 01_mapper_output.png")
