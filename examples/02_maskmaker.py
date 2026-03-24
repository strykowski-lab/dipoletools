"""Example 2: MaskMaker — Creating and combining HEALPix masks.

Demonstrates:
- Creating a MaskMaker with a given nside and coordinate system
- Masking the galactic plane with slices
- Masking point sources with discs
- Loading an external mask file
- Combining multiple masks (AND logic)
- Visualising the mask
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from dipoletools import MaskMaker

# -------------------------------------------------------------------------
# 1. Basic initialisation
# -------------------------------------------------------------------------
m = MaskMaker()  # default nside=64, coords='G'

# By default, all pixels are unmasked (True = keep)
print(f"Nside: {m.nside}")
print(f"Total pixels: {len(m.mask)}")
print(f"Unmasked pixels: {np.sum(m.mask)}")
print(f"Coordinate system: {m.coords()}")

# -------------------------------------------------------------------------
# 2. Mask the galactic plane: |b| < 10 degrees
# -------------------------------------------------------------------------
m.slices('|b| < 10')
print(f"\nAfter |b| < 10 cut:")
print(f"  Unmasked: {np.sum(m.mask)} / {len(m.mask)}")
print(f"  Masked fraction: {1 - np.mean(m.mask):.1%}")

# -------------------------------------------------------------------------
# 3. Mask point sources with circular discs
# -------------------------------------------------------------------------
# Mask 3 bright sources at known (l, b) positions with 5-degree radii
sources = [(80.0, -33.0), (201.0, -37.0), (140.0, 40.0)]
m.discs(sources, radii=5.0)
print(f"\nAfter disc masking:")
print(f"  Unmasked: {np.sum(m.mask)} / {len(m.mask)}")

# -------------------------------------------------------------------------
# 4. Two-sided slices
# -------------------------------------------------------------------------
# Mask a strip in longitude: 10 < l < 20
m.slices('10 < l < 20')
print(f"\nAfter longitude strip cut:")
print(f"  Unmasked: {np.sum(m.mask)} / {len(m.mask)}")

# -------------------------------------------------------------------------
# 5. Switching coordinate systems
# -------------------------------------------------------------------------
# You can set the coordinate system and then use those coordinates in slices.
# For example, mask declinations below -40 in celestial coordinates:
m.coords('C')
m.slices('dec < -40')
print(f"\nAfter dec < -40 cut (celestial):")
print(f"  Unmasked: {np.sum(m.mask)} / {len(m.mask)}")
print(f"  Current coords: {m.coords()}")

# -------------------------------------------------------------------------
# 6. Visualise
# -------------------------------------------------------------------------
# Convert mask to float for plotting (1=keep, NaN=masked)
mask_plot = np.where(m.mask, 1.0, np.nan)
hp.mollview(mask_plot, title='Combined Mask', cmap='Greys', cbar=False)
hp.graticule()
plt.savefig('example_outputs/02_mask_output.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nSaved 02_mask_output.png")
