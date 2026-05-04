"""Example 4: Expected dipole amplitude from a catalogue.

Demonstrates ``Analyser.expected_amplitude`` — the Ellis+Baldwin / CMB-rest-
frame prediction for the kinematic dipole amplitude D_CMB given a catalogue
of source fluxes (and their errors) and a flux cut.

Two-stage flow:
  1. Call without ``cutoff`` to inspect amp(F) and pick a sensible lower
     fit boundary above the survey's incompleteness roll-off.
  2. Call again with ``cutoff=...`` (mJy offset above the flux cut) to
     run the full MC and get D_CMB. The recovered value is also
     auto-stored on ``Analyser.D`` if not already set.

Test target: RACS-low1 with the galactic mask, 15 mJy flux cut, cutoff=5.
Expected D ≈ 4.27e-3.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from dipoletools import MapMaker, MaskMaker, Analyser

os.makedirs('example_outputs', exist_ok=True)

# -------------------------------------------------------------------------
# 1. Load RACS-low1 with the columns we need (incl. flux_err)
# -------------------------------------------------------------------------
# The default 'racs-low1' shorthand catalogue (allsources.csv) doesn't
# carry the per-source flux uncertainty, so we point at the fuller
# galacticcut catalogue and supply labels explicitly.
racs_path = '/Users/mali/repos/datastore/RACS-low1_sources_25arcsec_galacticcut.fits'

mm = MapMaker((racs_path, {
    'ra': 'ra', 'dec': 'dec',
    'flux': 'total_flux_source',
    'flux_err': 'e_total_flux_source',
    'id': 'source_id',
}))
mm.cut('flux', min=15)   # 15 mJy survey flux cut
print(f'After cut: {len(mm.catalogue())} sources')

# -------------------------------------------------------------------------
# 2. Apply the galactic mask and build the Analyser (no D yet)
# -------------------------------------------------------------------------
mask = MaskMaker('racs-low1')
a = Analyser(mm, mask=mask)

# -------------------------------------------------------------------------
# 3. Stage one: pick a cutoff by eye
# -------------------------------------------------------------------------
# With cutoff=None this runs 5 quick MC passes and shows amp vs flux
# limit so we can see where the curve flattens out above the survey
# incompleteness. No amplitude is returned in this mode.
a.expected_amplitude(alpha=0.75, seed=1)
plt.savefig('example_outputs/04_inspect.png', dpi=100, bbox_inches='tight')
plt.close()
print('Saved 04_inspect.png — pick a cutoff above the roll-off (~5 mJy here).')

# -------------------------------------------------------------------------
# 4. Stage two: full MC with the chosen cutoff
# -------------------------------------------------------------------------
D, D_std = a.expected_amplitude(alpha=0.75, cutoff=5,
                                return_std=True, seed=1)
plt.savefig('example_outputs/04_result.png', dpi=100, bbox_inches='tight')
plt.close()
print(f'D_CMB = {D:.4e} +/- {D_std:.2e} (target ~4.27e-3)')
print(f'Analyser.D auto-set to {a.D:.4e}')
