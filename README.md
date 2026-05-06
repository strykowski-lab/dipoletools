# dipoletools

A Python toolkit for measuring and analysing cosmic dipoles in source-count and temperature maps using Bayesian nested sampling.

`dipoletools` provides an end-to-end pipeline for:

- Loading radio/infrared/CMB catalogues and binning them into HEALPix maps
- Creating and combining sky masks (galactic plane cuts, point-source excision, coordinate-based slicing)
- Fitting dipole (and higher multipole) models to pixelised count maps via nested sampling — choice of [UltraNest](https://johannesbuchner.github.io/UltraNest/) (CPU) or [BlackJAX](https://blackjax-devs.github.io/blackjax/) (GPU-accelerated, via JAX)
- Analysing posterior chains: corner plots, sky projections, formatted LaTeX tables
- Computing Bayesian tension statistics (log-Bayes ratio, suspiciousness, calibrated sigma) between datasets

## Installation

Default install (UltraNest only — CPU nested sampling):

```bash
git clone https://github.com/strykowski-lab/dipoletools.git
cd dipoletools
pip install -e ".[dev,examples]"
```

Add the `blackjax` extra for GPU-accelerated nested sampling via JAX/BlackJAX:

```bash
pip install -e ".[dev,examples,blackjax]"
```

For Apple Silicon (Metal) or NVIDIA CUDA GPUs you also need a device-specific
JAX build.

### Requirements

- Python >= 3.10
- numpy, scipy, healpy, astropy, pandas, matplotlib
- ultranest, h5py, getdist, anesthetic
- `[blackjax]` extra: jax, blackjax (handley-lab `nested_sampling` branch)

## Quick start

### 1. Build a source-count map

```python
from dipoletools import MapMaker

mm = MapMaker()
mm.catalogue('my_catalogue.fits', labels={'ra': 'RA', 'dec': 'DEC', 'flux': 'FLUX'})
mm.cut('flux', min=15, max=1000)
count_map = mm.map(nside=64)
mm.show()
```

### 2. Create a mask

```python
from dipoletools import MaskMaker

mask = MaskMaker()
mask.slices('|b| < 10')                                    # galactic plane
mask.discs([(80.9, 69.1), (201.4, -43.0)], radii=5.0)     # bright sources
mask.show()
```

### 3. Fit a dipole model

```python
from dipoletools import Analyser

a = Analyser(map=count_map, mask=mask, D=0.0046, map_coords='G')
a.model(type='poisson', ell=[0, 1])
a.ultranest(savedir='results/my_run', min_num_live_points=400)
```

For GPU-accelerated nested sampling on the same model, use `.blackjax(...)`
instead. It supports `gaussian`, `poisson`, and `general_poisson` (with the
ecliptic-bias correction); for joint analyses it forces `v, theta, phi`
shared. Use `.ultranest(...)` for ell≥2, second-dipole models, or custom
shared/unshared layouts.

```python
a.blackjax(savedir='results/my_run', n_live=500, n_delete=50, seed=0)
```

### 4. Analyse the posterior

```python
from dipoletools import Posterior

p = Posterior('results/my_run', coords='G')
p.corner()                    # GetDist corner plot
p.sky()                       # Mollweide sky projection with contours
print(p.table())              # Formatted LaTeX table row
```

### 5. Compute tension between datasets

```python
from dipoletools import Tension

t = Tension('results/run_A', 'results/run_B', AB='results/run_A+run_B')
print(f"sigma = {t.sigma:.2f}")
print(f"logS  = {t.logS:.2f}")
print(t.table())
```

## Core classes

| Class | Purpose |
|---|---|
| `MapMaker` | Load catalogues, apply cuts, crossmatch, bin into HEALPix maps |
| `MaskMaker` | Create boolean HEALPix masks from coordinate slices, discs, or files |
| `Analyser` | Configure dipole/multipole models and run nested sampling |
| `Posterior` | Load chains, produce corner plots, sky plots, and LaTeX tables |
| `Tension` | Compute Bayesian tension statistics (log R, log I, log S, sigma) between two datasets |

## Supported models

- **Poisson** (`type='poisson'`): Standard Poisson likelihood for integer count maps
- **General Poisson** (`type='general_poisson'`): Consul-Jain generalised Poisson with a dispersion parameter
- **Gaussian** (`type='gaussian'`): Gaussian likelihood for continuous data (e.g. CMB temperature maps)

All models support:
- Arbitrary multipole combinations via `ell=[0, 1, 2, ...]`
- Ecliptic latitude bias correction (`bias=True`)
- Joint N-dataset analysis with shared parameters
- Adding an additional dipole (with optional fixed parameters)
- Custom likelihood functions

## Coordinate systems

dipoletools supports three coordinate systems throughout:

- `'C'` — Celestial / Equatorial (ICRS): RA, Dec
- `'G'` — Galactic: l, b
- `'E'` — Ecliptic: lon, lat

All classes provide a `.coords()` method to query, label, or convert between systems.

## Examples

The `examples/` directory contains worked scripts demonstrating the full pipeline:

| Script | Description |
|---|---|
| `01_mapmaker.py` | Loading catalogues, flux cuts, HEALPix binning |
| `02_maskmaker.py` | Galactic plane masks, disc masking, coordinate slices |
| `03_analyser_single.py` | Single-dataset dipole fit with synthetic data |
| `04_expected_amplitude.py` | Compute D_CMB from the source catalogue with a flux cut |
| `05_model_types.py` | Comparing Poisson, general Poisson, Gaussian, and bias models |
| `06_posterior_tension.py` | Loading real chains, posterior analysis, tension computation |
| `07_joint_tension.py` | Joint analysis of two discrepant datasets (high tension) |
| `08_joint_concordance.py` | Joint analysis of two concordant datasets (low tension) |
| `09_two_dipoles.py` | Add a second dipole moment to the expected counts |

## Tension statistics

The `Tension` class computes the following quantities from nested sampling evidence and information:

- **log R** — log Bayes ratio: `log Z_AB - log Z_A - log Z_B`
- **log I** — log information ratio: `D_KL(A) + D_KL(B) - D_KL(AB)`
- **log S** — log suspiciousness: `log R - log I`
- **d** — effective dimensionality of the shared parameter space
- **p** — calibrated p-value from the suspiciousness
- **sigma** — tension in units of Gaussian standard deviations

## Citation

If you use `dipoletools` in your research, please include a footnote to the repository:

> [https://github.com/strykowski-lab/dipoletools](https://github.com/strykowski-lab/dipoletools)

BibTeX:

```bibtex
@software{dipoletools,
  author       = {Land-Strykowski, Mali},
  title        = {dipoletools: A toolkit for measuring and analysing cosmic dipoles},
  url          = {https://github.com/strykowski-lab/dipoletools},
  version      = {0.1.0},
  year         = {2026},
}
```

## License

MIT
