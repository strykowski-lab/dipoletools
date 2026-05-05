# GPU acceleration for `Analyser.blackjax(...)`

`Analyser.blackjax(...)` runs nested sampling through JAX/BlackJAX. The same
code path runs on CPU, NVIDIA GPU, and Apple Silicon — what changes is which
JAX backend you install and a few performance knobs.

## Install the optional extra

```bash
pip install "dipoletools[blackjax]"
```

This pulls in `jax` and the nested-sampling-enabled fork of `blackjax`. By
default that gives you a CPU JAX. To get GPU speedups, install one of the
device-specific JAX builds below **after** installing the extra.

## Apple Silicon (M-series)

```bash
pip install jax-metal
```

Verify:

```python
import jax
print(jax.devices())  # -> [METAL(id=0)]
```

**float32 fallback.** `jax-metal` does not support float64. When dipoletools
detects the Metal backend, it switches BlackJAX to float32 and emits a single
`UserWarning` explaining the precision implication. The `.ultranest(...)` path
keeps using float64 throughout. If you need full float64 precision for a
specific run, fall back to `.ultranest(...)` or run on CPU/CUDA.

## NVIDIA cluster GPUs

```bash
pip install -U "jax[cuda12]"
```

Verify:

```python
import jax
print(jax.devices())  # -> [CudaDevice(id=0), ...]
```

Float64 is fully supported.

**Sharing a GPU.** If your cluster shares GPUs across users or jobs, set:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5  # or whatever fraction you've been allocated
```

By default JAX preallocates ~75% of GPU memory; the env var caps that.

## CPU fallback

```bash
pip install -U jax
```

This runs but defeats the purpose — for CPU NS, `.ultranest(...)` is faster
and more battle-tested. The CPU JAX path is mainly useful for development
and parity testing.

## Tuning `n_delete`

`Analyser.blackjax(n_delete=...)` controls how many particles are deleted and
replaced per NS step. This is the main GPU-throughput knob:

- Larger `n_delete` → more parallel work per step → better GPU utilisation,
  fewer kernel launches, faster wall-clock time per step.
- The trade-off is statistical efficiency: deleting too many at once
  smooths out the contraction and slightly inflates the evidence error.
- Rough rule of thumb: `n_delete = n_live / 10` is a reasonable starting
  point; bump it up until your GPU is saturated.
