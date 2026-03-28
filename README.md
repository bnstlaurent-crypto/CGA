# CGA Simulation Suite

**Companion code for:**
> B. St. Laurent, "Quantum Branched Flow: Coherence Graph Dynamics and the Spectral Geometry of Decoherence" (2026)

**Zenodo deposit:** https://zenodo.org/[DEPOSIT-DOI]

---

## Overview

This repository contains every simulation used to produce the numerical results
in the CGA paper series. The suite is structured as a single self-contained
Python module so that any reader can run it top-to-bottom and reproduce all
reported results independently.

The suite covers **37 results** across Papers 1-4 of the CGA series:

| Parts | Results | Description |
|-------|---------|-------------|
| 1-3 | (setup) | Hamiltonian/Laplacian construction, Lindblad dynamics, sector detection |
| 4-6 | 1-9 | Framework validation, 250-Hamiltonian ensemble, environmental regimes |
| 7-9 | 10-12 | Fringe visibility identity, Bell correlator, spectral transition |
| 10-12 | 13-18 | Perturbation robustness, formation diagnostics, Paper 3 validation |
| 13 | 19-21 | Universality and N-scaling tests |
| 14 | (runner) | Master runner — executes all results in sequence |
| 15 | 22-25 | Tier 1 stress tests (reviewer objections) |
| 16 | 26-29 | Tier 2 stress tests (physical Hamiltonians, finite-size scaling) |
| 17 | 30-32 | Tier 3 stress tests (topology interpolation, non-Markovian stub) |
| 18 | 33-37 | Paper 2 spectral validation and Paper 4 weight conservation |

---

## Dependencies

```
Python >= 3.8
numpy >= 1.21
scipy >= 1.7
matplotlib >= 3.4
```

No custom libraries. No physics-specific packages that could contain hidden assumptions. Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Running

```bash
python cga_simulation_suite.py
```

This runs all 37 results sequentially and saves figures to the `figures/` subdirectory
(created automatically at runtime; not included in this deposit). Runtime: approximately 20–25 minutes depending on hardware.

**Reproducibility:** All results use `RANDOM_SEED = 42`. Re-running on any
platform with the same dependencies should produce numerically identical results.

---

## Key claims confirmed numerically

- **250/250**: Fiedler vector sign partition = branch sector assignment, across
  250 block-structured random Hamiltonians with varied matrix elements (Result 6)
- **62/62**: Conditional precision — Fiedler accuracy = 1.000 in every trial
  where two-sector formation occurs (Results 19–21)
- **Exact**: Fringe visibility $V(t) = |\rho_{LR}(t)|$ to residual $1.98 \times 10^{-5}$
  (screen discretisation only) (Result 10)
- **Machine precision**: Bell correlator $S_\max = 2\sqrt{1+V^2}$ to
  $8.88 \times 10^{-16}$ (Result 11)
- **Negative results**: Uniform dephasing, energy eigenbasis, path/cycle/star
  graph topologies — all correctly predicted to not produce stable two-sector
  structure (Results 22–25, 31)
- **Paper 2 spectral chain**: Liouvillian gap Δ ≥ Cheeger lower bound confirmed
  across 5 coupling regimes (ratio 8–8000×); self-adjointness of L in KMS inner
  product confirmed to $7 \times 10^{-15}$ (Result 33)
- **First-order mixing**: $J^{(1)} = 0$ to machine precision ($< 10^{-14}$) for all
  20 random perturbations, exact and from dynamics (Result 35, Lemma 6.1)
- **N-sector**: 20/20 perfect 3-sector detection, 20/20 perfect 4-sector detection
  via generalized Fiedler eigenvectors; partition stable under perturbation (Result 36)
- **Sector weight conservation**: $dw_A/dt = 2 \cdot \sum \mathrm{Im}(H_{ij} \rho_{ji})$ confirmed two ways
  (Result 37): algebraic consistency check (both sides from same $\rho$) to $2.6 \times 10^{-18}$
  (machine precision, confirms identity is correctly implemented); empirical
  finite-difference check on ODE trajectory to $\sim 10^{-3}$--$10^{-5}$ (ODE truncation-limited,
  confirms identity holds in dynamics); intra-sector cancellation exact; Paper 4 Theorem 3.1

---

## Structure of the code

Each logical step includes a `# NOTE:` block in plain English explaining:
- what the code is computing
- what physical assumption is encoded
- what it would mean if the step were wrong

The intent is to make all assumptions auditable by a reader with no prior knowledge of the CGA framework.

---

## Stubs (deferred to companion papers)

- **Results 17-18**: Paper 3 validation (decoherence operator and discrimination margin tests) - stubs included, full implementation in Paper 3 companion code
- **Result 32**: Non-Markovian perturbation test — stub included, deferred to future work

---

## Files in this deposit

| File | Description |
|------|-------------|
| `cga_simulation_suite.py` | Main simulation suite (all 37 results) |
| `cga_numerical_companion.pdf` | Numerical companion document (all figures and result details) |
| `cga_paper1.pdf` | Paper 1 (foundation paper) |
| `README.md` | This file |

---

## Citation

If you use this code, please cite the companion paper:

```
B. St. Laurent, "Quantum Branched Flow: Coherence Graph Dynamics and the
Spectral Geometry of Decoherence," Zenodo preprint (2026).
DOI: https://zenodo.org/[DEPOSIT-DOI]
```

---

## License

This deposit is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. You are free to share and adapt the material for any purpose, provided appropriate credit is given.

---

## Contact

Brian St. Laurent — bnstlaurent@gmail.com
