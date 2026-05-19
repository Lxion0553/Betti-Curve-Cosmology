# Betti-Curve-Cosmology

Code and data used in the paper [**Counting voids and filaments: Betti Curves as a Topological Probe of Cosmology**](https://arxiv.org/abs/2512.07236)

This repository includes:
1. Betti curve generation from halo catalogs.
2. Auto-sklearn emulators for Betti curves and power spectra.
3. Nested sampling (UltraNest) for cosmological parameter inference.

## Repository layout
1. **BettiCurveCalaulation/**: persistence/Betti-curve utilities and generation scripts.
2. **Emulator/**: emulator training and prediction utilities (GPR and Auto-sklearn).
3. **MCMC/**: UltraNest-based parameter inference and plotting scripts.
4. **Data/**: processed data products, emulator inputs, and test data. The original simulation data is not included, which can be obtained from the Quijote simulations (https://quijote-simulations.readthedocs.io/).
5. **PowSpec/**: power-spectrum utilities.
6. **Fig/**: figures used in the paper.
