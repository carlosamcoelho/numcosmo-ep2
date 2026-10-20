# Numerical Cosmology — EP2

This repository contains the implementation and numerical experiments for
**EP2** developed as an exercise for the Numerical Cosmology class at IF-USP, 
São Paulo (SP), Brazil.

The project covers linear perturbation growth, Standard Perturbation Theory,
Lagrangian Perturbation Theory, spherical collapse, and validation of
CLASS and CAMB Einstein–Boltzmann solvers.

Boltzmann solvers used for Exercise 5:
- CLASS commit 64bbab707faf4de4779a9e04edd180fef18d98fa
- CAMB 2.0.4

The numerical work is divided according to the exercises of the assignment.

## Exercise 1 - Linear growth

Numerical calculation and validation of the linear growth of matter
perturbations in an expanding cosmological background.

The soruce code `src/numcosmo_ep2/growth.py` is followed by a corresponding 
notebook `notebooks/01_growth.ipynb` that contains the numerical solution 
and comparisons with the expected limiting behaviour.

## Exercise 2 - Standard Pertubation Theory

Implementation of the second-order density field in Eulerian Standard
Perturbation Theory (SPT),

$$
\delta = \delta^{(1)} + \delta^{(2)}.
$$

The implementation includes:
- the EdS second-order kernel $F_2$ from mode-coupling;
- a pseudospectral real-space evaluation of $\delta^{(2)}$;
- Fourier-space de-aliasing;
- auto- and cross-power spectra;
- higher-order statistics, including skewness and reduced skewness $S_3$;
- ensemble tests over Gaussian realizations;
- convergence tests with box size and spatial resolution;
- perturbative amplitude-scaling tests.

The main implementation is in `src/numcosmo_ep2/spt.py` and tested in 
`notebooks/02_spt.ipynb`.

## Exercise 3 - Lagrangian Pertubation Theory

Generation of Gaussian density fields and implementation of first-order
Lagrangian Perturbation Theory using the Zel'dovich approximation:

$$
\mathbf{\Psi}_\mathbf{k}^{(1)}(a_i)=i D_1(a_i)\frac{\mathbf{k}}{k^2}\delta_{\mathrm{ref,}\mathbf{k}},
$$

and displacement of a grid of particle samples of $\mathbf{x}_p=\mathbf{q}_p+\mathbf{\Psi}_\mathbf{k}^{(1)}$, $\mathbf{v}_p=\mathcal{H}_i f_{\mathrm{1,}i}\mathbf{\Psi}_\mathbf{k}^{(1)}$.

The implementation includes:
- Gaussian random fields generated from an input linear power spectrum;
- the first-order displacement field;
- particle displacement with the Zel'dovich approximation;
- CIC deposition on a regular mesh;
- power-spectrum measurements;
- resolution and realization tests.

The main implementation is in `src/numcosmo_ep2/lpt.py`, with a corresponding
notebook analysis in `notebooks/03_lpt.ipynb`. We check that the procedure generates
a gaussian field and particle displacements that follow the input power spectra
away from the Nyquist frequency.

## Exercise 4 - Spherical collapse

Numerical integration of the spherical-collapse equations, validation and 
comparison with the standard analytical expectations for an Einstein-de Sitter
(EdS) universe. The numerical implementation follows three independent ODE
formulations for the spherical collapse from Pace, Meyer & Bartelmann (2017).

The corresponding routines are contained in `src/numcosmo_ep2/collapse.py`
and tested in `notebooks/04_collapse.ipynb`, where we also explain the three 
equivalent ODEs that specify the methods A, B, C. We also compare this evolution
with the linear prediction counterpart, find turnaround and collapse events, 
compare results with a virial prescription, and compare convergence of the result
and their dependence with cosmological parameters.

## Exercise 5 - Einstein-Boltzmann solver validation

This exercise uses existing Einstein-Boltzmann solvers (CLASS and CAMB) rather 
than implementing a new solver, since it is outside of the scope for this class.

The analysis contains three parts:

1. Recontruction of the CLASS linear matter power spectrum from the density transfer
   function and primordial curvature spectrum;
2. Numerical convergence tests of CLASS by varying quantities such as precision, 
   sampling, hierarchy and approximation-switch;
3. And independent comparision between CLASS and CAMB after matching cosmological
   parameters, matter definition, units, and conventions.

The fiducial cosmology is:

    h          = 0.674
    omega_b    = 0.0224
    omega_cdm  = 0.120
    Omega_k    = 0
    A_s        = 2.10e-9
    n_s        = 0.965
    k_pivot    = 0.05 Mpc^-1
    tau_reio   = 0.054
    sum m_nu   = 0.06 eV
    z          = 0

## Tests

Unit and numerical consistency tests are provided in `tests/`.

They can be executed from the repository root with

    pytest -v

for a detailed output. The tests cover the main numerical operators, limiting and
analytical cases of the implementations of SPT, LPT, growth and collapse.

## Notebooks

The `notebooks/` directory contains the numerical experiments and figures for each 
exercise. It also includes markdowns commenting each result, or giving the physical
framework and considerations used for each section.

The notebooks are intended to be run from the repository environment after
installing the package.

## Outputs

Generated plots and numerical tables are stored under `outputs/figures/` and 
`outputs/tables/`, respectively. 

The generated data from the CLASS executions are provided at `outputs/class/`, as to 
be used in the analysis from `notebooks/05_boltzmann.ipynb`. 

## Notes on numerical conventions

Fourier, power-spectrum, mesh, and unit conventions are documented directly
in the corresponding modules and notebooks.
