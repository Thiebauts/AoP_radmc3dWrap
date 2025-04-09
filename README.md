# RADMC-3D Iterative Dust Temperature Calculation for Water Fountains

This project provides a Python framework for running RADMC-3D dust temperature calculations iteratively with convergence checking, specifically optimized for "water fountain" astrophysical objects. These are a special type of evolved star with high-velocity water maser jets, typically consisting of a central star, a torus-like structure, and outflow lobes.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Water Fountain Model](#water-fountain-model)
- [Temperature-Dependent Dust Opacities](#temperature-dependent-dust-opacities)
- [SED Calculation](#sed-calculation)
- [Convergence Criteria](#convergence-criteria)
- [Usage Examples](#usage-examples)
- [Command Line Arguments](#command-line-arguments)
  - [Model Parameters](#model-parameters)
  - [Physics Parameters](#physics-parameters)
  - [Calculation Control](#calculation-control)
  - [Output & Visualization](#output--visualization)
- [Convergence Analysis](#convergence-analysis)
- [Outputs](#outputs)
- [References](#references)

## Overview

The code performs the following tasks:
1. Creates all necessary input files for RADMC-3D based on the water fountain physical model
2. Runs `radmc3d mctherm` with increasing photon numbers
3. Analyzes the dust temperature after each run to check for convergence
4. Visualizes and saves the results
5. Performs temperature-dependent dust opacity calculations by default
6. Computes spectral energy distribution (SED) after temperature calculation

## Quick Start

To get started with a basic water fountain model calculation:

```bash
# Install dependencies
pip install numpy matplotlib

# Run with default water fountain settings
python main_radmc3d.py

# Run with customized parameters (example)
python main_radmc3d.py --Mdtorus 0.01 --Rtorus 1200 --oangle 15

# Skip temperature calculation and just compute SED
python main_radmc3d.py --use_existing_temperature --compute_sed --inclination 45 --phi 90
```

For more options and detailed configurations, see the [Usage Examples](#usage-examples) and [Command Line Arguments](#command-line-arguments) sections.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- RADMC-3D (must be installed and accessible in your PATH)

## File Structure

- `main_radmc3d.py`: Main script that runs the iterative calculation
- `create_input.py`: Functions to create all RADMC-3D input files, including the water fountain model
- `radmc3d_aux.py`: Auxiliary functions for running RADMC-3D and analyzing outputs
- `dustkapscatmat_E40R_300K_a0.3.inp`: Standard dust opacity file
- `dustkapscatmat_E40R_10K_a0.3.inp`, `dustkapscatmat_E40R_100K_a0.3.inp`, `dustkapscatmat_E40R_200K_a0.3.inp`: Additional dust opacity files for temperature-dependent calculations

## Water Fountain Model

This implementation models water fountains with:

1. **Torus Component**: A dense torus-like structure defined by:
   - Base density and scaling parameters
   - Angular dependence to create the torus shape
   - Radial structure with exponential terms
   
2. **Outflow Lobes**: Bipolar outflow lobes defined by:
   - Characteristic size and opening angle
   - Density distribution with appropriate width parameter

3. **Stellar Source**: Central stellar source with:
   - Typical parameters for evolved stars (radius, temperature)
   - Appropriate spectral distribution

## Temperature-Dependent Dust Opacities

A major feature of this code is the ability to perform temperature-dependent dust opacity calculations. This works by:

1. Starting with a single dust species and running an initial temperature calculation
2. Redistributing dust density across multiple temperature-dependent opacity bins:
   - Cold dust (< 50K) → 10K dust opacities
   - Warm dust (50-150K) → 100K dust opacities
   - Hot dust (150-250K) → 200K dust opacities
   - Very hot dust (> 250K) → 300K dust opacities
3. Iteratively running temperature calculations and updating density distributions until convergence

This approach provides more physically accurate dust temperature calculations by accounting for temperature-dependent dust properties.

## SED Calculation

After computing the dust temperature (whether or not convergence is reached), the code can automatically compute the spectral energy distribution (SED) using RADMC-3D. You can specify the observer's position using:

- **Inclination**: Angle in degrees from the pole (default: 0°)
- **Phi**: Azimuthal angle in degrees (default: 0°)

The SED is calculated using the command `radmc3d sed incl [value] phi [value]` and the results are saved in the output directory.

## Convergence Criteria

The code uses two convergence criteria for temperature-dependent calculations:

1. **Mean Temperature Difference**: The mean relative temperature difference between iterations must fall below the specified threshold (default: 2%). This uses the average difference across all cells, which is less sensitive to outliers than the maximum difference.

2. **Cell Temperature Group Changes**: The percentage of cells that change their temperature group (e.g., from cold to warm) between iterations must fall below the specified threshold (default: 1%).

Both criteria must be satisfied for convergence, ensuring that not only have the temperature values stabilized, but also that the assignment of cells to temperature groups has stabilized.

## Usage Examples

### Basic Usage

To run the iterative calculation with default water fountain settings:

```bash
python main_radmc3d.py
```

### Customizing Model Structure

To customize the torus parameters:

```bash
python main_radmc3d.py --Mdtorus 0.01 --Rtorus 1200 --A 2 --B 4
```

To customize the outflow lobes:

```bash
python main_radmc3d.py --Mdlobe 0.001 --Rlobe 3000 --oangle 15
```

### Customizing Calculations

To customize the convergence parameters:

```bash
python main_radmc3d.py --nphotons_start 1e6 --nphotons_max 1e8 --threshold 0.005 --max_iterations 8
```

To disable temperature-dependent dust opacities (which are enabled by default):

```bash
python main_radmc3d.py --no_temp_dependent
```

To customize temperature-dependent dust opacity calculations:

```bash
python main_radmc3d.py --temp_ranges 50,150,250 --max_temp_iterations 5 --temp_threshold 0.05 --cells_change_threshold 0.5
```

### SED Calculation Options

To compute SED with specific viewing angles (using "sloppy" mode by default):

```bash
python main_radmc3d.py --compute_sed --inclination 45 --phi 90
```

To compute SED with more accurate but slower standard mode:

```bash
python main_radmc3d.py --compute_sed --no_sloppy --inclination 45 --phi 90
```

### Using Pre-computed Data

To skip temperature calculation and use an existing temperature file:

```bash
python main_radmc3d.py --use_existing_temperature --temperature_file path/to/dust_temperature.dat --compute_sed --inclination 45 --phi 90
```

**Important Note**: When using an existing `dust_temperature.dat` file, ensure that your `radmc3d.inp` and `dust_density.inp` files are consistent with it, particularly regarding the number of dust species. These three files are interdependent:
- `dust_temperature.dat` contains temperature data for each dust species
- `dust_density.inp` contains density data for each dust species
- `radmc3d.inp` must be configured for the same number of species

Inconsistencies between these files (e.g., different numbers of species) will cause RADMC-3D to fail or produce incorrect results.

### Visualization Options

To visualize results with different multi-species handling:

```bash
python main_radmc3d.py --advanced_plots --multi_species_handling all
```

## Command Line Arguments

### Model Parameters

#### Geometry Parameters
- `--rin`: Inner radius in AU (default: 100)
- `--rout`: Outer radius in AU (default: 5000)
- `--nr`: Number of radial grid cells (default: 1000)
- `--ntheta`: Number of theta grid cells (default: 150)

#### Star Parameters
- `--stellar_radius`: Radius in solar radii (default: 288)
- `--stellar_temp`: Temperature in K (default: 3000)

#### Torus Parameters
- `--Mdtorus`: Torus mass in solar masses (default: 0.005)
- `--Rtorus`: Characteristic radius in AU (default: 1000)
- `--A`: Torus angular density enhancement (default: 1, higher values create stronger torus concentration)
- `--B`: Radial density falloff rate (default: 3, higher values create steeper density falloff)
- `--C`: Angular dependence of radial falloff (default: 0, non-zero values create angle-dependent density falloff)
- `--D`: Outer edge sharpness (default: 10, higher values create sharper outer edge cutoff)
- `--E`: Radial dependence of angular structure (default: 3, controls how angular distribution changes with radius)
- `--F`: Angular profile shape (default: 2, controls smoothness of angular density distribution)

#### Outflow Lobe Parameters
- `--Mdlobe`: Outflow lobe mass in solar masses (default: 0.0005)
- `--Rlobe`: Characteristic radius in AU (default: 2500)
- `--oangle`: Opening angle in degrees (default: 18)
- `--width`: Width parameter (default: 0.005)

### Physics Parameters

#### Dust Opacity Parameters
- `--no_temp_dependent`: Disable temperature-dependent dust opacities (enabled by default)
- `--temp_ranges`: Temperature range boundaries for dust opacity selection (default: "50,150,250")
- `--dust_file`: Dust opacity file for single-species models (default: "dustkapscatmat_E40R_300K_a0.3.inp")

### Calculation Control

#### Convergence Parameters
- `--nphotons_start`: Initial number of photon packages (default: 1e4)
- `--nphotons_max`: Maximum number of photon packages (default: 1e8)
- `--scale_factor`: Factor to increase photons by each iteration (default: 2.0)
- `--threshold`: Convergence threshold for mean relative temperature difference in normal calculations (default: 2%)
- `--max_iterations`: Maximum number of iterations (default: 8)
- `--temp_threshold`: Convergence threshold for mean relative temperature difference in temperature-dependent opacity iterations (default: 2%)
- `--cells_change_threshold`: Convergence threshold for cells changing temperature groups (default: 1%)
- `--max_temp_iterations`: Maximum number of temperature-dependent opacity iterations (default: 8)
- `--setseed`: Random seed for reproducibility (default: None)
- `--density_weighted`: Use density-weighted convergence metrics (default: False)

#### SED Calculation Parameters
- `--no_compute_sed`: Disable SED computation after temperature calculation (enabled by default)
- `--inclination`: Observer's inclination angle in degrees (default: 0)
- `--phi`: Observer's azimuthal angle in degrees (default: 0)
- `--no_sloppy`: Disable the "sloppy" option in SED calculation for more accurate but slower results

By default, the code uses the "sloppy" flag with RADMC-3D, which uses a faster but less accurate integration for the SED calculation. This can significantly speed up computations for models with many cells or complex structures, at the cost of some accuracy. It's useful for quick previews or when absolute accuracy is not critical. For high-precision results, use the `--no_sloppy` option.

#### Existing Temperature File Options
- `--use_existing_temperature`: Skip temperature calculation and use existing dust_temperature.dat file
- `--temperature_file`: Path to existing temperature file (default: "dust_temperature.dat")

Note that when using these options, you must ensure all RADMC-3D input files are consistent with your temperature file:
1. The number of grid cells in `amr_grid.inp` must match the temperature data dimensions
2. The number of dust species in `dust_density.inp` must match the temperature data
3. The dust opacity settings in `dustopac.inp` must be configured for the correct number of species
4. The `radmc3d.inp` control file must have compatible settings

The safest approach is to use files that were generated together in a previous calculation.

### Output & Visualization

#### Visualization Parameters
- `--advanced_plots`: Create enhanced visualizations (default: False)
- `--figures_dir`: Directory to save figures (default: "figures")
- `--save_iter_plots`: Save plots for each iteration (default: False)
- `--species_index`: Dust species index to plot (0-3, default: 0)
- `--multi_species_handling`: Method for handling multiple species in plots (default: "specific")
- `--no_plots`: Disable plotting (default: False)
- `--output_dir`: Output directory name (default: "radmc3d_model")

The `--multi_species_handling` option controls how multiple dust species are displayed in plots:
- `specific`: Show only the species specified by `--species_index` (default)
- `average`: Average temperatures across all species
- `weighted_avg`: Weight temperatures by dust density
- `all`: Create a panel showing all species

## Convergence Analysis

The code checks for convergence between iterations by comparing the dust temperature solutions and calculating:

1. Maximum relative difference
2. Mean relative difference (used as the primary convergence criterion)
3. Median relative difference
4. 90th percentile difference
5. Percentage of cells changing temperature groups (for temperature-dependent calculations)

Convergence is reached when both:
- The mean relative temperature difference falls below the specified threshold
- The percentage of cells changing temperature groups falls below the specified threshold

For temperature-dependent iterations, convergence is assessed both in terms of temperature stability and the stability of cell assignments to temperature zones.

## Outputs

### Standard Outputs
- Temperature files: `dust_temperature.dat` and backups for each iteration
- Visualizations: Temperature slices, comparison plots, and distribution histograms
- Convergence data: Saved in `convergence_results.npz`
- SED data: If requested, saved as `spectrum.out`

### Temperature-Dependent Outputs
- Multi-species temperature files with backups for each iteration
- Multi-species density files showing the distribution across temperature zones
- Enhanced visualizations showing temperature distributions for different dust species
- Temperature zone maps showing the spatial distribution of different dust temperature ranges
- Summary plot showing temperature convergence metrics and cells changing temperature groups

### Visualization Options
The code provides several visualization options for multi-species models:
- Temperature maps for each individual dust species
- Averaged temperature maps across all species
- Density-weighted temperature maps
- Panel views showing all species side-by-side
- Temperature zone maps highlighting different temperature regions
- Pie charts and histograms showing dust temperature distribution

## References

- [RADMC-3D Manual](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/index.html) 