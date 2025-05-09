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

To set specific cutoff radii for components:

```bash
python main_radmc3d.py --Rmax_torus 4000 --Rmax_lobe 3500 --cone_angle 25
```

### Customizing Calculations

To customize the convergence parameters:

```bash
python main_radmc3d.py --nphotons_start 1e6 --nphotons_max 1e8 --threshold 0.005 --iterations 10
```

To disable temperature-dependent dust opacities (which are enabled by default):

```bash
python main_radmc3d.py --no_temp_dependent
```

To customize temperature-dependent dust opacity calculations:

```bash
python main_radmc3d.py --temp_ranges 50,150,250 --iterations 5 --threshold 0.05 --cells_change_threshold 0.5
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

To skip temperature calculation and use an existing temperature file:

```bash
python main_radmc3d.py --use_existing_temperature --temperature_file path/to/dust_temperature.dat --compute_sed --inclination 45 --phi 90
```

To only compute SED using existing dust temperature file (SED-only mode):

```bash
python main_radmc3d.py --sed_only --inclination 45 --phi 90
```

Note: With the parameter persistence feature added in v2.3.0, you no longer need to specify `--dust_size` and `--dust_material` in SED-only mode if you're using a temperature file generated by the same script. The parameters are automatically loaded from the `model_params.json` file.

To compute SED in a different directory than the default:

```bash
python main_radmc3d.py --sed_only --output_dir my_model_dir --inclination 45 --phi 90
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

To generate advanced plots from existing data without rerunning any calculations:

```bash
python main_radmc3d.py --plot_only
```

To generate plots with specific visualization options:

```bash
python main_radmc3d.py --plot_only --output_dir my_model_dir --figures_dir custom_figures --multi_species_handling weighted_avg --species_index 2
```

To read input files from one directory and save plots to another:

```bash
python main_radmc3d.py --plot_only --input_dir existing_model_dir --output_dir new_plots_dir --figures_dir temperature_plots
```

To compute SED from data in one directory and save results to another:

```bash
python main_radmc3d.py --sed_only --input_dir simulation_data --output_dir sed_results --inclination 45 --phi 90
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
- `--cone_angle`: Cone cavity opening angle in degrees (default: same as --oangle)
- `--width`: Width parameter (default: 0.005)
- `--Rmax_torus`: Maximum radius for the torus in AU (default: 0 = no limit)
- `--Rmax_lobe`: Maximum radius for the outflow lobes in AU (default: 0 = no limit)

### Physics Parameters

#### Dust Opacity Parameters
- `--no_temp_dependent`: Disable temperature-dependent dust opacities (enabled by default)
- `--temp_ranges`: Temperature range boundaries for dust opacity selection (default: "50,150,250")
- `--dust_file`: Dust opacity file for single-species models (default: "dustkapscatmat_E40R_300K_a0.3.inp")
- `--scattering_mode_max`: Scattering mode for RADMC-3D (default: 1)
  - 0: No scattering (absorption only)
  - 1: Isotropic scattering
  - 2-3-4-5: Anisotropic scattering (see RADMC-3D Documentation)
- `--dust_material`: Dust material composition to use in opacity calculations (default: "E40R")
- `--dust_size`: Characteristic dust grain size in microns (default: 0.3)

When using SED-only mode (`--sed_only`), the values for `--dust_material` and `--dust_size` will be automatically loaded from `model_params.json` if this file was generated in a previous temperature calculation. This ensures consistent parameters between temperature calculation and SED generation. You can override this behavior by explicitly providing these parameters on the command line.

#### Dust Composition
The following dust compositions are available through their prefix identifiers:

- **x series (magnesium silicates):**
  - `x035`: (0.65)MgO-(0.35)SiO2, density: 2.7 g/cm³
  - `x040`: (0.60)MgO-(0.40)SiO2, density: 2.7 g/cm³
  - `x050A`: (0.50)MgO-(0.50)SiO2 structure A, density: 2.7 g/cm³
  - `x050B`: (0.50)MgO-(0.50)SiO2 structure B, density: 2.7 g/cm³

- **E series (iron-magnesium silicates):**
  - `E10`: Mg(0.9)Fe(0.1)SiO3 with Fe³⁺, density: 2.8 g/cm³
  - `E10R`: Mg(0.9)Fe(0.1)SiO3 with Fe²⁺, density: 2.8 g/cm³
  - `E20`: Mg(0.8)Fe(0.2)SiO3 with Fe³⁺, density: 2.9 g/cm³
  - `E20R`: Mg(0.8)Fe(0.2)SiO3 with Fe²⁺, density: 2.9 g/cm³
  - `E30`: Mg(0.7)Fe(0.3)SiO3 with Fe³⁺, density: 3.0 g/cm³
  - `E30R`: Mg(0.7)Fe(0.3)SiO3 with Fe²⁺, density: 3.0 g/cm³
  - `E40`: Mg(0.6)Fe(0.4)SiO3 with Fe³⁺, density: 3.1 g/cm³
  - `E40R`: Mg(0.6)Fe(0.4)SiO3 with Fe²⁺, density: 3.1 g/cm³

The R suffix in the E series indicates the reduced form of iron (Fe²⁺ instead of Fe³⁺), which affects the optical properties of the dust.

### Calculation Control

#### Convergence Parameters
- `--nphotons_start`: Initial number of photon packages (default: 1e4)
- `--nphotons_max`: Maximum number of photon packages (default: 1e8)
- `--scale_factor`: Factor to increase photons by each iteration (default: 2.0)
- `--threshold`: Convergence threshold for temperature difference in all calculations (default: 2%)
- `--iterations`: Maximum number of iterations for all calculation types (default: 8)
- `--cells_change_threshold`: Threshold for cells changing temperature groups (default: 1%)
- `--density_weighted`: Use density-weighted convergence metrics (default: False)

#### SED Calculation Parameters
- `--no_compute_sed`: Disable SED computation after temperature calculation (enabled by default)
- `--sed_only`: Only compute SED using existing dust_temperature.dat file (skips temperature calculation)
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
- `--plot_only`: Only generate plots from existing data without running temperature calculations
- `--input_dir`: Directory containing existing data files to read (for use with --plot_only or --sed_only)
- `--output_dir`: Directory to save outputs (default: "radmc3d_model")
- `--species_index`: Dust species index to plot (0-3, default: 0)
- `--multi_species_handling`: Method for handling multiple species in plots (default: "specific")
- `--no_plots`: Disable plotting (default: False)

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

## Version History

### v1.0.0
- Initial release of the iterative dust temperature calculation framework
- Water fountain model implementation with torus and bipolar outflow components
- Basic temperature convergence checking and visualization

### v1.1.0
- Enhanced main_radmc3d.py with new command line arguments for SED-only and plot-only modes
- Added ability to compute SED from existing data without rerunning temperature calculations
- Added visualization-only mode to generate plots from existing data
- Improved error handling for file operations

### v1.2.0
- Added support for configurable scattering mode through the `--scattering_mode_max` parameter
- Removed `--setseed` parameter for simplicity
- Fixed parameter passing for scattering_mode_max to ensure consistent behavior
- Fixed bugs related to path misconnections between input and output directories
- Merged related input parameters for temperature calculations to improve usability
- Streamlined temperature convergence thresholds for consistent behavior across calculation modes

### v2.0.0 
- Main change: coupling with Optool
- Added support for custom dust properties through new `--dust_material` and `--dust_size` parameters
- Using all the Demyk's silicates

### v2.1.0
- Added separate parameter for cone cavity opening angle (`--cone_angle`) that can be equal to or larger than the lobe opening angle
- Added maximum radius parameters for torus (`--Rmax_torus`) and outflow lobes (`--Rmax_lobe`) to control their spatial extent
- Fixed AU scaling in plots to correctly represent distances in astronomical units
- Improved visualization annotations for model components
- Enhanced documentation for model geometry parameters
- Set minimum log density value to -24 in density plotting functions to provide more physically realistic visualization. This prevents showing extremely low (physically meaningless) density values in the colormap
- Changed the density within the cone region from 0.001 times the outside density to completely null (zero density)
- When running temperature calculations, dust parameters (material, size) are saved to `model_params.json`
- In SED-only mode, these parameters are automatically loaded if not explicitly specified, this eliminates the need to re-specify `--dust_size` and `--dust_material` when generating SEDs from existing temperature files
- Modified the density model to replace zero density values with very small non-zero values (Instead of setting density to exactly zero in regions like the cone cavity or beyond maximum radius, now using a very small fraction (1e-10) of the minimum non-zero density)

### v2.2.0
- Fixed critical bug in dust opacity file selection (thank you Theo!)
  - Previously, when running Optool with a specific temperature (e.g., 300K), the code would use the first .lnk file it found for the material regardless of temperature
  - Now properly searches for and uses temperature-specific .lnk files (e.g., E40R_300K.lnk)
  - Added warning when falling back to a .lnk file that doesn't match the requested temperature
- Fixed duplicate temperature indicators in output filenames (e.g., "E40R_10K_10K_a0.3.dat" → "E40R_10K_a0.3.dat")
- Improved logging to show which .lnk file is actually being used for calculations
- Removed confusing .dat extension from Optool output paths, making directory names cleaner (e.g., "E40R_10K_a0.3.dat" → "E40R_10K_a0.3")
