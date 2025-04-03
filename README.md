# RADMC-3D Iterative Dust Temperature Calculation for Water Fountains

Test for the branch

Test for the branch 2 

Test 3

Test 4
This project provides a Python framework for running RADMC-3D dust temperature calculations iteratively with convergence checking, specifically optimized for "water fountain" astrophysical objects. These are a special type of evolved star with high-velocity water maser jets, typically consisting of a central star, a torus-like structure, and outflow lobes.

## Overview

The code performs the following tasks:
1. Creates all necessary input files for RADMC-3D based on the water fountain physical model
2. Runs `radmc3d mctherm` with increasing photon numbers
3. Analyzes the dust temperature after each run to check for convergence
4. Visualizes and saves the results

The code supports parallelization to speed up calculations:
- **OpenMP parallelization**: Uses multiple CPU cores on a single machine
- **MPI parallelization**: Distributes the calculation across multiple machines or nodes (requires RADMC-3D to be compiled with MPI support)

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- RADMC-3D (must be installed and accessible in your PATH)

## File Structure

- `main_radmc3d.py`: Main script that runs the iterative calculation
- `create_input.py`: Functions to create all RADMC-3D input files, including the water fountain model
- `radmc3d_aux.py`: Auxiliary functions for running RADMC-3D and analyzing outputs
- `dustkapscatmat_E40R_300K_a0.3.inp`: Dust opacity file (provided)

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

## Usage

To run the iterative calculation with default water fountain settings:

```bash
python main_radmc3d.py
```

To customize the torus parameters:

```bash
python main_radmc3d.py --Mdtorus 0.01 --Rtorus 1200 --A 2 --B 4
```

To customize the outflow lobes:

```bash
python main_radmc3d.py --Mdlobe 0.001 --Rlobe 3000 --oangle 15
```

To customize the convergence parameters:

```bash
python main_radmc3d.py --nphotons_start 1e6 --nphotons_max 1e8 --threshold 0.005 --max_iterations 8
```

To use parallelization for faster calculations:

```bash
# Using OpenMP (shared memory) parallelization with 8 threads
python main_radmc3d.py --nthreads 8

# Using MPI (distributed memory) parallelization with 4 processes
# (requires RADMC-3D to be compiled with MPI support)
python main_radmc3d.py --mpi --np 4
```

## Command Line Arguments

### Model Geometry
- `--rin`: Inner radius in AU (default: 100)
- `--rout`: Outer radius in AU (default: 5000)
- `--nr`: Number of radial grid cells (default: 1000)
- `--ntheta`: Number of theta grid cells (default: 150)

### Star Parameters
- `--stellar_radius`: Radius in solar radii (default: 288)
- `--stellar_temp`: Temperature in K (default: 3000)

### Torus Parameters
- `--Mdtorus`: Torus mass in solar masses (default: 0.005)
- `--Rtorus`: Characteristic radius in AU (default: 1000)
- `--A`, `--B`, `--C`, `--D`, `--E`, `--F`: Shape parameters for the torus

### Outflow Lobe Parameters
- `--Mdlobe`: Outflow lobe mass in solar masses (default: 0.0005)
- `--Rlobe`: Characteristic radius in AU (default: 2500)
- `--oangle`: Opening angle in degrees (default: 18)
- `--width`: Width parameter (default: 0.005)

### Convergence Parameters
- `--nphotons_start`: Initial number of photon packages (default: 1e6)
- `--nphotons_max`: Maximum number of photon packages (default: 1e8)
- `--scale_factor`: Factor to increase photons by each iteration (default: 5.0)
- `--threshold`: Convergence threshold (default: 1%)
- `--max_iterations`: Maximum number of iterations (default: 10)

### Parallelization Options
- `--nthreads`: Number of OpenMP threads for parallelization (default: 4)
- `--mpi`: Use MPI parallelization (requires RADMC-3D compiled with MPI)
- `--np`: Number of MPI processes if using MPI (default: 4)

### Other Options
- `--no_plots`: Disable plotting
- `--density_weighted`: Use density-weighted convergence metrics
- `--setseed`: Random seed for reproducibility
- `--output_dir`: Output directory name (default: "radmc3d_model")
- `--dust_file`: Dust opacity file (default: "dustkapscatmat_E40R_300K_a0.3.inp")

## Convergence Analysis

The code checks for convergence between iterations by comparing the dust temperature solutions and calculating:

1. Maximum relative difference
2. Mean relative difference
3. Median relative difference
4. 90th percentile difference

Convergence is reached when the maximum relative difference falls below the specified threshold.

## Outputs

- Temperature files: `dust_temperature.dat` and backups for each iteration
- Visualizations: Temperature slices, comparison plots, and distribution histograms
- Convergence data: Saved in `convergence_results.npz`


## References

- [RADMC-3D Manual](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/index.html) 