#!/usr/bin/env python3
"""
RADMC-3D Iterative Dust Temperature Calculation for Water Fountain Models

This script performs iterative dust temperature calculations with RADMC-3D,
increasing the number of photon packages until convergence is reached.
Specifically optimized for water fountain astrophysical objects.

Command Line Arguments:
-----------------------
Model Parameters:
  --rin: Inner radius in AU (default: 100)
  --rout: Outer radius in AU (default: 5000)
  --nr: Number of radial grid cells (default: 1000)
  --ntheta: Number of theta grid cells (default: 150)

Physics Parameters:
  --dust_file: Dust opacity file (default: "dustkapscatmat_E40R_300K_a0.3.inp")
  --scattering_mode_max: Maximum scattering mode (0=no scattering, 1=isotropic, 2=anisotropic)
                       (default: 1)

Calculation Control:
  --nphotons_start: Initial number of photon packages (default: 1e4)
  --nphotons_max: Maximum number of photon packages (default: 1e8)
  --scale_factor: Factor to increase photons by each iteration (default: 2.0)
  --threshold: Convergence threshold (default: 0.02)
  --max_iterations: Maximum number of iterations (default: 8)
"""

import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import time
import shutil
import argparse
import subprocess
import sys
from create_input import setup_model, au, Rsun, Msun, verify_dust_temperature_file
from radmc3d_aux import (run_radmc3d, read_dust_temperature, check_convergence, 
                        plot_convergence_history, 
                        analyze_temperature_distribution, plot_advanced_temperature_density, 
                        read_dust_density, plot_temperature_dependent_summary,
                        plot_initial_dust_density, read_amr_grid, 
                        redistribute_density_by_temperature, write_temp_dependent_density)

# Add import to check if run_optool.py exists
import importlib.util

def backup_temperature_file(iteration):
    """
    Create a backup of the dust_temperature.dat file for the current iteration.
    
    Parameters:
    -----------
    iteration : int
        Current iteration number
    """
    # Create backup directory if it doesn't exist
    backup_dir = "temp_backups"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Copy the temperature file with iteration number
    if os.path.exists("dust_temperature.dat"):
        shutil.copy("dust_temperature.dat", 
                   os.path.join(backup_dir, "dust_temperature_iter{}.dat".format(iteration)))
        print("Backed up temperature file for iteration {}".format(iteration))
    else:
        print("Warning: dust_temperature.dat not found, could not create backup")

def iterative_mctherm(nphotons_start=1e6, nphotons_max=1e8, 
                      scale_factor=5.0, convergence_threshold=0.01,
                      max_iterations=10, plot_slices=True, plot_progress=True,
                      density_weighted=False, density_array=None,
                      return_temp_history=False,
                      multi_species_handling='specific', species_index=0,
                      scattering_mode_max=1):
    """
    Run RADMC-3D Monte Carlo dust temperature calculation iteratively
    with increasing photon numbers until convergence.
    
    Parameters:
    -----------
    nphotons_start : float
        Initial number of photon packages
    nphotons_max : float
        Maximum number of photon packages
    scale_factor : float
        Factor by which to increase nphotons each iteration
    convergence_threshold : float
        Maximum relative difference for convergence
    max_iterations : int
        Maximum number of iterations
    plot_slices : bool
        Whether to plot temperature slices after each iteration
    plot_progress : bool
        Whether to plot convergence progress after completion
    density_weighted : bool
        Whether to use density-weighted convergence metrics
    density_array : ndarray
        Density array for weighted convergence checking
    return_temp_history : bool
        Whether to return full temperature history arrays
    multi_species_handling : str
        How to handle multiple species: 'specific', 'average', 'weighted_avg', or 'all'
    species_index : int
        Dust species index to plot when using the 'specific' handling
    scattering_mode_max : int
        Maximum scattering mode (0=no scattering, 1=isotropic, 2=anisotropic)
    
    Returns:
    --------
    bool
        True if converged, False otherwise
    dict
        Dictionary containing convergence results and history
    """
    # Initialize variables to store results
    nphotons_history = []
    metrics_history = []
    temp_history = []
    previous_temp = None
    converged = False
    
    # Start timing
    start_time = time.time()
    
    # Create directory for plots if needed
    if plot_slices and not os.path.exists('temp_plots'):
        os.mkdir('temp_plots')
    
    # Run iterations
    for iteration in range(1, max_iterations + 1):
        # Calculate number of photons for this iteration
        nphotons = int(min(nphotons_start * (scale_factor**(iteration-1)), nphotons_max))
        nphotons_history.append(nphotons)
        
        print("\n===== Iteration {}/{} =====".format(iteration, max_iterations))
        print("Running RADMC-3D with {} photon packages...".format(nphotons))
        
        # Update RADMC-3D control file with new photon number
        with open('radmc3d.inp', 'w') as f:
            f.write('nphot = {}\n'.format(nphotons))
            f.write('modified_random_walk = 1\n')
            f.write('scattering_mode_max = {}\n'.format(scattering_mode_max))
            f.write('istar_sphere = 1\n')
        
        # Run RADMC-3D Monte Carlo thermal calculation
        run_radmc3d(cmd='mctherm')
        
        # Read temperature data
        temp_data, grid_info = read_dust_temperature()
        
        # Backup the temperature file
        backup_temperature_file(iteration)
        
        # Plot temperature slice for this iteration
        if plot_slices:
            # Create a directory for temperature plots if it doesn't exist
            os.makedirs("temp_plots", exist_ok=True)
            
            # Use plot_advanced_temperature_density with appropriate multi-species handling
            try:
                # Get density data if available (for weighted averages)
                try:
                    density_data = read_dust_density(grid_info=grid_info)[0] if density_weighted else None
                except Exception as e:
                    print("Warning: Could not read density data for plotting: {}".format(e))
                    density_data = None
                
                # Create advanced temperature plots for this iteration
                plot_advanced_temperature_density(
                    grid_info=grid_info,
                    temp_data=temp_data,
                    density_data=density_data,
                    output_dir="temp_plots",
                    species=species_index,
                    multi_species_handling=multi_species_handling,
                    save_fig=True,
                    show_fig=False,
                    iteration=iteration
                )
                
                # Create a dedicated temperature zone map for this iteration
                plot_advanced_temperature_density(
                    grid_info=grid_info,
                    temp_data=temp_data,
                    density_data=density_data,
                    output_dir="temp_plots",
                    species=species_index,
                    multi_species_handling=multi_species_handling,
                    save_fig=True,
                    show_fig=False,
                    iteration=iteration,
                    create_zone_map=True  # Create a dedicated temperature zone map
                )
            except Exception as e:
                print("Warning: Error creating temperature plots: {}".format(e))
        
        # Add to temperature history
        if return_temp_history:
            temp_history.append(temp_data.copy())
        
        # Check convergence if we have a previous solution
        if previous_temp is not None:
            # Calculate convergence metrics
            metrics = check_convergence(temp_data, previous_temp, threshold=convergence_threshold,
                                       density_weights=density_array if density_weighted else None,
                                       return_details=True)
            
            metrics_history.append(metrics)
            
            # Print convergence metrics
            print("Maximum difference: {:.4f}%".format(metrics['max_diff']*100))
            print("Mean difference: {:.4f}%".format(metrics['mean_diff']*100))
            print("Median difference: {:.4f}%".format(metrics['median_diff']*100))
            print("90th percentile difference: {:.4f}%".format(metrics['p90_diff']*100))
            
            # Check if converged
            if metrics['mean_diff'] < convergence_threshold:
                converged = True
                print("\nConvergence reached after {} iterations!".format(iteration))
                print("Final photon count: {}".format(nphotons))
                break
        
        # Store current temperature for next iteration
        previous_temp = temp_data.copy()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print("\nTotal calculation time: {:.1f} seconds".format(elapsed_time))
    
    # Plot convergence history
    if plot_progress and len(metrics_history) > 0:
        plot_convergence_history(metrics_history, nphotons_history, 
                                save_fig=True, filename="temp_plots/convergence_history.png",
                                threshold=convergence_threshold, show_fig=False)
        
        # Plot temperature distribution
        if return_temp_history and len(temp_history) >= 2:
            analyze_temperature_distribution(temp_data, grid_info, 
                                          save_fig=True, filename="temp_plots/temperature_distribution.png",
                                          show_fig=False)
    
    # Return results
    results = {
        'converged': converged,
        'iterations': len(nphotons_history),
        'nphotons_history': nphotons_history,
        'metrics_history': metrics_history if len(metrics_history) > 0 else None,
        'elapsed_time': elapsed_time
    }
    
    if return_temp_history:
        results['temp_history'] = temp_history
    
    return converged, results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run RADMC-3D iterative dust temperature calculation for water fountain models')
    
    # Model parameters
    parser.add_argument('--rin', type=float, default=100, help='Inner radius in AU')
    parser.add_argument('--rout', type=float, default=5000, help='Outer radius in AU')
    parser.add_argument('--nr', type=int, default=1000, help='Number of radial grid cells')
    parser.add_argument('--ntheta', type=int, default=150, help='Number of theta grid cells')
    
    # Star parameters
    parser.add_argument('--stellar_radius', type=float, default=288, help='Stellar radius in solar radii')
    parser.add_argument('--stellar_temp', type=float, default=3000, help='Stellar temperature in K')
    
    # Torus parameters
    parser.add_argument('--Mdtorus', type=float, default=0.005, help='Torus mass in solar masses')
    parser.add_argument('--Rtorus', type=float, default=1000, help='Torus characteristic radius in AU')
    parser.add_argument('--A', type=float, default=1, help='Torus parameter A')
    parser.add_argument('--B', type=float, default=3, help='Torus parameter B')
    parser.add_argument('--C', type=float, default=0, help='Torus parameter C')
    parser.add_argument('--D', type=float, default=10, help='Torus parameter D')
    parser.add_argument('--E', type=float, default=3, help='Torus parameter E')
    parser.add_argument('--F', type=float, default=2, help='Torus parameter F')
    
    # Outflow lobe parameters
    parser.add_argument('--Mdlobe', type=float, default=0.0005, help='Outflow lobe mass in solar masses')
    parser.add_argument('--Rlobe', type=float, default=2500, help='Lobe characteristic radius in AU')
    parser.add_argument('--oangle', type=float, default=18, help='Opening angle in degrees')
    parser.add_argument('--width', type=float, default=0.005, help='Lobe width parameter')
    
    # Physics and Convergence Parameters
    physics = parser.add_argument_group('Physics and Convergence Parameters')
    physics.add_argument('--nphotons_start', type=float, default=1e4, help='Initial number of photon packages')
    physics.add_argument('--nphotons_max', type=float, default=1e8, help='Maximum number of photon packages')
    physics.add_argument('--scale_factor', type=float, default=2.0, help='Factor to increase photons by each iteration')
    physics.add_argument('--threshold', type=float, default=0.02, help='Convergence threshold for temperature difference (default: 2%%)')
    physics.add_argument('--iterations', type=int, default=8, help='Maximum number of iterations for all calculation types')
    physics.add_argument('--density_weighted', action='store_true', help='Use density-weighted convergence metrics')
    physics.add_argument('--cells_change_threshold', type=float, default=1.0, help='Threshold for cells changing temperature groups (default: 1%%)')
    physics.add_argument('--temp_ranges', type=str, default="50,150,250", help='Temperature range boundaries for dust opacity selection (K)')
    physics.add_argument('--dust_file', default="dustkapscatmat_E40R_300K_a0.3.inp", help='Dust opacity file for single-species models')
    physics.add_argument('--scattering_mode_max', type=int, default=1, help='Maximum scattering mode (0=no scattering, 1=isotropic, 2=anisotropic)')
    physics.add_argument('--dust_material', type=str, default="E40R", help='Dust material composition (default: E40R)')
    physics.add_argument('--dust_size', type=float, default=0.3, help='Characteristic dust grain size in microns (default: 0.3)')
    
    # Other options
    parser.add_argument('--no_plots', action='store_true', help='Disable plotting')
    parser.add_argument('--output_dir', type=str, default='radmc3d_model', help='Output directory name')
    parser.add_argument('--input_dir', type=str, default=None, 
                      help='Input directory containing existing data files (for --plot_only or --sed_only)')
    
    # Add advanced visualization option
    parser.add_argument('--advanced_plots', action='store_true', help='Create advanced temperature and density visualizations')
    parser.add_argument('--figures_dir', type=str, default='figures', help='Directory to save figures')
    parser.add_argument('--save_iter_plots', action='store_true', help='Save advanced temperature plots for each iteration')
    parser.add_argument('--plot_only', action='store_true', 
                       help='Only generate plots from existing data (no temperature calculation)')
    
    # Add temperature-dependent dust opacity options (now default)
    parser.add_argument('--no_temp_dependent', action='store_true', help='Disable temperature-dependent dust opacities')
    
    # Add multi-species visualization options
    parser.add_argument('--species_index', type=int, default=0, help='Dust species index to plot (0-3)')
    parser.add_argument('--multi_species_handling', type=str, default='specific', 
                      choices=['specific', 'average', 'weighted_avg', 'all'],
                      help='How to handle multiple species in plots: specific=show only species_index, '
                           'average=average temp across species, weighted_avg=weight by density, '
                           'all=show all species as separate panels')
    
    # Add SED calculation options
    parser.add_argument('--no_compute_sed', action='store_true', help='Disable SED computation after temperature calculation')
    parser.add_argument('--sed_only', action='store_true', help='Only compute SED using existing dust_temperature.dat file')
    parser.add_argument('--inclination', type=float, default=0, help='Observer inclination angle in degrees')
    parser.add_argument('--phi', type=float, default=0, help='Observer azimuthal angle in degrees')
    parser.add_argument('--no_sloppy', action='store_true', help='Disable sloppy integration in SED calculation')
    
    # Add existing temperature file option
    parser.add_argument('--use_existing_temperature', action='store_true', help='Use existing dust_temperature.dat file (skip calculation)')
    parser.add_argument('--temperature_file', type=str, default='dust_temperature.dat', 
                      help='Path to existing dust temperature file (when using --use_existing_temperature)')
    
    # Add option to force regeneration of dustopac.inp
    parser.add_argument('--regenerate-dustopac', action='store_true', 
                      help='Force regeneration of dustopac.inp to match specified dust_material and dust_size')
    
    args = parser.parse_args()
    # Set temp_dependent to True by default (use --no_temp_dependent to disable)
    args.temp_dependent = not args.no_temp_dependent
    # Set compute_sed to True by default (use --no_compute_sed to disable)
    args.compute_sed = not args.no_compute_sed
    # If --sed_only is specified, make sure we're using existing temperature file and forcing SED computation
    if args.sed_only:
        args.use_existing_temperature = True
        args.compute_sed = True
        args.no_plots = True  # Disable plotting for SED-only mode
    # If --plot_only is specified, force use of existing temperature file and disable SED computation
    if args.plot_only:
        args.use_existing_temperature = True
        args.compute_sed = False
        args.advanced_plots = True  # Force advanced plots
        # Check for mutually exclusive options
        if args.sed_only:
            print("ERROR: --plot_only and --sed_only cannot be used together")
            sys.exit(1)
    return args

def main():
    """Main function"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Store the original directory (where the script is being run from)
    original_dir = os.path.abspath(os.getcwd())
    
    # Check if dust opacity files need to be generated
    if args.temp_dependent and not args.sed_only and not args.plot_only and not args.use_existing_temperature:
        # Define the required dust opacity files
        required_files = [
            f'dustkapscatmat_{args.dust_material}_10K_a{args.dust_size}.inp',
            f'dustkapscatmat_{args.dust_material}_100K_a{args.dust_size}.inp',
            f'dustkapscatmat_{args.dust_material}_200K_a{args.dust_size}.inp',
            f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp'
        ]
        
        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
            
        # Check if nk_optool directory exists in original directory or output_dir
        original_nk_dir = "nk_optool"
        output_nk_dir = os.path.join(args.output_dir, "nk_optool")
        
        # If nk_optool doesn't exist in output_dir but exists in original directory, copy it
        if not os.path.exists(output_nk_dir) and os.path.exists(original_nk_dir):
            print(f"Copying nk_optool directory to {args.output_dir}")
            shutil.copytree(original_nk_dir, output_nk_dir)
        
        # Check if all required files exist in the output_dir
        missing_files = []
        for file in required_files:
            output_file_path = os.path.join(args.output_dir, file)
            if not os.path.exists(output_file_path):
                missing_files.append(file)
        
        # If any files are missing, check if run_optool.py exists and run it
        if missing_files:
            print(f"\n=== Dust opacity files missing for {args.dust_material} with size {args.dust_size}μm ===")
            print("Will try to generate them using run_optool.py...")
            
            # Check if run_optool.py exists in the current directory
            if os.path.exists('run_optool.py'):
                try:
                    # Run the optool script to generate dust opacity files
                    cmd = [
                        sys.executable, 'run_optool.py', 
                        '--material', args.dust_material,
                        '--grain-size', str(args.dust_size),
                        '--temperatures', '10,100,200,300',
                        '--output-dir', args.output_dir
                    ]
                    
                    # If nk_optool directory exists in output_dir, add it as a parameter
                    if os.path.exists(output_nk_dir):
                        cmd.extend(['--nk-dir', output_nk_dir])
                    
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    print("Successfully generated dust opacity files.")
                    
                    # Verify the files were created in the output directory
                    still_missing = []
                    for file in missing_files:
                        output_file_path = os.path.join(args.output_dir, file)
                        if not os.path.exists(output_file_path):
                            still_missing.append(file)
                    
                    if still_missing:
                        print(f"Warning: The following files are still missing after running run_optool.py:")
                        for file in still_missing:
                            print(f"  - {file}")
                        print("Will attempt to continue with available files.")
                    else:
                        print("All required dust opacity files were successfully created.")
                except Exception as e:
                    print(f"Error running run_optool.py: {e}")
                    print("ERROR: Could not generate required dust opacity files.")
                    print(f"Make sure all files exist in the output directory ({args.output_dir}):")
                    print(f"- dustkapscatmat_{args.dust_material}_10K_a{args.dust_size}.inp")
                    print(f"- dustkapscatmat_{args.dust_material}_100K_a{args.dust_size}.inp")
                    print(f"- dustkapscatmat_{args.dust_material}_200K_a{args.dust_size}.inp")
                    print(f"- dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp")
                    return 1
            else:
                print("Warning: run_optool.py script not found in the current directory.")
                print("Please make sure it exists or generate the dust opacity files manually.")
    
    # Convert input_dir to absolute path if provided
    if args.input_dir is not None:
        args.input_dir = os.path.abspath(args.input_dir)
    
    # Handle input_dir vs output_dir for plot_only and sed_only modes
    if args.input_dir is not None and (args.plot_only or args.sed_only):
        input_dir = args.input_dir
        if not os.path.exists(input_dir):
            print("ERROR: Input directory '{}' does not exist".format(input_dir))
            return 1
        
        # Only create output directory if:
        # 1. User specified a non-default output dir, or 
        # 2. We need to save results somewhere different than the input dir
        if args.output_dir != 'radmc3d_model' or args.input_dir != args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
                print(f"Created output directory: {args.output_dir}")
    else:
        # For normal operations, just use output_dir for everything
        input_dir = args.output_dir
        output_dir = args.output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
    
    # Change to output directory for all operations
    print(f"\n=== Changing to output directory: {args.output_dir} ===")
    os.chdir(args.output_dir)
    
    # After changing directory, output_dir should be "." (current directory)
    # This prevents using nested paths when we use output_dir later
    output_dir = "."
    
    # Import required functions right after changing directory to avoid scope issues 
    from create_input import setup_model, au, Rsun, Msun, verify_dust_temperature_file
    from radmc3d_aux import (run_radmc3d, read_dust_temperature, check_convergence, 
                           plot_convergence_history, analyze_temperature_distribution, 
                           read_amr_grid, read_dust_density, plot_advanced_temperature_density, 
                           plot_initial_dust_density, plot_temperature_dependent_summary,
                           redistribute_density_by_temperature, write_temp_dependent_density)
    
    # Handle the --regenerate-dustopac flag if specified
    if args.regenerate_dustopac:
        print("\n=== Regenerating dustopac.inp file with specified material and size ===")
        
        # Import functions needed for dustopac regeneration
        from create_input import create_dustopac, verify_dust_opacity_files, copy_dust_opacity
        
        # Check for the required dust opacity files
        if args.temp_dependent:
            # Temperature-dependent case
            files_exist, missing_files = verify_dust_opacity_files(
                dust_material=args.dust_material,
                dust_size=args.dust_size,
                temp_dependent=True
            )
            
            if not files_exist:
                print(f"WARNING: Missing dust opacity files for material {args.dust_material} with size {args.dust_size}μm")
                print("Missing files:")
                for file in missing_files:
                    print(f"  - {file}")
                print("\nAttempting to generate missing files with run_optool.py...")
                
                if os.path.exists('run_optool.py') or os.path.exists(os.path.join(original_dir, 'run_optool.py')):
                    try:
                        # Use either local run_optool.py or the one in the original directory
                        optool_script = 'run_optool.py' if os.path.exists('run_optool.py') else os.path.join(original_dir, 'run_optool.py')
                        
                        cmd = [
                            sys.executable, optool_script,
                            '--material', args.dust_material,
                            '--grain-size', str(args.dust_size),
                            '--temperatures', '10,100,200,300',
                            '--output-dir', '.'
                        ]
                        print(f"Running: {' '.join(cmd)}")
                        subprocess.run(cmd, check=True)
                        
                        # Check if files were created
                        files_exist, still_missing = verify_dust_opacity_files(
                            dust_material=args.dust_material,
                            dust_size=args.dust_size,
                            temp_dependent=True
                        )
                        
                        if not files_exist:
                            print(f"WARNING: Still missing some files after running run_optool.py:")
                            for file in still_missing:
                                print(f"  - {file}")
                            print("Will continue with available files.")
                        else:
                            print("Successfully generated all required dust opacity files.")
                    except Exception as e:
                        print(f"Error running run_optool.py: {e}")
                        print("Will continue with available files.")
                else:
                    print("run_optool.py not found. Cannot generate missing files.")
                    print("Will continue with available files.")
            
            # Create the temperature-dependent dustopac.inp file
            create_dustopac(temp_dependent=True, dust_material=args.dust_material, dust_size=args.dust_size)
            print(f"Created dustopac.inp with temperature-dependent opacities for material {args.dust_material} with size {args.dust_size}μm")
            
        else:
            # Single-species case
            main_file = f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp'
            
            # Check if the main opacity file exists
            if not os.path.exists(main_file):
                print(f"WARNING: Missing dust opacity file: {main_file}")
                print("\nAttempting to generate it with run_optool.py...")
                
                if os.path.exists('run_optool.py') or os.path.exists(os.path.join(original_dir, 'run_optool.py')):
                    try:
                        # Use either local run_optool.py or the one in the original directory
                        optool_script = 'run_optool.py' if os.path.exists('run_optool.py') else os.path.join(original_dir, 'run_optool.py')
                        
                        cmd = [
                            sys.executable, optool_script,
                            '--material', args.dust_material,
                            '--grain-size', str(args.dust_size),
                            '--no-temp-dependent',
                            '--output-dir', '.'
                        ]
                        print(f"Running: {' '.join(cmd)}")
                        subprocess.run(cmd, check=True)
                        
                        if os.path.exists(main_file):
                            print(f"Successfully generated {main_file}")
                        else:
                            print(f"WARNING: Could not generate {main_file}")
                            print("Will continue with available files.")
                    except Exception as e:
                        print(f"Error running run_optool.py: {e}")
                        print("Will continue with available files.")
                else:
                    print("run_optool.py not found. Cannot generate missing files.")
                    print("Will continue with available files.")
            
            # If the main file exists, create the standard dust_kapscatmat file and dustopac.inp
            if os.path.exists(main_file):
                copy_dust_opacity(dust_file=main_file, new_name='dust_kapscatmat_1.inp')
                create_dustopac(nspecies=1, scattering_mode=args.scattering_mode_max, 
                              water_fountain=False, dust_material=args.dust_material, dust_size=args.dust_size)
                print(f"Created dustopac.inp with single-species opacity for material {args.dust_material} with size {args.dust_size}μm")
            else:
                print(f"WARNING: Required dust opacity file {main_file} not found.")
                print("Cannot create proper dustopac.inp without the required file.")
        
        if args.sed_only:
            print("Continuing with SED calculation using the regenerated dustopac.inp file...")
        elif args.plot_only:
            print("Continuing with plot generation...")
        elif args.use_existing_temperature:
            print("Continuing with existing temperature file and regenerated dustopac.inp file...")
        else:
            print("Continuing with regular temperature calculation...")
    
    # Special message for plot-only mode
    if args.plot_only:
        print("\n=== PLOT-ONLY MODE ===")
        
        if args.input_dir:
            print("Reading data from: {}".format(args.input_dir))
            print("Saving plots to: {}".format(os.path.join(output_dir, args.figures_dir)))
            
            # Change to input directory to read files
            os.chdir(args.input_dir)
        else:
            print("Reading data and saving plots in: {}".format(output_dir))
            # Change to output directory
            os.chdir(output_dir)
        
        # Verify required files exist
        required_files = ['dust_temperature.dat', 'amr_grid.inp']
        optional_files = ['dust_density.inp']
        
        missing_required = []
        for file in required_files:
            if not os.path.exists(file):
                missing_required.append(file)
        
        if missing_required:
            print("\nERROR: The following required files for plotting are missing:")
            for file in missing_required:
                print("  - {}".format(file))
            print("Make sure they exist in: {}".format(os.path.abspath('.')))
            return 1
        
        # Check for optional files
        missing_optional = []
        for file in optional_files:
            if not os.path.exists(file):
                missing_optional.append(file)
        
        if missing_optional:
            print("\nWARNING: The following optional files for plotting are missing:")
            for file in missing_optional:
                print("  - {}".format(file))
            print("Some plots may not include density information.")
        
        print("\n=== Loading data for plots ===")
        try:
            # Read the grid information using the local import
            grid_info = read_amr_grid()
            print("Successfully loaded grid information")
            
            # Read the temperature data
            temp_data, _ = read_dust_temperature()
            print("Successfully loaded temperature data")
            
            # Try to read density data if available
            try:
                density_data, _, _, _ = read_dust_density(grid_info=grid_info)
                print("Successfully loaded density data")
            except Exception as e:
                print("Could not read density data: {}".format(e))
                density_data = None
            
            # If using separate input/output dirs, change to output dir for saving
            if args.input_dir:
                # Create parent output directory if needed
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Change to output directory
                os.chdir(output_dir)
            
            # Create figures directory if it doesn't exist
            figures_dir = args.figures_dir
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)
            
            print("\n=== Generating advanced visualizations ===")
            # Create high-quality visualization of the density structure (if available)
            if density_data is not None:
                print("Creating dust density visualization...")
                try:
                    # Import the function directly within this scope
                    from radmc3d_aux import plot_initial_dust_density as plot_density_func
                    
                    density_stats = plot_density_func(
                        grid_info=grid_info,
                        density_data=density_data,
                        output_dir=figures_dir,
                        save_fig=True,
                        show_fig=False,
                        add_annotations=True
                    )
                    
                    if density_stats:
                        print("\n=== Dust Density Statistics ===")
                        print("Minimum density: {:.2e} g/cm³".format(density_stats['min_density']))
                        print("Maximum density: {:.2e} g/cm³".format(density_stats['max_density']))
                        print("Mean density: {:.2e} g/cm³".format(density_stats['mean_density']))
                        print(f"Median density: {density_stats['median_density']:.2e} g/cm³")
                except Exception as e:
                    print("Error creating density plot: {}".format(e))
            
            # Create temperature visualizations
            print("Creating temperature visualizations...")
            # Import function within scope to avoid reference issues
            from radmc3d_aux import plot_advanced_temperature_density as plot_temp_func
            
            # First, try with the user-specified multi-species handling
            stats = plot_temp_func(
                grid_info=grid_info,
                temp_data=temp_data,
                density_data=density_data,
                output_dir=figures_dir,
                species=args.species_index,
                multi_species_handling=args.multi_species_handling,
                save_fig=True,
                show_fig=False
            )
            
            # If multiple species are detected, create a panel showing all species
            if temp_data.shape[-1] > 1:
                print("\nMultiple dust species detected, creating all-species panel...")
                plot_temp_func(
                    grid_info=grid_info,
                    temp_data=temp_data,
                    density_data=density_data,
                    output_dir=figures_dir,
                    species=args.species_index,
                    multi_species_handling="all",  # Show all species
                    save_fig=True,
                    show_fig=False
                )
            
            # Create temperature zone map
            print("Creating temperature zone maps...")
            plot_temp_func(
                grid_info=grid_info,
                temp_data=temp_data,
                density_data=density_data,
                output_dir=figures_dir,
                species=args.species_index,
                multi_species_handling=args.multi_species_handling,
                save_fig=True,
                show_fig=False,
                create_zone_map=True
            )
            
            # Print temperature statistics
            if stats:
                print("\n=== Temperature Statistics ===")
                print("Minimum temperature: {:.2f} K".format(stats['min_temperature']))
                print("Maximum temperature: {:.2f} K".format(stats['max_temperature']))
                print("Mean temperature: {:.2f} K".format(stats['mean_temperature']))
                print(f"Median temperature: {stats['median_temperature']:.2f} K")
                
                # Print cells in different temperature ranges
                total_cells = stats['total_cells']
                print("Temperature distribution for {}species {}".format('all' if args.multi_species_handling in ['average', 'weighted_avg'] else 'dust species {}'.format(args.species_index), ":"))
                print("Below 50K: {} cells ({:.2f}%)".format(stats['below_50K'], stats['below_50K']/total_cells*100))
                print("50K-150K: {} cells ({:.2f}%)".format(stats['between_50_150K'], stats['between_50_150K']/total_cells*100))
                print("150K-250K: {} cells ({:.2f}%)".format(stats['between_150_250K'], stats['between_150_250K']/total_cells*100))
                print("Above 250K: {} cells ({:.2f}%)".format(stats['above_250K'], stats['above_250K']/total_cells*100))
            
            print("All plots saved to {}".format(os.path.join(output_dir, figures_dir)))
            print("\nDone.")
            return 0
            
        except Exception as e:
            print("Error generating plots: {}".format(e))
            return 1
    
    # Special message for SED-only mode
    if args.sed_only:
        print("\n=== SED-ONLY MODE ===")
        
        # Always make sure the output directory exists before trying to change to it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        if args.input_dir:
            print("Reading data from: {}".format(args.input_dir))
            print("Saving SED to: {}".format(os.path.abspath(output_dir)))
            
            # Change to input directory to read files
            os.chdir(args.input_dir)
            
            # Need to copy temperature file to output directory for radmc3d
            if not os.path.samefile(args.input_dir, os.path.abspath(output_dir)):
                if os.path.exists('dust_temperature.dat'):
                    # Copy the temperature file
                    shutil.copy('dust_temperature.dat', os.path.join(os.path.abspath(output_dir), 'dust_temperature.dat'))
                    
                    # Copy other required files too
                    for req_file in ['amr_grid.inp', 'wavelength_micron.inp', 
                                    'dust_density.inp', 'stars.inp', 'radmc3d.inp']:
                        if os.path.exists(req_file):
                            shutil.copy(req_file, os.path.join(os.path.abspath(output_dir), req_file))
                
                # Change to output directory for calculation
                os.chdir(os.path.abspath(output_dir))
        else:
            # We're already in the output directory (current directory)
            print(f"Computing SED in current directory: {os.path.abspath('.')}")
            
        print("Observer position: inclination = {}°, phi = {}°".format(args.inclination, args.phi))
        
        # Verify temperature file exists
        if not os.path.exists('dust_temperature.dat'):
            print("\nERROR: dust_temperature.dat not found in the current directory.")
            print("Make sure it exists in: {}".format(os.path.abspath('.')))
            print("Alternatively, specify a temperature file with --temperature_file.")
            return 1
        
        # Check the number of dust species in the dust_temperature.dat file
        n_species_in_temp_file = 1  # Default assumption
        try:
            with open('dust_temperature.dat', 'r') as f:
                # Read format (should be 1)
                iformat = int(f.readline().strip())
                # Read number of cells
                ncells = int(f.readline().strip())
                # Read number of species
                n_species_in_temp_file = int(f.readline().strip())
            
            print(f"\nDetected {n_species_in_temp_file} dust species in the temperature file.")
        except Exception as e:
            print(f"Warning: Could not read number of species from dust_temperature.dat: {e}")
            print("Assuming 1 dust species in temperature file.")
        
        # Create or update dustopac.inp file with the specified dust material and size
        # Use the same number of species as in the dust_temperature.dat file
        from create_input import create_dustopac, copy_dust_opacity
        
        # Check if the required dustkapscatmat files exist for the specified material and size
        required_files = []
        if args.temp_dependent and n_species_in_temp_file >= 4:
            # For temperature-dependent mode with 4 species
            required_files = [
                f'dustkapscatmat_{args.dust_material}_10K_a{args.dust_size}.inp',
                f'dustkapscatmat_{args.dust_material}_100K_a{args.dust_size}.inp',
                f'dustkapscatmat_{args.dust_material}_200K_a{args.dust_size}.inp',
                f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp'
            ]
            print(f"Using temperature-dependent dust opacities for material {args.dust_material} with size {args.dust_size}μm")
        else:
            # For single-species mode (or if temperature file has fewer than 4 species)
            temp_dependent = False
            required_files = [f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp']
            print(f"Using single dust opacity file for material {args.dust_material} with size {args.dust_size}μm")
            
            # If temperature-dependent was requested but not possible
            if args.temp_dependent and n_species_in_temp_file < 4:
                print(f"\nWARNING: Temperature-dependent opacity requested but dust_temperature.dat has only {n_species_in_temp_file} species.")
                print("Falling back to single-species mode for SED calculation.")
        
        # Check for missing dust opacity files
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print("\nWARNING: The following dust opacity files are missing:")
            for file in missing_files:
                print(f"  - {file}")
            print("\nAttempting to find or generate these files...")
            
            # Try to copy from other locations or generate with run_optool.py
            if os.path.exists('run_optool.py'):
                try:
                    # Run optool to generate the required dust opacity files
                    temperatures_arg = '10,100,200,300' if args.temp_dependent and n_species_in_temp_file >= 4 else '300'
                    no_temp_dependent_arg = [] if args.temp_dependent and n_species_in_temp_file >= 4 else ['--no-temp-dependent']
                    
                    cmd = [
                        sys.executable, 'run_optool.py', 
                        '--material', args.dust_material,
                        '--grain-size', str(args.dust_size),
                        '--temperatures', temperatures_arg,
                        '--output-dir', '.'
                    ]
                    if no_temp_dependent_arg:
                        cmd.extend(no_temp_dependent_arg)
                    
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    
                    # Check if files were created
                    still_missing = []
                    for file in missing_files:
                        if not os.path.exists(file):
                            still_missing.append(file)
                    
                    if still_missing:
                        print("\nWARNING: The following files are still missing:")
                        for file in still_missing:
                            print(f"  - {file}")
                        print("SED calculation may fail!")
                    else:
                        print("Successfully generated all required dust opacity files.")
                except Exception as e:
                    print(f"Error generating dust opacity files: {e}")
                    print("SED calculation may fail!")
            else:
                print("run_optool.py not found. Cannot generate missing files.")
                print("SED calculation may fail!")
        
        # Create dustopac.inp with the correct number of species
        if args.temp_dependent and n_species_in_temp_file >= 4:
            create_dustopac(temp_dependent=True, dust_material=args.dust_material, dust_size=args.dust_size)
            print(f"Created dustopac.inp with temperature-dependent dust species, scattering mode {args.scattering_mode_max}")
        else:
            # For a single-species mode (or if temperature file has < 4 species)
            # First, make sure we have the dust opacity file in the correct format
            main_opacity_file = f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp'
            
            if os.path.exists(main_opacity_file):
                # Copy with the standard name if needed
                if n_species_in_temp_file == 1:
                    copy_dust_opacity(dust_file=main_opacity_file, new_name='dust_kapscatmat_1.inp')
                    # Create dustopac.inp for a single species
                    create_dustopac(nspecies=1, scattering_mode=args.scattering_mode_max, 
                                   water_fountain=False, dust_material=args.dust_material, dust_size=args.dust_size)
                    print(f"Created dustopac.inp with single dust species, scattering mode {args.scattering_mode_max}")
                else:
                    # Handle multiple species but not temperature-dependent case
                    for i in range(n_species_in_temp_file):
                        copy_dust_opacity(dust_file=main_opacity_file, new_name=f'dust_kapscatmat_{i+1}.inp')
                    
                    # Create dustopac with the exact number of species from temperature file
                    create_dustopac(nspecies=n_species_in_temp_file, scattering_mode=args.scattering_mode_max, 
                                   water_fountain=False, dust_material=args.dust_material, dust_size=args.dust_size)
                    print(f"Created dustopac.inp with {n_species_in_temp_file} identical dust species, scattering mode {args.scattering_mode_max}")
            else:
                print(f"WARNING: Required dust opacity file {main_opacity_file} not found.")
                print("SED calculation may fail!")
        
        # Display the content of dustopac.inp
        if os.path.exists('dustopac.inp'):
            print("\nCreated dustopac.inp with the following content:")
            with open('dustopac.inp', 'r') as f:
                print(f.read())
                
        # Display which dust opacity files are being used
        if os.path.exists('dustopac.inp'):
            print("\nUsing dust opacity files for:")
            if args.temp_dependent and n_species_in_temp_file >= 4:
                print(f"  - Material: {args.dust_material}")
                print(f"  - Grain size: {args.dust_size}μm")
                print(f"  - Temperature ranges: <50K, 50-150K, 150-250K, >250K")
                print(f"  - Files: dustkapscatmat_{args.dust_material}_[10K,100K,200K,300K]_a{args.dust_size}.inp")
            else:
                print(f"  - Material: {args.dust_material}")
                print(f"  - Grain size: {args.dust_size}μm")
                print(f"  - Number of species in temperature file: {n_species_in_temp_file}")
                if n_species_in_temp_file == 1:
                    print(f"  - File: dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp (copied to dust_kapscatmat_1.inp)")
                else:
                    print(f"  - File: dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp (copied to dust_kapscatmat_1.inp through dust_kapscatmat_{n_species_in_temp_file}.inp)")
    
    # Check if we should use existing temperature file
    using_existing_temp_file = False
    if args.use_existing_temperature:
        # Verify the specified temperature file exists and is valid
        if verify_dust_temperature_file(args.temperature_file):
            print("\n=== Using existing temperature file: {} ===".format(args.temperature_file))
            
            # Save current directory to return to it later if needed
            current_dir = os.getcwd()
            
            # Check the number of dust species in the temperature file
            n_species_in_temp_file = 1  # Default assumption
            try:
                with open(args.temperature_file, 'r') as f:
                    # Read format (should be 1)
                    iformat = int(f.readline().strip())
                    # Read number of cells
                    ncells = int(f.readline().strip())
                    # Read number of species
                    n_species_in_temp_file = int(f.readline().strip())
                
                print(f"Detected {n_species_in_temp_file} dust species in the temperature file.")
            except Exception as e:
                print(f"Warning: Could not read number of species from {args.temperature_file}: {e}")
                print("Assuming 1 dust species in temperature file.")
            
            # If temperature file is in a different location, copy it to the current directory
            if args.temperature_file != 'dust_temperature.dat' and os.path.exists(args.temperature_file):
                if not os.path.samefile(args.temperature_file, 'dust_temperature.dat'):
                    shutil.copy(args.temperature_file, 'dust_temperature.dat')
                    print(f"Copied {args.temperature_file} to {output_dir}/dust_temperature.dat")
            
            # Ensure we have the proper dust opacity files for the specified material and size
            from create_input import verify_dust_opacity_files, create_dustopac, copy_dust_opacity
            
            # Check if required dust opacity files exist
            if args.temp_dependent and n_species_in_temp_file >= 4:
                # Check for temperature-dependent files
                files_exist, missing_files = verify_dust_opacity_files(
                    dust_material=args.dust_material,
                    dust_size=args.dust_size,
                    temp_dependent=True
                )
                
                if not files_exist:
                    print(f"\nWARNING: Missing temperature-dependent dust opacity files for {args.dust_material} with size {args.dust_size}μm")
                    print("Will attempt to generate them using run_optool.py...")
                    
                    # Try to generate missing files
                    if os.path.exists('run_optool.py'):
                        try:
                            cmd = [
                                sys.executable, 'run_optool.py',
                                '--material', args.dust_material,
                                '--grain-size', str(args.dust_size),
                                '--temperatures', '10,100,200,300',
                                '--output-dir', '.'
                            ]
                            print(f"Running: {' '.join(cmd)}")
                            subprocess.run(cmd, check=True)
                            
                            # Check again if files were created
                            files_exist, still_missing = verify_dust_opacity_files(
                                dust_material=args.dust_material,
                                dust_size=args.dust_size,
                                temp_dependent=True
                            )
                            
                            if not files_exist:
                                print("\nWARNING: Could not generate all required dust opacity files.")
                                print("This may affect your results!")
                            else:
                                print("Successfully generated all required dust opacity files.")
                        except Exception as e:
                            print(f"Error running run_optool.py: {e}")
                            print("WARNING: This may affect your results!")
                    else:
                        print("run_optool.py not found. Cannot generate missing files.")
                        print("WARNING: This may affect your results!")
                
                # Create temperature-dependent dustopac.inp
                create_dustopac(temp_dependent=True, dust_material=args.dust_material, dust_size=args.dust_size)
                print(f"Created temperature-dependent dustopac.inp for {args.dust_material} with size {args.dust_size}μm")
            else:
                # Check for single dust opacity file
                main_opacity_file = f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp'
                
                if not os.path.exists(main_opacity_file):
                    print(f"\nWARNING: Missing dust opacity file {main_opacity_file}")
                    print("Will attempt to generate it using run_optool.py...")
                    
                    # Try to generate missing file
                    if os.path.exists('run_optool.py'):
                        try:
                            cmd = [
                                sys.executable, 'run_optool.py',
                                '--material', args.dust_material,
                                '--grain-size', str(args.dust_size),
                                '--no-temp-dependent',
                                '--output-dir', '.'
                            ]
                            print(f"Running: {' '.join(cmd)}")
                            subprocess.run(cmd, check=True)
                            
                            if os.path.exists(main_opacity_file):
                                print(f"Successfully generated {main_opacity_file}")
                            else:
                                print(f"WARNING: Could not generate {main_opacity_file}")
                                print("This may affect your results!")
                        except Exception as e:
                            print(f"Error running run_optool.py: {e}")
                            print("WARNING: This may affect your results!")
                    else:
                        print("run_optool.py not found. Cannot generate missing files.")
                        print("WARNING: This may affect your results!")
                
                # Create appropriate dustopac.inp for the number of species in temperature file
                if os.path.exists(main_opacity_file):
                    if n_species_in_temp_file == 1:
                        # Single species
                        copy_dust_opacity(dust_file=main_opacity_file, new_name='dust_kapscatmat_1.inp')
                        create_dustopac(nspecies=1, scattering_mode=args.scattering_mode_max, 
                                      water_fountain=False, dust_material=args.dust_material, dust_size=args.dust_size)
                        print(f"Created single-species dustopac.inp for {args.dust_material} with size {args.dust_size}μm")
                    else:
                        # Multiple species - copy the same opacity file for all species
                        for i in range(n_species_in_temp_file):
                            copy_dust_opacity(dust_file=main_opacity_file, new_name=f'dust_kapscatmat_{i+1}.inp')
                        
                        create_dustopac(nspecies=n_species_in_temp_file, scattering_mode=args.scattering_mode_max, 
                                      water_fountain=False, dust_material=args.dust_material, dust_size=args.dust_size)
                        print(f"Created {n_species_in_temp_file}-species dustopac.inp for {args.dust_material} with size {args.dust_size}μm")
                else:
                    print(f"WARNING: Required dust opacity file {main_opacity_file} not found.")
                    print("WARNING: This may affect your results!")
            
            # If output directory is different from current directory, fix dustopac.inp there too
            if os.path.exists(args.output_dir) and os.path.abspath(args.output_dir) != os.path.abspath('.'):
                # Copy dustopac.inp to output directory
                if os.path.exists('dustopac.inp'):
                    shutil.copy('dustopac.inp', os.path.join(args.output_dir, 'dustopac.inp'))
                    print(f"Copied dustopac.inp to {args.output_dir}")
                
                # Also create proper dustopac.inp directly in the output directory
                print(f"\nEnsuring correct dustopac.inp in output directory {args.output_dir}...")
                
                # Temporarily change to output directory
                os.chdir(args.output_dir)
                
                try:
                    # Try to create dustopac.inp with correct settings in output directory
                    if args.temp_dependent and n_species_in_temp_file >= 4:
                        # Check if all required files exist in output directory
                        if all(os.path.exists(f'dustkapscatmat_{args.dust_material}_{temp}K_a{args.dust_size}.inp') 
                              for temp in ['10', '100', '200', '300']):
                            create_dustopac(temp_dependent=True, dust_material=args.dust_material, dust_size=args.dust_size)
                            print(f"Created proper temperature-dependent dustopac.inp in {args.output_dir}")
                        else:
                            # Copy required files from current directory
                            for temp in ['10', '100', '200', '300']:
                                src_file = os.path.join(current_dir, f'dustkapscatmat_{args.dust_material}_{temp}K_a{args.dust_size}.inp')
                                if os.path.exists(src_file):
                                    shutil.copy(src_file, f'dustkapscatmat_{args.dust_material}_{temp}K_a{args.dust_size}.inp')
                            
                            # Try again to create dustopac.inp
                            if all(os.path.exists(f'dustkapscatmat_{args.dust_material}_{temp}K_a{args.dust_size}.inp') 
                                 for temp in ['10', '100', '200', '300']):
                                create_dustopac(temp_dependent=True, dust_material=args.dust_material, dust_size=args.dust_size)
                                print(f"Created proper temperature-dependent dustopac.inp in {args.output_dir}")
                            else:
                                print(f"WARNING: Could not create proper dustopac.inp in {args.output_dir} due to missing files")
                    else:
                        # Single or multi-species case
                        main_file = f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp'
                        if os.path.exists(main_file):
                            # Copy to standard names and create dustopac.inp
                            for i in range(n_species_in_temp_file):
                                copy_dust_opacity(dust_file=main_file, new_name=f'dust_kapscatmat_{i+1}.inp')
                            
                            create_dustopac(nspecies=n_species_in_temp_file, scattering_mode=args.scattering_mode_max,
                                          water_fountain=False, dust_material=args.dust_material, dust_size=args.dust_size)
                            print(f"Created proper {n_species_in_temp_file}-species dustopac.inp in {args.output_dir}")
                        else:
                            # Try to copy from current directory
                            src_file = os.path.join(current_dir, main_file)
                            if os.path.exists(src_file):
                                shutil.copy(src_file, main_file)
                                
                                # Now create the standard copies and dustopac.inp
                                for i in range(n_species_in_temp_file):
                                    copy_dust_opacity(dust_file=main_file, new_name=f'dust_kapscatmat_{i+1}.inp')
                                
                                create_dustopac(nspecies=n_species_in_temp_file, scattering_mode=args.scattering_mode_max,
                                              water_fountain=False, dust_material=args.dust_material, dust_size=args.dust_size)
                                print(f"Created proper {n_species_in_temp_file}-species dustopac.inp in {args.output_dir}")
                            else:
                                print(f"WARNING: Could not create proper dustopac.inp in {args.output_dir} due to missing files")
                except Exception as e:
                    print(f"Error creating dustopac.inp in output directory: {e}")
                
                # Return to original directory
                os.chdir(current_dir)
            
            # Set flag to skip temperature calculation
            using_existing_temp_file = True
            
            # Load the temperature data for visualization if needed
            if args.advanced_plots and not args.sed_only:
                try:
                    # Make sure the read_amr_grid function is properly imported
                    from radmc3d_aux import read_amr_grid, read_dust_temperature
                    
                    # Read grid and temperature data
                    grid_info = read_amr_grid()
                    temp_data, _ = read_dust_temperature()
                    print("Successfully loaded temperature data for visualization")
                except Exception as e:
                    print(f"Warning: Could not load temperature data for visualization: {e}")
                    temp_data = None
                    grid_info = None
        else:
            print(f"Warning: Could not use temperature file {args.temperature_file}. Will calculate temperature.")
            using_existing_temp_file = False
    
    # Only do temperature calculation if not using existing file
    if not using_existing_temp_file:
        # Copy dust opacity file(s)
        if args.temp_dependent:
            from create_input import copy_temp_dependent_dust_opacities
            
            # First try to use files from the output_dir
            if not copy_temp_dependent_dust_opacities(args.output_dir, dust_material=args.dust_material, dust_size=args.dust_size):
                # Then try current directory
                print("Trying to find dust opacity files in current directory...")
                if not copy_temp_dependent_dust_opacities(".", dust_material=args.dust_material, dust_size=args.dust_size):
                    print("Trying to find dust opacity files in original directory...")
                    if not copy_temp_dependent_dust_opacities(os.path.dirname(os.path.abspath(__file__)), 
                                                             dust_material=args.dust_material, dust_size=args.dust_size):
                        print("Trying to find dust opacity files in parent of original directory...")
                        # Try the parent of the original directory
                        parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
                        if not copy_temp_dependent_dust_opacities(parent_dir, 
                                                                dust_material=args.dust_material, dust_size=args.dust_size):
                            # Try generating the dust opacity files using run_optool.py
                            print("\nCould not find all required temperature-dependent dust opacity files.")
                            print(f"Attempting to generate them using run_optool.py for material {args.dust_material} and size {args.dust_size}μm...")
                            
                            # Check if run_optool.py exists in the original directory
                            run_optool_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_optool.py")
                            if not os.path.exists(run_optool_path):
                                run_optool_path = os.path.join(parent_dir, "run_optool.py")
                            
                            # Check if nk_optool directory exists in current directory
                            current_nk_dir = os.path.join(os.getcwd(), "nk_optool")
                            
                            if os.path.exists(run_optool_path) or os.path.exists("run_optool.py"):
                                try:
                                    # Use local or found run_optool.py
                                    optool_script = "run_optool.py" if os.path.exists("run_optool.py") else run_optool_path
                                    
                                    # Build command
                                    cmd = [
                                        sys.executable, optool_script, 
                                        '--material', args.dust_material,
                                        '--grain-size', str(args.dust_size),
                                        '--temperatures', '10,100,200,300',
                                        '--output-dir', args.output_dir
                                    ]
                                    
                                    # Add nk_dir if it exists
                                    if os.path.exists(current_nk_dir):
                                        cmd.extend(['--nk-dir', current_nk_dir])
                                    elif os.path.exists("nk_optool"):
                                        cmd.extend(['--nk-dir', 'nk_optool'])
                                    
                                    print(f"Running: {' '.join(cmd)}")
                                    subprocess.run(cmd, check=True)
                                    
                                    # Try copying the files again after generation
                                    if copy_temp_dependent_dust_opacities(args.output_dir, dust_material=args.dust_material, dust_size=args.dust_size):
                                        print("Successfully generated and found all required dust opacity files.")
                                    else:
                                        print("ERROR: Still missing some required dust opacity files after generation attempt.")
                                        print(f"Make sure all files exist in the output directory ({args.output_dir}):")
                                        print(f"- dustkapscatmat_{args.dust_material}_10K_a{args.dust_size}.inp")
                                        print(f"- dustkapscatmat_{args.dust_material}_100K_a{args.dust_size}.inp")
                                        print(f"- dustkapscatmat_{args.dust_material}_200K_a{args.dust_size}.inp")
                                        print(f"- dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp")
                                        return 1
                                except Exception as e:
                                    print(f"Error running run_optool.py: {e}")
                                    print("ERROR: Could not generate required dust opacity files.")
                                    print(f"Make sure all files exist in the output directory ({args.output_dir}):")
                                    print(f"- dustkapscatmat_{args.dust_material}_10K_a{args.dust_size}.inp")
                                    print(f"- dustkapscatmat_{args.dust_material}_100K_a{args.dust_size}.inp")
                                    print(f"- dustkapscatmat_{args.dust_material}_200K_a{args.dust_size}.inp")
                                    print(f"- dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp")
                                    return 1
                            else:
                                print("ERROR: Could not find all required temperature-dependent dust opacity files.")
                                print("ERROR: run_optool.py script not found to generate the files.")
                                print(f"Make sure all files exist in the output directory ({args.output_dir}):")
                                print(f"- dustkapscatmat_{args.dust_material}_10K_a{args.dust_size}.inp")
                                print(f"- dustkapscatmat_{args.dust_material}_100K_a{args.dust_size}.inp")
                                print(f"- dustkapscatmat_{args.dust_material}_200K_a{args.dust_size}.inp")
                                print(f"- dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp")
                                return 1
        else:
            # Check if the dust opacity file exists in output directory first
            output_dust_file = os.path.join(args.output_dir, args.dust_file)
            if os.path.exists(output_dust_file):
                print(f"Using existing {args.dust_file} from output directory")
                # Copy to current directory if needed
                if not os.path.exists(args.dust_file) or not os.path.samefile(output_dust_file, args.dust_file):
                    shutil.copy(output_dust_file, ".")
                    print(f"Copied {args.dust_file} from output directory to current directory")
            # Otherwise, check if the dust opacity file exists in current directory
            elif os.path.exists(args.dust_file):
                print(f"Using existing {args.dust_file} from current directory")
            # Otherwise, try to copy from original directory
            else:
                parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
                
                if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dust_file)):
                    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dust_file), ".")
                    print(f"Copied {args.dust_file} from original directory to current directory")
                # Otherwise, try to copy from parent directory
                elif os.path.exists(os.path.join(parent_dir, args.dust_file)):
                    shutil.copy(os.path.join(parent_dir, args.dust_file), ".")
                    print(f"Copied {args.dust_file} from parent directory to current directory")
                else:
                    print(f"Dust file {args.dust_file} not found in output, current, original or parent directory.")
                    print(f"Attempting to generate it using run_optool.py...")
                    
                    # Extract dust material and size from the dust_file name if not in standard format
                    try:
                        # Try to parse dust_material and dust_size from filename if possible
                        if args.dust_file.startswith('dustkapscatmat_') and args.dust_file.endswith('.inp'):
                            # Extract from filename like dustkapscatmat_E40R_300K_a0.3.inp
                            parts = args.dust_file.replace('.inp', '').split('_')
                            if len(parts) >= 4:
                                material = parts[1]
                                temp = parts[2].replace('K', '')
                                size = parts[3].replace('a', '')
                                
                                # Check if run_optool.py exists
                                run_optool_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_optool.py")
                                if not os.path.exists(run_optool_path):
                                    run_optool_path = os.path.join(parent_dir, "run_optool.py")
                                
                                if os.path.exists("run_optool.py") or os.path.exists(run_optool_path):
                                    # Use local or found run_optool.py
                                    optool_script = "run_optool.py" if os.path.exists("run_optool.py") else run_optool_path
                                    
                                    # Run the optool script to generate the single dust opacity file
                                    cmd = [
                                        sys.executable, optool_script, 
                                        '--material', material,
                                        '--grain-size', size,
                                        '--no-temp-dependent',
                                        '--output-dir', args.output_dir
                                    ]
                                    print(f"Running: {' '.join(cmd)}")
                                    subprocess.run(cmd, check=True)
                                    
                                    # Check if the file was generated in the output directory
                                    output_dust_file = os.path.join(args.output_dir, args.dust_file)
                                    if os.path.exists(output_dust_file):
                                        print(f"Successfully generated {args.dust_file}")
                                        # Copy to current directory
                                        shutil.copy(output_dust_file, ".")
                                        print(f"Copied {args.dust_file} from output directory to current directory")
                                    else:
                                        # Try with temperature-dependent mode and specific temp
                                        cmd = [
                                            sys.executable, optool_script, 
                                            '--material', material,
                                            '--grain-size', size,
                                            '--temperatures', temp,
                                            '--output-dir', args.output_dir
                                        ]
                                        print(f"Trying alternative command: {' '.join(cmd)}")
                                        subprocess.run(cmd, check=True)
                                        
                                        # Check again if file exists in output directory
                                        output_dust_file = os.path.join(args.output_dir, args.dust_file)
                                        if os.path.exists(output_dust_file):
                                            print(f"Successfully generated {args.dust_file}")
                                            # Copy to current directory
                                            shutil.copy(output_dust_file, ".")
                                            print(f"Copied {args.dust_file} from output directory to current directory")
                                        else:
                                            print(f"Warning: Could not generate {args.dust_file}")
                                else:
                                    print(f"Warning: run_optool.py not found, cannot generate {args.dust_file}")
                            else:
                                print(f"Warning: Could not parse dust material and size from {args.dust_file}")
                        else:
                            # Use the default dust_material and dust_size arguments
                            run_optool_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_optool.py")
                            if not os.path.exists(run_optool_path):
                                run_optool_path = os.path.join(parent_dir, "run_optool.py")
                            
                            if os.path.exists("run_optool.py") or os.path.exists(run_optool_path):
                                # Use local or found run_optool.py
                                optool_script = "run_optool.py" if os.path.exists("run_optool.py") else run_optool_path
                                
                                # Run the optool script with the default values
                                cmd = [
                                    sys.executable, optool_script, 
                                    '--material', args.dust_material,
                                    '--grain-size', str(args.dust_size),
                                    '--no-temp-dependent',
                                    '--output-dir', args.output_dir
                                ]
                                print(f"Running: {' '.join(cmd)}")
                                subprocess.run(cmd, check=True)
                                
                                # Check if a suitable file was generated in the output directory
                                output_dust_file = os.path.join(args.output_dir, args.dust_file)
                                if os.path.exists(output_dust_file):
                                    print(f"Successfully generated {args.dust_file}")
                                    # Copy to current directory
                                    shutil.copy(output_dust_file, ".")
                                    print(f"Copied {args.dust_file} from output directory to current directory")
                                else:
                                    print(f"Warning: Could not generate {args.dust_file}")
                            else:
                                print(f"Warning: run_optool.py not found, cannot generate {args.dust_file}")
                    except Exception as e:
                        print(f"Error attempting to generate dust opacity file: {e}")
                    
                    print(f"Warning: Dust file {args.dust_file} not found or could not be generated.")
                    print(f"Current directory: {os.getcwd()}")
                    print("Current directory contents:", os.listdir("."))
        
        # Parse temperature ranges if using temperature-dependent opacities
        if args.temp_dependent:
            temp_ranges = [float(t) for t in args.temp_ranges.split(',')]
            if len(temp_ranges) != 3:
                print("ERROR: temp_ranges must contain exactly 3 values (e.g., '50,150,250')")
                return 1
        
        # Convert parameters
        oangle_rad = args.oangle * np.pi / 180  # Convert degrees to radians
        
        # Set up the water fountain model
        print("\n=== Setting up water fountain model ===")
        grid_info, density = setup_model(
            model_type='water_fountain',
            rin=args.rin * au,
            rout=args.rout * au,
            nr=args.nr,
            ntheta=args.ntheta,
            stellar_radius=args.stellar_radius * Rsun,
            stellar_temp=args.stellar_temp,
            Mdtorus=args.Mdtorus,
            Mdlobe=args.Mdlobe,
            Rtorus=args.Rtorus * au,
            A=args.A,
            B=args.B,
            C=args.C,
            D=args.D,
            E=args.E,
            F=args.F,
            Rlobe=args.Rlobe * au,
            oangle=oangle_rad,
            width=args.width,
            scattering_mode_max=args.scattering_mode_max,
            dust_material=args.dust_material,
            dust_size=args.dust_size
        )
        
        # Backup original density for temperature-dependent iterations
        if args.temp_dependent:
            original_density = density.copy()
        
        # Create initial dustopac.inp file
        from create_input import create_dustopac
        if args.temp_dependent:
            create_dustopac(temp_dependent=True, dust_material=args.dust_material, dust_size=args.dust_size)
        else:
            create_dustopac(water_fountain=True, dust_material=args.dust_material, dust_size=args.dust_size)
        
        # Create RADMC-3D control file
        from create_input import create_radmc3d_control
        create_radmc3d_control(
            nphot_therm=int(args.nphotons_start),
            scattering_mode_max=args.scattering_mode_max,
            modified_random_walk=1,
            istar_sphere=1,
            water_fountain=True
        )
        
        # Initialize a variable to track the current iteration's temperature file
        current_temp_file = None
        
        # If using temperature-dependent opacities, we need an outer iteration loop
        if args.temp_dependent:
            print("\n=== Running temperature-dependent dust opacity iterations ===")
            
            # Verify that opacity files exist before starting
            required_files = [
                f'dustkapscatmat_{args.dust_material}_10K_a{args.dust_size}.inp',
                f'dustkapscatmat_{args.dust_material}_100K_a{args.dust_size}.inp', 
                f'dustkapscatmat_{args.dust_material}_200K_a{args.dust_size}.inp',
                f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp'
            ]
            
            print("\nVerifying opacity files...")
            missing_files = []
            for file in required_files:
                if not os.path.exists(file):
                    missing_files.append(file)
            
            if missing_files:
                print("Some required opacity files are missing. Attempting to generate them with run_optool.py...")
                
                # Check if run_optool.py exists in the current directory or original directory
                run_optool_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_optool.py")
                parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
                if not os.path.exists(run_optool_path):
                    run_optool_path = os.path.join(parent_dir, "run_optool.py")
                
                if os.path.exists("run_optool.py") or os.path.exists(run_optool_path):
                    try:
                        # Use local or found run_optool.py
                        optool_script = "run_optool.py" if os.path.exists("run_optool.py") else run_optool_path
                        
                        # Run the optool script to generate dust opacity files
                        cmd = [
                            sys.executable, optool_script, 
                            '--material', args.dust_material,
                            '--grain-size', str(args.dust_size),
                            '--temperatures', '10,100,200,300'
                        ]
                        print(f"Running: {' '.join(cmd)}")
                        subprocess.run(cmd, check=True)
                        
                        # Check if files were generated
                        still_missing = []
                        for file in missing_files:
                            if not os.path.exists(file):
                                still_missing.append(file)
                        
                        if still_missing:
                            print("ERROR: The following required opacity files are still missing after generation attempt:")
                            for file in still_missing:
                                print(f"  - {file}")
                            print("\nPlease make sure these files exist in the current directory.")
                            print("Current directory contents:")
                            for file in os.listdir('.'):
                                if file.startswith('dustkap'):
                                    print(f"  - {file}")
                            return 1
                        else:
                            print("All required opacity files were successfully generated.")
                    except Exception as e:
                        print(f"Error running run_optool.py: {e}")
                        print("ERROR: The following required opacity files are missing:")
                        for file in missing_files:
                            print(f"  - {file}")
                        print("\nPlease make sure these files exist in the current directory.")
                        print("Current directory contents:")
                        for file in os.listdir('.'):
                            if file.startswith('dustkap'):
                                print(f"  - {file}")
                        return 1
                else:
                    print("ERROR: The following required opacity files are missing:")
                    for file in missing_files:
                        print(f"  - {file}")
                    print("\nrun_optool.py script not found to generate the files.")
                    print("Please make sure these files exist in the current directory.")
                    print("Current directory contents:")
                    for file in os.listdir('.'):
                        if file.startswith('dustkap'):
                            print(f"  - {file}")
                    return 1
            else:
                print("All required opacity files found.")
            
            # Run initial temperature calculation with single dust species
            # (Using only 300K opacity for the first run)
            print("\n=== Starting initial mctherm calculation with 300K opacities ===")
            
            # Create temporary single-species dust opacity file
            from create_input import create_single_dust_opacity
            create_single_dust_opacity(dust_material=args.dust_material, dust_temperature='300K', dust_size=args.dust_size)
            
            # Verify the dustopac.inp file content
            print("dustopac.inp content:")
            with open('dustopac.inp', 'r') as f:
                print(f.read())
            
            # Run initial temperature calculation
            run_radmc3d(cmd='mctherm')
            
            # Read the initial temperature
            temp_data, _ = read_dust_temperature()
            
            # Create backup of the initial temperature
            shutil.copy("dust_temperature.dat", "dust_temperature_initial.dat")
            
            # Create a special dust density visualization for the initial phase
            if args.advanced_plots:
                print("\n=== Creating initial dust density visualization ===")
                try:
                    # Create figures directory if it doesn't exist
                    if not os.path.exists(args.figures_dir):
                        os.makedirs(args.figures_dir)
                    
                    # Read grid information if not already available
                    if 'grid_info' not in locals():
                        from radmc3d_aux import read_amr_grid
                        grid_info = read_amr_grid()
                    
                    # Read the original density for visualization
                    try:
                        from radmc3d_aux import read_dust_density, plot_initial_dust_density
                        
                        # Read density data
                        density_data, _, _, _ = read_dust_density(grid_info=grid_info)
                        
                        # Create the enhanced initial density plot with annotations
                        density_stats = plot_initial_dust_density(
                            grid_info=grid_info,
                            density_data=density_data,
                            output_dir=args.figures_dir,
                            save_fig=True,
                            show_fig=False,
                            add_annotations=True
                        )
                        
                        # Also create the original simple density plot for backward compatibility
                        # Simple function to create a clear dust density plot
                        def plot_dust_density(density_data, grid_info, output_dir, save_fig=True, show_fig=False):
                            """Create a clear visualization of the dust density structure."""
                            import matplotlib.pyplot as plt
                            import numpy as np
                            
                            if grid_info['type'] != 'spherical':
                                print("Warning: Only spherical coordinates are supported for dust density plots")
                                return
                            
                            # Create coordinate grid for visualization (R-z plane)
                            r = grid_info['rc']
                            theta = grid_info['thetac']
                            
                            R, T = np.meshgrid(r, theta, indexing='ij')
                            X = R * np.sin(T)
                            Y = R * np.cos(T)
                            
                            # For single species, plot density
                            if len(density_data.shape) == 2 or density_data.shape[-1] == 1:
                                # Extract 2D density slice
                                if len(density_data.shape) == 3:
                                    density_slice = density_data[:, :, 0]
                                else:
                                    density_slice = density_data
                                    
                                plt.figure(figsize=(12, 10))
                                plt.pcolormesh(X/1e15, Y/1e15, np.log10(density_slice),
                                            cmap='viridis', shading='auto')
                                plt.colorbar(label='log₁₀(density) [g/cm³]')
                                plt.axis('equal')
                                plt.xlabel('R [1000 AU]')
                                plt.ylabel('z [1000 AU]')
                                plt.title('Initial Dust Density Structure (Pre-Temperature-Dependent)')
                                
                                # Add grid lines for better visibility
                                plt.grid(True, linestyle='--', alpha=0.3)
                                
                                if save_fig:
                                    plt.savefig(f"{output_dir}/initial_dust_density.png", dpi=300, bbox_inches='tight')
                                
                                if not show_fig:
                                    plt.close()
                        
                        # Create the original style density plot too for backward compatibility
                        plot_dust_density(density_data, grid_info, args.figures_dir)
                        
                        # Print density statistics
                        if density_stats:
                            print("\n=== Initial Dust Density Statistics ===")
                            print(f"Minimum density: {density_stats['min_density']:.2e} g/cm³")
                            print(f"Maximum density: {density_stats['max_density']:.2e} g/cm³")
                            print(f"Mean density: {density_stats['mean_density']:.2e} g/cm³")
                            print(f"Median density: {density_stats['median_density']:.2e} g/cm³")
                        
                        print("Created initial dust density visualizations")
                        
                    except Exception as e:
                        print(f"Error creating initial dust density plot: {e}")
                except Exception as e:
                    print(f"Error in initial visualization: {e}")
            
            # Initialize variables for temperature-dependent iterations
            temp_converged = False
            temp_iterations = 0
            prev_temp = temp_data.copy()
            
            # Initialize photon tracking for temperature-dependent iterations
            temp_nphotons_history = [int(args.nphotons_start)]
            
            # Initialize arrays to track convergence metrics and cells changing groups
            temp_metrics_history = []
            cells_changed_history = []
            
            # Define function to classify cells into temperature groups
            def classify_temperature_groups(temperature, temp_ranges):
                """
                Classify cells into temperature groups based on temperature ranges.
                
                Parameters:
                -----------
                temperature : ndarray
                    Temperature data array
                temp_ranges : list
                    List of temperature boundaries [t1, t2, t3]
                    
                Returns:
                --------
                ndarray
                    Array with same shape as temperature, with values 0, 1, 2, 3
                    representing the temperature groups:
                    0: T < t1
                    1: t1 <= T < t2
                    2: t2 <= T < t3
                    3: T >= t3
                """
                t1, t2, t3 = temp_ranges
                groups = np.zeros_like(temperature, dtype=int)
                groups = np.where(temperature < t1, 0, groups)
                groups = np.where((temperature >= t1) & (temperature < t2), 1, groups)
                groups = np.where((temperature >= t2) & (temperature < t3), 2, groups)
                groups = np.where(temperature >= t3, 3, groups)
                return groups
            
            # Get initial temperature groups
            prev_groups = classify_temperature_groups(prev_temp, temp_ranges)
            
            # Outer loop for temperature-dependent iterations
            while not temp_converged and temp_iterations < args.iterations:
                temp_iterations += 1
                print(f"\n=== Temperature-dependent iteration {temp_iterations}/{args.iterations} ===")
                
                # Calculate number of photons for this iteration (increasing with each iteration)
                current_nphotons = int(min(args.nphotons_start * (args.scale_factor**(temp_iterations-1)), args.nphotons_max))
                temp_nphotons_history.append(current_nphotons)
                print(f"Using {current_nphotons:,} photon packages for this iteration")
                
                # Redistribute dust density based on temperature
                new_density = redistribute_density_by_temperature(
                    temp_data, 
                    original_density, 
                    grid_info, 
                    temp_ranges=temp_ranges
                )
                
                # Write the new dust density file
                write_temp_dependent_density(new_density, grid_info)
                
                # Make sure we're using the multi-species dust opacity file
                create_dustopac(temp_dependent=True, dust_material=args.dust_material, dust_size=args.dust_size)
                
                # Verify the dustopac.inp file content
                print("Multi-species dustopac.inp content:")
                with open('dustopac.inp', 'r') as f:
                    print(f.read())
                
                # Update RADMC-3D control file with new photon number
                with open('radmc3d.inp', 'w') as f:
                    f.write(f'nphot = {current_nphotons}\n')
                    f.write('modified_random_walk = 1\n')
                    f.write(f'scattering_mode_max = {args.scattering_mode_max}\n')
                    f.write('istar_sphere = 1\n')
                
                # Run a new temperature calculation with the redistributed densities
                print(f"\n=== Running mctherm with temperature-dependent opacities ({temp_iterations}) ===")
                print(f"Using {current_nphotons:,} photon packages")
                run_radmc3d(cmd='mctherm')
                
                # Read the new temperature
                new_temp_data, _ = read_dust_temperature()
                
                # Print temperature data shapes for debugging
                print(f"Previous temperature data shape: {prev_temp.shape}")
                print(f"New temperature data shape: {new_temp_data.shape}")
                
                # Create backup of this temperature file
                shutil.copy("dust_temperature.dat", f"dust_temperature_tempdep_{temp_iterations}.dat")
                
                # Check convergence between temperature iterations
                try:
                    print("\nChecking convergence between iterations...")
                    metrics = check_convergence(
                        new_temp_data, 
                        prev_temp, 
                        threshold=args.threshold, 
                        return_details=True
                    )
                    
                    # Store metrics history
                    temp_metrics_history.append(metrics)
                    
                    # Print convergence metrics
                    print(f"Maximum temperature difference: {metrics['max_diff']*100:.4f}%")
                    print(f"Mean temperature difference: {metrics['mean_diff']*100:.4f}%")
                    print(f"Median temperature difference: {metrics['median_diff']*100:.4f}%")
                    print(f"90th percentile difference: {metrics['p90_diff']*100:.4f}%")
                    
                    # Check cells changing temperature groups
                    new_groups = classify_temperature_groups(new_temp_data, temp_ranges)
                    cells_changed = np.sum(new_groups != prev_groups)
                    cells_total = new_groups.size
                    cells_changed_percent = (cells_changed / cells_total) * 100
                    
                    # Store cells changed history
                    cells_changed_history.append(cells_changed_percent)
                    
                    # Print cells changed information
                    print(f"Cells changed temperature group: {cells_changed} of {cells_total} ({cells_changed_percent:.2f}%)")
                    
                    # Update temperature groups for next iteration
                    prev_groups = new_groups.copy()
                    
                    # Update for next iteration - check both temperature convergence and group changes
                    temp_converged = (metrics['mean_diff'] < args.threshold and 
                                     cells_changed_percent < args.cells_change_threshold)
                    
                    # If converged, print reason
                    if temp_converged:
                        if metrics['mean_diff'] < args.threshold and cells_changed_percent < args.cells_change_threshold:
                            print(f"\nConvergence reached: both temperature difference (<{args.threshold*100:.1f}%) and group changes (<{args.cells_change_threshold:.1f}%) are below thresholds")
                        elif metrics['mean_diff'] < args.threshold:
                            print("\nConvergence reached: temperature difference is below threshold")
                        elif cells_changed_percent < args.cells_change_threshold:
                            print("\nConvergence reached: group changes are below threshold")
                    
                except Exception as e:
                    print(f"Error checking convergence: {e}")
                    print("Continuing with next iteration without convergence check...")
                    temp_converged = False
                
                # Update temperature data for next iteration
                prev_temp = new_temp_data.copy()
                temp_data = new_temp_data.copy()
                
                # Create temperature zone maps for each temperature-dependent iteration if requested
                if args.advanced_plots:
                    print(f"\n=== Creating temperature zone visualizations for iteration {temp_iterations} ===")
                    try:
                        # Create figures directory if it doesn't exist
                        if not os.path.exists(args.figures_dir):
                            os.makedirs(args.figures_dir)
                        
                        # Import the advanced plotting function
                        from radmc3d_aux import plot_advanced_temperature_density
                        # Assign it to the variable used in the code
                        plot_temp_func = plot_advanced_temperature_density
                        
                        # Try to read density data
                        try:
                            density_data, _, _, _ = read_dust_density(grid_info=grid_info)
                        except Exception as e:
                            print(f"Could not read density data: {e}")
                            density_data = None
                        
                        # Create visualizations for this iteration
                        # First, create a visualization showing all species
                        plot_temp_func(
                            grid_info=grid_info,
                            temp_data=temp_data,
                            density_data=density_data,
                            output_dir=args.figures_dir,
                            species=args.species_index,
                            multi_species_handling='all',  # Show all species
                            save_fig=True,
                            show_fig=False,
                            iteration=temp_iterations
                        )
                        
                        # Then create a visualization of the temperature zones
                        # This will specifically show the <50K, 50-150K, 150-250K, >250K regions
                        if args.multi_species_handling != 'all':
                            plot_temp_func(
                                grid_info=grid_info,
                                temp_data=temp_data,
                                density_data=density_data,
                                output_dir=args.figures_dir,
                                species=args.species_index,
                                multi_species_handling=args.multi_species_handling,
                                save_fig=True,
                                show_fig=False,
                                iteration=temp_iterations
                            )
                        
                        # Create a dedicated temperature zone map
                        plot_temp_func(
                            grid_info=grid_info,
                            temp_data=temp_data,
                            density_data=density_data,
                            output_dir=args.figures_dir,
                            species=args.species_index,
                            multi_species_handling=args.multi_species_handling,
                            save_fig=True,
                            show_fig=False,
                            iteration=temp_iterations,
                            create_zone_map=True  # Create a dedicated temperature zone map
                        )
                        
                        print(f"Saved temperature zone visualizations for iteration {temp_iterations} to {args.figures_dir}")
                    except Exception as e:
                        print(f"Error creating temperature zone visualizations: {e}")
                        print("Continuing with the calculation...")
                
                if temp_converged:
                    print("\n=== Temperature-dependent opacities converged! ===")
                else:
                    print(f"\n=== Continuing to iteration {temp_iterations+1} ===")
            
            # Print summary of temperature-dependent iterations
            print("\n=== Temperature-dependent iterations summary ===")
            print(f"Total iterations: {temp_iterations}")
            print("Photon packages used in each iteration:")
            for i, nphotons in enumerate(temp_nphotons_history[1:], 1):  # Skip the initial value
                print(f"  Iteration {i}: {nphotons:,} photons")
            print(f"Final photon count: {temp_nphotons_history[-1]:,}")
            
            # Create summary figure for temperature-dependent iterations
            if temp_iterations > 0:
                print("\n=== Creating summary figure for temperature-dependent iterations ===")
                try:
                    # Create figures directory if it doesn't exist
                    if not os.path.exists(args.figures_dir):
                        os.makedirs(args.figures_dir)
                    
                    # Use the specialized summary plot function
                    summary_filename = os.path.join(args.figures_dir, 'temp_dependent_summary.png')
                    plot_temperature_dependent_summary(
                        metrics_history=temp_metrics_history,
                        cells_changed_history=cells_changed_history,
                        temp_threshold=args.threshold,
                        cells_change_threshold=args.cells_change_threshold,
                        save_fig=True,
                        filename=summary_filename,
                        show_fig=False
                    )
                except Exception as e:
                    print(f"Error creating summary figure: {e}")
            
            # Save temperature-dependent convergence results
            np.savez('temp_dependent_convergence.npz', 
                    iterations=temp_iterations,
                    nphotons=temp_nphotons_history[1:],  # Skip the initial value
                    metrics=temp_metrics_history if len(temp_metrics_history) > 0 else None,
                    cells_changed=cells_changed_history if len(cells_changed_history) > 0 else None)
            
            # Set current_temp_file to the final temperature-dependent result
            current_temp_file = "dust_temperature.dat"
            
        # Run standard iterative Monte Carlo temperature calculation if not already done
        if not args.temp_dependent or current_temp_file is None:
            print("\n=== Starting iterative mctherm calculation ===")
            
            converged, results = iterative_mctherm(
                nphotons_start=args.nphotons_start,
                nphotons_max=args.nphotons_max,
                scale_factor=args.scale_factor,
                convergence_threshold=args.threshold,
                max_iterations=args.iterations,
                plot_slices=not args.no_plots,
                plot_progress=not args.no_plots,
                density_weighted=args.density_weighted,
                density_array=density if args.density_weighted else None,
                return_temp_history=True,  # Always return temperature history if we need iteration plots
                multi_species_handling=args.multi_species_handling,
                species_index=args.species_index,
                scattering_mode_max=args.scattering_mode_max
            )
            
            # Print final results
            print("\n=== Results ===")
            if converged:
                print("Temperature calculation converged successfully")
            else:
                print("Maximum iterations reached without convergence")
            
            print(f"Number of iterations: {results['iterations']}")
            print(f"Final photon count: {results['nphotons_history'][-1]:,.0f}")
            print(f"Total time: {results['elapsed_time']:.1f} seconds")
            
            # Save convergence results
            if results['metrics_history'] is not None:
                np.savez('convergence_results.npz', 
                        nphotons=results['nphotons_history'],
                        metrics=results['metrics_history'],
                        converged=converged)
                print("Saved convergence results to convergence_results.npz")
        else:
            # Just use the temperature-dependent results
            print("\n=== Using temperature-dependent dust opacity results ===")
            
            # If we want to create visualizations, we need to read the temperature data
            if args.advanced_plots:
                # Read the final temperature data (already saved in temp_data)
                pass
    
    # After the calculation finishes, add:
    if args.advanced_plots:
        # Only proceed with advanced plots when:
        # 1. Not in sed_only mode
        # 2. Either we did a calculation OR we're using existing temp file 
        if not args.sed_only and (not using_existing_temp_file or args.use_existing_temperature):
            print("\n=== Creating advanced visualizations ===")
            try:
                # Create figures directory if it doesn't exist
                if not os.path.exists(args.figures_dir):
                    os.makedirs(args.figures_dir)
                
                # Import the advanced plotting function
                from radmc3d_aux import plot_advanced_temperature_density, read_amr_grid, read_dust_temperature
                # Assign it to the variable used in the code
                plot_temp_func = plot_advanced_temperature_density
                
                # Read the temperature data if not already available
                if 'temp_data' not in locals() or temp_data is None:
                    try:
                        temp_data, grid_info = read_dust_temperature()
                        print("Read temperature data for visualization")
                    except Exception as e:
                        print(f"Error reading temperature data: {e}")
                        print("Cannot create visualizations without temperature data")
                        # Continue with the rest of the program
                        return 0
                
                # If grid_info is not available, try to read it
                if 'grid_info' not in locals() or grid_info is None:
                    try:
                        grid_info = read_amr_grid()
                        print("Read grid information for visualization")
                    except Exception as e:
                        print(f"Error reading grid information: {e}")
                        print("Cannot create visualizations without grid information")
                        # Continue with the rest of the program
                        return 0
                
                # Try to read density data if available
                try:
                    density_data, _, _, _ = read_dust_density(grid_info=grid_info)
                    
                    # Create a high-quality visualization of the initial density structure
                    # This ensures we have this plot in both temperature-dependent and non-temperature-dependent cases
                    if not args.temp_dependent:
                        print("\n=== Creating initial dust density visualization ===")
                        density_stats = plot_initial_dust_density(
                            grid_info=grid_info,
                            density_data=density_data,
                            output_dir=args.figures_dir,
                            save_fig=True,
                            show_fig=False,
                            add_annotations=True
                        )
                        
                        if density_stats:
                            print("\n=== Dust Density Statistics ===")
                            print(f"Minimum density: {density_stats['min_density']:.2e} g/cm³")
                            print(f"Maximum density: {density_stats['max_density']:.2e} g/cm³")
                            print(f"Mean density: {density_stats['mean_density']:.2e} g/cm³")
                            print(f"Median density: {density_stats['median_density']:.2e} g/cm³")
                    
                except Exception as e:
                    print(f"Could not read density data: {e}")
                    density_data = None
                
                # Create advanced visualizations for final result
                stats = plot_temp_func(
                    grid_info=grid_info,
                    temp_data=temp_data,
                    density_data=density_data,
                    output_dir=args.figures_dir,
                    species=args.species_index,  # Use the specified species index
                    multi_species_handling=args.multi_species_handling  # Use the specified handling method
                )
                
                # Also create a dedicated temperature zone map for the final result
                plot_temp_func(
                    grid_info=grid_info,
                    temp_data=temp_data,
                    density_data=density_data,
                    output_dir=args.figures_dir,
                    species=args.species_index,
                    multi_species_handling=args.multi_species_handling,
                    save_fig=True,
                    show_fig=False,
                    create_zone_map=True  # Create a dedicated temperature zone map
                )
                
                # Print additional temperature statistics
                print("\n=== Temperature Statistics ===")
                print(f"Minimum temperature: {stats['min_temperature']:.2f} K")
                print(f"Maximum temperature: {stats['max_temperature']:.2f} K")
                print(f"Mean temperature: {stats['mean_temperature']:.2f} K")
                print(f"Median temperature: {stats['median_temperature']:.2f} K")
                
                # Create plots for each iteration if requested
                if args.save_iter_plots and not args.temp_dependent and 'results' in locals() and 'temp_history' in results and results['temp_history']:
                    print("\n=== Creating temperature plots for each iteration ===")
                    temp_history = results['temp_history']
                    
                    for i, iter_temp in enumerate(temp_history, 1):
                        # Create temperature plot for this iteration
                        print(f"Creating plots for iteration {i}...")
                        
                        # Create advanced temperature plots for this iteration
                        iter_stats = plot_temp_func(
                            grid_info=grid_info,
                            temp_data=iter_temp,
                            density_data=density_data,  # Use the same density data for all iterations
                            output_dir=args.figures_dir,  # Save directly to figures directory
                            species=args.species_index,
                            multi_species_handling=args.multi_species_handling,
                            save_fig=True,
                            show_fig=False,
                            iteration=i
                        )
                        
                        # No need to rename the files since they now include species information
                    
                    print(f"Iteration plots saved to {args.figures_dir}")
                    
            except Exception as e:
                print(f"Error creating advanced visualizations: {e}")
                print("Continuing with the rest of the program...")
    
    # Compute SED if requested
    if args.compute_sed:
        # Check that all required files exist for SED computation
        required_files = ['dust_temperature.dat', 'amr_grid.inp', 'wavelength_micron.inp', 
                         'dust_density.inp', 'stars.inp', 'radmc3d.inp']
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print("\nERROR: The following required files for SED calculation are missing:")
            for file in missing_files:
                print(f"  - {file}")
            
            if args.sed_only:
                print("\nWhen using --sed_only, ensure all required RADMC-3D input files are in the specified directory.")
                print(f"Current directory: {os.path.abspath('.')}")
                print("Required files:")
                print("  - dust_temperature.dat: Dust temperature distribution")
                print("  - amr_grid.inp: Grid specification")
                print("  - wavelength_micron.inp: Wavelength grid")
                print("  - dustopac.inp: Dust opacity specification")
                print("  - dust_density.inp: Dust density distribution")
                print("  - stars.inp: Stellar parameters")
                print("  - radmc3d.inp: RADMC-3D control file")
            
            print("\nCannot compute SED without these files. Exiting.")
            return 1
        
        print("\n=== Computing SED ===")
        print(f"Observer position: inclination = {args.inclination}°, phi = {args.phi}°")
        
        # Check if we need to regenerate dustopac.inp with the correct material and size
        from create_input import verify_dust_opacity_files, create_dustopac
        
        # First, save current directory to return to it later if needed
        current_dir = os.getcwd()
        
        # Check dust opacity files
        if args.temp_dependent:
            # Check temperature-dependent dust opacity files
            files_exist, missing_opacity_files = verify_dust_opacity_files(
                dust_material=args.dust_material, 
                dust_size=args.dust_size,
                temp_dependent=True
            )
            
            if not files_exist:
                print(f"\nWARNING: Missing temperature-dependent dust opacity files for {args.dust_material} with size {args.dust_size}μm")
                print("Will attempt to generate them using run_optool.py...")
                
                # Try to generate missing files
                if os.path.exists('run_optool.py'):
                    try:
                        cmd = [
                            sys.executable, 'run_optool.py',
                            '--material', args.dust_material,
                            '--grain-size', str(args.dust_size),
                            '--temperatures', '10,100,200,300',
                            '--output-dir', '.'
                        ]
                        print(f"Running: {' '.join(cmd)}")
                        subprocess.run(cmd, check=True)
                        
                        # Check again if files were created
                        files_exist, still_missing = verify_dust_opacity_files(
                            dust_material=args.dust_material,
                            dust_size=args.dust_size,
                            temp_dependent=True
                        )
                        
                        if not files_exist:
                            print("\nWARNING: Could not generate all required dust opacity files.")
                            print("SED calculation may fail!")
                        else:
                            print("Successfully generated all required dust opacity files.")
                    except Exception as e:
                        print(f"Error running run_optool.py: {e}")
                        print("SED calculation may fail!")
                else:
                    print("run_optool.py not found. Cannot generate missing files.")
                    print("SED calculation may fail!")
            
            # (Re)create dustopac.inp with the correct material and size
            create_dustopac(temp_dependent=True, dust_material=args.dust_material, dust_size=args.dust_size)
            print(f"Created dustopac.inp for temperature-dependent opacities with material {args.dust_material} and size {args.dust_size}μm")
            
            # If output_dir is different from current_dir, update dustopac.inp there as well
            output_dir_abs = os.path.abspath(args.output_dir)
            current_dir_abs = os.path.abspath('.')
            if output_dir_abs != current_dir_abs and os.path.exists(output_dir_abs):
                # Save current dustopac.inp to the output directory
                if os.path.exists('dustopac.inp'):
                    shutil.copy('dustopac.inp', os.path.join(output_dir_abs, 'dustopac.inp'))
                    print(f"Copied updated dustopac.inp to {output_dir_abs}")
            
        else:
            # Check single dust opacity file
            files_exist, missing_opacity_files = verify_dust_opacity_files(
                dust_material=args.dust_material,
                dust_size=args.dust_size,
                temp_dependent=False
            )
            
            if not files_exist:
                print(f"\nWARNING: Missing dust opacity file for {args.dust_material} with size {args.dust_size}μm")
                print("Will attempt to generate it using run_optool.py...")
                
                # Try to generate missing file
                if os.path.exists('run_optool.py'):
                    try:
                        cmd = [
                            sys.executable, 'run_optool.py',
                            '--material', args.dust_material,
                            '--grain-size', str(args.dust_size),
                            '--no-temp-dependent',
                            '--output-dir', '.'
                        ]
                        print(f"Running: {' '.join(cmd)}")
                        subprocess.run(cmd, check=True)
                        
                        # Check again if file was created
                        files_exist, still_missing = verify_dust_opacity_files(
                            dust_material=args.dust_material,
                            dust_size=args.dust_size,
                            temp_dependent=False
                        )
                        
                        if not files_exist:
                            print("\nWARNING: Could not generate required dust opacity file.")
                            print("SED calculation may fail!")
                        else:
                            print("Successfully generated required dust opacity file.")
                    except Exception as e:
                        print(f"Error running run_optool.py: {e}")
                        print("SED calculation may fail!")
                else:
                    print("run_optool.py not found. Cannot generate missing file.")
                    print("SED calculation may fail!")
            
            # (Re)create dustopac.inp with the correct material and size
            from create_input import copy_dust_opacity
            main_opacity_file = f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp'
            
            # Make sure we have the file
            if os.path.exists(main_opacity_file):
                # Copy the file to the standard name used in non-temperature-dependent mode
                copy_dust_opacity(dust_file=main_opacity_file, new_name='dust_kapscatmat_1.inp')
                
                # Create dustopac.inp with the correct parameters
                create_dustopac(nspecies=1, scattering_mode=args.scattering_mode_max,
                               water_fountain=False, dust_material=args.dust_material, dust_size=args.dust_size)
                print(f"Created dustopac.inp for single opacity with material {args.dust_material} and size {args.dust_size}μm")
                
                # If output_dir is different from current_dir, copy the files there as well
                output_dir_abs = os.path.abspath(args.output_dir)
                current_dir_abs = os.path.abspath('.')
                if output_dir_abs != current_dir_abs and os.path.exists(output_dir_abs):
                    # Copy dust_kapscatmat_1.inp to the output directory
                    if os.path.exists('dust_kapscatmat_1.inp'):
                        shutil.copy('dust_kapscatmat_1.inp', os.path.join(output_dir_abs, 'dust_kapscatmat_1.inp'))
                        print(f"Copied dust_kapscatmat_1.inp to {output_dir_abs}")
                    
                    # Copy dustopac.inp to the output directory
                    if os.path.exists('dustopac.inp'):
                        shutil.copy('dustopac.inp', os.path.join(output_dir_abs, 'dustopac.inp'))
                        print(f"Copied updated dustopac.inp to {output_dir_abs}")
            else:
                print(f"WARNING: Required dust opacity file {main_opacity_file} not found.")
                print("SED calculation may fail!")
                
        # Force creation of correct dustopac.inp directly in the output directory
        if os.path.exists(args.output_dir) and os.path.abspath(args.output_dir) != os.path.abspath('.'):
            print(f"\nEnsuring correct dustopac.inp in output directory {args.output_dir}...")
            
            # Temporarily change to output directory
            os.chdir(args.output_dir)
            
            # Check if we have the required opacity files here
            try:
                # Try to use files that exist in this directory
                if args.temp_dependent:
                    if all(os.path.exists(f'dustkapscatmat_{args.dust_material}_{temp}K_a{args.dust_size}.inp') 
                          for temp in ['10', '100', '200', '300']):
                        # Create the dustopac.inp file with the correct parameters
                        create_dustopac(temp_dependent=True, dust_material=args.dust_material, dust_size=args.dust_size)
                        print(f"Created proper dustopac.inp in {args.output_dir} with material {args.dust_material} and size {args.dust_size}μm")
                    else:
                        print(f"WARNING: Not all required temperature-dependent opacity files exist in {args.output_dir}")
                        print("Using current dustopac.inp which may have incorrect material/size settings")
                else:
                    # Single species
                    main_file = f'dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp'
                    if os.path.exists(main_file):
                        # Copy to the standard name and create dustopac.inp
                        copy_dust_opacity(dust_file=main_file, new_name='dust_kapscatmat_1.inp')
                        create_dustopac(nspecies=1, scattering_mode=args.scattering_mode_max,
                                      water_fountain=False, dust_material=args.dust_material, dust_size=args.dust_size)
                        print(f"Created proper dustopac.inp in {args.output_dir} with material {args.dust_material} and size {args.dust_size}μm")
                    else:
                        print(f"WARNING: Required opacity file {main_file} not found in {args.output_dir}")
                        print("Using current dustopac.inp which may have incorrect material/size settings")
            except Exception as e:
                print(f"Error creating dustopac.inp in output directory: {e}")
            
            # Return to original directory
            os.chdir(current_dir)
        
        # Display which dust opacity files are being used
        if os.path.exists('dustopac.inp'):
            print("\nUsing dust opacity files for:")
            if args.temp_dependent:
                print(f"  - Material: {args.dust_material}")
                print(f"  - Grain size: {args.dust_size}μm")
                print(f"  - Temperature ranges: <50K, 50-150K, 150-250K, >250K")
                print(f"  - Files: dustkapscatmat_{args.dust_material}_[10K,100K,200K,300K]_a{args.dust_size}.inp")
            else:
                print(f"  - Material: {args.dust_material}")
                print(f"  - Grain size: {args.dust_size}μm")
                print(f"  - File: dustkapscatmat_{args.dust_material}_300K_a{args.dust_size}.inp (copied to dust_kapscatmat_1.inp)")
        
        # Run RADMC-3D with the sed command
        sed_cmd = f"sed"
        # Add sloppy option by default unless disabled
        if not args.no_sloppy:
            sed_cmd += " sloppy"
            print("Using 'sloppy' mode for faster but less accurate SED calculation")
        else:
            print("Using standard (non-sloppy) mode for accurate SED calculation")
        # Add inclination and phi parameters
        sed_cmd += f" incl {args.inclination} phi {args.phi}"
        
        run_result = run_radmc3d(cmd=sed_cmd)
        
        if run_result == 0:
            print("SED calculation completed successfully")
            # Check if spectrum.out was created
            if os.path.exists('spectrum.out'):
                print(f"SED data saved to {output_dir}/spectrum.out")
                
                # Copy to a unique filename including inclination and phi angles
                unique_name = f"spectrum_{args.inclination}deg_{args.phi}deg.out"
                shutil.copy('spectrum.out', unique_name)
                print(f"Also saved to {output_dir}/{unique_name}")
            else:
                print("Warning: spectrum.out file not found after SED calculation")
        else:
            print(f"Error: SED calculation failed with return code {run_result}")
            return 1  # Return an error code
    
    print("\nDone.")
    
    return 0

if __name__ == "__main__":
    main() 