#!/usr/bin/env python3
"""
RADMC-3D Iterative Dust Temperature Calculation for Water Fountain Models

This script performs iterative dust temperature calculations with RADMC-3D,
increasing the number of photon packages until convergence is reached.
Specifically optimized for water fountain astrophysical objects.
"""

import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import time
import shutil
import argparse
import subprocess
from create_input import setup_model, au, Rsun, Msun
from radmc3d_aux import (run_radmc3d, read_dust_temperature, check_convergence, 
                        plot_temperature_slice, plot_convergence_history, 
                        compare_temperature_slices, analyze_temperature_distribution,
                        plot_advanced_temperature_density, read_dust_density)

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
                   os.path.join(backup_dir, f"dust_temperature_iter{iteration}.dat"))
        print(f"Backed up temperature file for iteration {iteration}")
    else:
        print("Warning: dust_temperature.dat not found, could not create backup")

def iterative_mctherm(nphotons_start=1e6, nphotons_max=1e8, 
                      scale_factor=5.0, convergence_threshold=0.01,
                      max_iterations=10, plot_slices=True, plot_progress=True,
                      density_weighted=False, density_array=None,
                      return_temp_history=False, setseed=None):
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
    setseed : int, optional
        Random seed for reproducible results
    
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
        
        print(f"\n===== Iteration {iteration}/{max_iterations} =====")
        print(f"Running RADMC-3D with {nphotons:,.0f} photon packages...")
        
        # Update RADMC-3D control file with new photon number
        with open('radmc3d.inp', 'w') as f:
            f.write(f'nphot = {nphotons}\n')
            f.write('modified_random_walk = 1\n')
            f.write('scattering_mode_max = 1\n')
            f.write('istar_sphere = 1\n')
        
        # Run RADMC-3D Monte Carlo thermal calculation
        run_radmc3d(cmd='mctherm', setseed=setseed)
        
        # Read temperature data
        temp_data, grid_info = read_dust_temperature()
        
        # Backup the temperature file
        backup_temperature_file(iteration)
        
        # Plot temperature slice for this iteration
        if plot_slices:
            # For water fountain models, we're using a 2D spherical grid (r, theta)
            # We'll create a directory for temperature plots if it doesn't exist
            os.makedirs("temp_plots", exist_ok=True)
            
            # Instead of using plot_temperature_slice, we'll use plot_advanced_temperature_density
            # which automatically converts to (R, z) coordinates
            from radmc3d_aux import plot_advanced_temperature_density
            
            # Create advanced temperature plots for this iteration
            plot_advanced_temperature_density(
                grid_info=grid_info,
                temp_data=temp_data,
                density_data=None,  # We don't need density plots for convergence checking
                output_dir="temp_plots",
                species=0,
                save_fig=True,
                show_fig=False,
                iteration=iteration
            )
            
            # Rename the files to include iteration number in the temp_plots directory
            os.rename(
                os.path.join("temp_plots", 'dust_temperature.png'),
                os.path.join("temp_plots", f'temperature_iter{iteration}.png')
            )
            os.rename(
                os.path.join("temp_plots", 'temperature_zones_map.png'),
                os.path.join("temp_plots", f'temperature_zones_iter{iteration}.png')
            )
        
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
            print(f"Maximum difference: {metrics['max_diff']*100:.4f}%")
            print(f"Mean difference: {metrics['mean_diff']*100:.4f}%")
            print(f"Median difference: {metrics['median_diff']*100:.4f}%")
            print(f"90th percentile difference: {metrics['p90_diff']*100:.4f}%")
            
            # Check if converged
            if metrics['max_diff'] < convergence_threshold:
                converged = True
                print(f"\nConvergence reached after {iteration} iterations!")
                print(f"Final photon count: {nphotons:,.0f}")
                break
        
        # Store current temperature for next iteration
        previous_temp = temp_data.copy()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nTotal calculation time: {elapsed_time:.1f} seconds")
    
    # Plot convergence history
    if plot_progress and len(metrics_history) > 0:
        plot_convergence_history(metrics_history, nphotons_history, 
                                save_fig=True, filename="temp_plots/convergence_history.png",
                                threshold=convergence_threshold, show_fig=False)
        
        # Plot comparison between first and last temperature
        if return_temp_history and len(temp_history) >= 2:
            compare_temperature_slices(temp_history[0], temp_history[-1], grid_info,
                                     slice_dim='phi', slice_idx=0,
                                     save_fig=True, filename="temp_plots/temperature_comparison.png",
                                     titles=["First iteration", "Last iteration"], show_fig=False)
            
            # Plot temperature distribution
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
    
    # Convergence parameters
    parser.add_argument('--nphotons_start', type=float, default=1e6, help='Initial number of photon packages')
    parser.add_argument('--nphotons_max', type=float, default=1e8, help='Maximum number of photon packages')
    parser.add_argument('--scale_factor', type=float, default=5.0, help='Factor to increase photons by each iteration')
    parser.add_argument('--threshold', type=float, default=0.01, help='Convergence threshold (maximum relative difference)')
    parser.add_argument('--max_iterations', type=int, default=10, help='Maximum number of iterations')
    
    # Parallelization options
    parser.add_argument('--nthreads', type=int, default=4, help='Number of OpenMP threads for parallelization')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelization (requires RADMC-3D compiled with MPI)')
    parser.add_argument('--np', type=int, default=4, help='Number of MPI processes (if --mpi is used)')
    
    # Other options
    parser.add_argument('--no_plots', action='store_true', help='Disable plotting')
    parser.add_argument('--density_weighted', action='store_true', help='Use density-weighted convergence metrics')
    parser.add_argument('--setseed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='radmc3d_model', help='Output directory name')
    parser.add_argument('--dust_file', type=str, default='dustkapscatmat_E40R_300K_a0.3.inp', help='Dust opacity file')
    
    # Add advanced visualization option
    parser.add_argument('--advanced_plots', action='store_true', help='Create advanced temperature and density visualizations')
    parser.add_argument('--figures_dir', type=str, default='figures', help='Directory to save figures')
    parser.add_argument('--save_iter_plots', action='store_true', help='Save advanced temperature plots for each iteration')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Change to output directory
    os.chdir(args.output_dir)
    
    # Copy dust opacity file if needed
    if os.path.exists(f"../{args.dust_file}"):
        shutil.copy(f"../{args.dust_file}", ".")
        print(f"Copied {args.dust_file} to {args.output_dir}")
    else:
        print(f"Warning: Dust file {args.dust_file} not found in parent directory")
        print("Current directory:", os.getcwd())
        print("Parent directory contents:", os.listdir(".."))
    
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
        width=args.width
    )
    
    # Create RADMC-3D control file with parallelization options
    from create_input import create_radmc3d_control
    create_radmc3d_control(
        nphot_therm=int(args.nphotons_start),
        scattering_mode_max=1,
        modified_random_walk=1,
        istar_sphere=1,
        water_fountain=True,
        nthreads=args.nthreads
    )
    
    # Set up MPI command prefix if using MPI
    mpi_prefix = ""
    if args.mpi:
        mpi_prefix = f"mpirun -np {args.np} "
        print(f"\n=== Using MPI parallelization with {args.np} processes ===")
    else:
        print(f"\n=== Using OpenMP parallelization with {args.nthreads} threads ===")
    
    # Run iterative Monte Carlo temperature calculation
    print("\n=== Starting iterative mctherm calculation ===")
    
    # If using MPI, we need to modify the run_radmc3d function call
    if args.mpi:
        # Import the run_radmc3d function to use with MPI
        from radmc3d_aux import run_radmc3d
        
        # Monkey patch the run_radmc3d function to use MPI
        original_run_radmc3d = run_radmc3d
        
        def mpi_run_radmc3d(cmd='mctherm', params=None, verbose=True, setseed=None):
            """Wrapper for run_radmc3d that uses MPI"""
            # Build command with MPI prefix
            command = [f"{mpi_prefix}radmc3d"]
            
            # Add command
            command.append(cmd)
            
            # Add parameters
            if params:
                for key, value in params.items():
                    command.append(f"{key}={value}")
            
            # Add random seed if specified
            if setseed is not None:
                command.append(f"setseed={setseed}")
            
            # Convert command list to string
            command_str = ' '.join(command)
            
            # Print command
            if verbose:
                print(f"Running: {command_str}")
            
            # Run command
            process = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            # Print output
            if verbose:
                if stdout:
                    print(stdout.decode())
                if stderr:
                    print(stderr.decode())
            
            return process.returncode
        
        # Replace the run_radmc3d function with our MPI version
        import radmc3d_aux
        radmc3d_aux.run_radmc3d = mpi_run_radmc3d
    
    converged, results = iterative_mctherm(
        nphotons_start=args.nphotons_start,
        nphotons_max=args.nphotons_max,
        scale_factor=args.scale_factor,
        convergence_threshold=args.threshold,
        max_iterations=args.max_iterations,
        plot_slices=not args.no_plots,
        plot_progress=not args.no_plots,
        density_weighted=args.density_weighted,
        density_array=density if args.density_weighted else None,
        return_temp_history=True,  # Always return temperature history if we need iteration plots
        setseed=args.setseed
    )
    
    # Restore original run_radmc3d function if we modified it
    if args.mpi:
        radmc3d_aux.run_radmc3d = original_run_radmc3d
    
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
    
    # After the iterative calculation finishes, add:
    if args.advanced_plots:
        print("\n=== Creating advanced visualizations ===")
        try:
            # Create figures directory if it doesn't exist
            if not os.path.exists(args.figures_dir):
                os.makedirs(args.figures_dir)
            
            # Read the final temperature data
            temp_data, grid_info = read_dust_temperature()
            
            # Try to read density data if available
            try:
                density_data, _, _, _ = read_dust_density(grid_info=grid_info)
            except Exception as e:
                print(f"Could not read density data: {e}")
                density_data = None
            
            # Create advanced visualizations for final result
            stats = plot_advanced_temperature_density(
                grid_info=grid_info,
                temp_data=temp_data,
                density_data=density_data,
                output_dir=args.figures_dir,
                species=0  # Default to first dust species
            )
            
            # Print additional temperature statistics
            print("\n=== Temperature Statistics ===")
            print(f"Minimum temperature: {stats['min_temperature']:.2f} K")
            print(f"Maximum temperature: {stats['max_temperature']:.2f} K")
            print(f"Mean temperature: {stats['mean_temperature']:.2f} K")
            print(f"Median temperature: {stats['median_temperature']:.2f} K")
            
            # Create plots for each iteration if requested
            if args.save_iter_plots and 'temp_history' in results and results['temp_history']:
                print("\n=== Creating temperature plots for each iteration ===")
                temp_history = results['temp_history']
                
                for i, iter_temp in enumerate(temp_history, 1):
                    # Create temperature plot for this iteration
                    print(f"Creating plots for iteration {i}...")
                    
                    # Create advanced temperature plots for this iteration
                    iter_stats = plot_advanced_temperature_density(
                        grid_info=grid_info,
                        temp_data=iter_temp,
                        density_data=density_data,  # Use the same density data for all iterations
                        output_dir=args.figures_dir,  # Save directly to figures directory
                        species=0,
                        save_fig=True,
                        show_fig=False,
                        iteration=i
                    )
                    
                    # Rename the files to include iteration number
                    os.rename(
                        os.path.join(args.figures_dir, 'dust_temperature.png'),
                        os.path.join(args.figures_dir, f'dust_temperature_iter_{i}.png')
                    )
                    os.rename(
                        os.path.join(args.figures_dir, 'temperature_zones_map.png'),
                        os.path.join(args.figures_dir, f'temperature_zones_map_iter_{i}.png')
                    )
                
                print(f"Iteration plots saved to {args.figures_dir}")
                
        except Exception as e:
            print(f"Error creating advanced visualizations: {e}")
            print("Continuing with the rest of the program...")
    
    print("\nDone.")
    
    return 0

if __name__ == "__main__":
    main() 