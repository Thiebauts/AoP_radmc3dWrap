#!/usr/bin/env python3
"""
Optool Runner for RADMC-3D Dust Temperature Calculations

This script provides functions to run Optool to generate dust opacity files
for various materials, temperatures, and grain sizes, which are used by
RADMC-3D for dust temperature calculations.

References:
-----------
Optool: C. Dominik et al., https://github.com/cdominik/optool
"""

import os
import subprocess
import shutil
import argparse
import sys
from pathlib import Path
import re

# Dictionary of densities (g/cm³) for each composition
DENSITIES = {
    'x035': 2.7,  # (0.65)MgO-(0.35)SiO2
    'x040': 2.7,  # (0.60)MgO-(0.40)SiO2
    'x050A': 2.7, # (0.50)MgO-(0.50)SiO2 structure A
    'x050B': 2.7,  # (0.50)MgO-(0.50)SiO2 structure B
    'E10': 2.8,  # Mg(0.9)Fe(0.1)SiO3; Fe3+  
    'E10R': 2.8,  # Mg(0.9)Fe(0.1)SiO3; Fe2+
    'E20': 2.9,  # Mg(0.8)Fe(0.2)SiO3; Fe3+
    'E20R': 2.9,  # Mg(0.8)Fe(0.2)SiO3; Fe2+
    'E30': 3.0,  # Mg(0.7)Fe(0.3)SiO3; Fe3+
    'E30R': 3.0,  # Mg(0.7)Fe(0.3)SiO3; Fe2+
    'E40': 3.1,  # Mg(0.6)Fe(0.4)SiO3; Fe3+
    'E40R': 3.1   # Mg(0.6)Fe(0.4)SiO3; Fe2+    
}

# Temperature values for which to generate opacity files
TEMPERATURE_VALUES = [10, 100, 200, 300]

def check_optool_installed():
    """
    Check if Optool is installed and available in the PATH.
    
    Returns:
    --------
    bool
        True if Optool is installed, False otherwise
    """
    try:
        result = subprocess.run(["optool", "--version"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                universal_newlines=True, 
                                check=False)
        
        # Check if we got some kind of output that suggests Optool exists
        if result.returncode == 0 or "optool" in result.stdout.lower() or "optool" in result.stderr.lower():
            return True
        return False
    except FileNotFoundError:
        return False

def find_nk_file(material, temp=None, nk_dir="nk_optool"):
    """
    Find the .lnk file for the specified material and temperature in the nk_dir.
    
    Parameters:
    -----------
    material : str
        Material identifier (e.g., 'E40R')
    temp : int, optional
        Temperature in K to look for specific temperature file
    nk_dir : str
        Directory containing the .lnk files
    
    Returns:
    --------
    str or None
        Path to the .lnk file if found, None otherwise
    """
    if not os.path.exists(nk_dir):
        print(f"Error: nk directory {nk_dir} not found.")
        return None
    
    if temp is not None:
        # Try exact match with temperature
        temp_match = os.path.join(nk_dir, f"{material}_{temp}K.lnk")
        if os.path.exists(temp_match):
            print(f"Found temperature-specific file: {temp_match}")
            return temp_match
    
    # Try exact match without temperature
    exact_match = os.path.join(nk_dir, f"{material}.lnk")
    if os.path.exists(exact_match):
        return exact_match
    
    # If not found, search through all files
    for filename in os.listdir(nk_dir):
        if filename.lower().endswith('.lnk') and material.lower() in filename.lower():
            print(f"Warning: Using {filename} which may not match the requested temperature ({temp}K).")
            return os.path.join(nk_dir, filename)
    
    return None

def run_optool(material, grain_size, temp=None, nk_dir="nk_optool", output_dir="optool_output"):
    """
    Run Optool to generate opacity files for the specified material, grain size, and temperature.
    
    Parameters:
    -----------
    material : str
        Material identifier (e.g., 'E40R')
    grain_size : float
        Grain size in microns
    temp : int, optional
        Temperature in K (for file naming and .lnk file selection)
    nk_dir : str
        Directory containing the .lnk files
    output_dir : str
        Directory to save the output files
    
    Returns:
    --------
    str or None
        Path to the generated dustkapscatmat file if successful, None otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the appropriate .lnk file with temperature consideration
    nk_file = find_nk_file(material, temp, nk_dir)
    if not nk_file:
        print(f"Error: No .lnk file found for material {material} with temperature {temp}K in {nk_dir}")
        return None
    
    # Extract the base material name without any temperature indicators
    # This prevents duplicate temperature indicators in output directory names
    # Strip any existing temperature indicators (like _10K, _100K, etc.)
    base_material = re.sub(r'_\d+K$', '', material)
    
    # Prepare filename components
    size_str = f"a{grain_size}"
    temp_str = f"{temp}K" if temp else ""
    
    # Create a unique output directory name (without .dat extension)
    output_dir_name = f"{base_material}_{temp_str}_{size_str}"
    
    output_path = os.path.join(output_dir, output_dir_name)
    
    # Construct Optool command - do not include -T parameter as it's not supported
    cmd = [
        "optool",
        nk_file,  # material file
        "-s",     # generate scattering matrix
        "-radmc", # output in RADMC-3D format
        "-a", str(grain_size),  # grain size in microns
        "-o", output_path       # output file
    ]
    
    try:
        print(f"Running Optool for {material} with grain size {grain_size}μm" + 
              (f" (using {os.path.basename(nk_file)} for {temp}K)" if temp else "") + "...")
        print(f"Output will be in directory: {output_path}")
        
        # Run Optool
        result = subprocess.run(cmd, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               universal_newlines=True, 
                               check=True)
        
        # Check if output directory was created
        output_dir_path = output_path  # No longer need to add file extension
        if not os.path.exists(output_dir_path):
            print(f"Warning: Expected output directory {output_dir_path} not found.")
            print(f"Optool output: {result.stdout}")
            print(f"Optool errors: {result.stderr}")
            return None
        
        # Find and rename the dustkapscatmat.inp file
        source_file = os.path.join(output_dir_path, "dustkapscatmat.inp")
        
        if os.path.exists(source_file):
            # Create new filename based on the model parameters
            if temp:
                new_filename = f"dustkapscatmat_{base_material}_{temp}K_{size_str}.inp"
            else:
                new_filename = f"dustkapscatmat_{base_material}_{size_str}.inp"
            
            # Copy to output_dir
            destination_file = os.path.join(output_dir, new_filename)
            shutil.copy2(source_file, destination_file)
            
            print(f"Created opacity file: {new_filename}")
            print(f"Saved to: {os.path.abspath(destination_file)}")
            
            return destination_file
        else:
            print(f"Warning: dustkapscatmat.inp not found in {output_dir_path}")
            return None
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Optool for {material}: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def generate_temperature_dependent_opacities(material, grain_size, 
                                           temperatures=TEMPERATURE_VALUES,
                                           nk_dir="nk_optool", 
                                           output_dir="optool_output",
                                           target_dir=None):
    """
    Generate opacity files for a range of temperatures for the specified material and grain size.
    
    Parameters:
    -----------
    material : str
        Material identifier (e.g., 'E40R')
    grain_size : float
        Grain size in microns
    temperatures : list
        List of temperatures in K
    nk_dir : str
        Directory containing the .lnk files
    output_dir : str
        Directory to save the output files
    target_dir : str, optional
        Directory to copy the final files to (e.g., RADMC-3D model directory)
    
    Returns:
    --------
    list
        List of paths to the generated dustkapscatmat files
    """
    # Check if Optool is installed
    if not check_optool_installed():
        print("Error: Optool not found. Please make sure it's installed and in your PATH.")
        return []
    
    # If target_dir is specified, place optool_output inside it
    if target_dir:
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        # Set output_dir to be inside target_dir
        output_dir = os.path.join(target_dir, "optool_output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the main directory (where main_radmc3d.py is located)
    main_dir = os.getcwd()
    
    # Generate opacity files for each temperature
    opacity_files = []
    
    for temp in temperatures:
        opacity_file = run_optool(material, grain_size, temp, nk_dir, output_dir)
        
        if opacity_file:
            opacity_files.append(opacity_file)
            
            # Always copy to target_dir if specified (even if output_dir is inside target_dir)
            if target_dir:
                # Get the filename only
                filename = os.path.basename(opacity_file)
                # Create path in target directory (not in optool_output subdirectory)
                target_file = os.path.join(target_dir, filename)
                # Copy the file
                shutil.copy2(opacity_file, target_file)
                print(f"Copied {filename} to {target_dir}")
        else:
            print(f"Failed to generate opacity file for {material} at {temp}K")
    
    if len(opacity_files) == len(temperatures):
        print(f"Successfully generated all {len(temperatures)} temperature-dependent opacity files for {material} with grain size {grain_size}μm")
        if target_dir:
            print(f"Files are available in the target directory: {target_dir}")
        else:
            print(f"Files are available in: {output_dir}")
    else:
        print(f"Warning: Only generated {len(opacity_files)} of {len(temperatures)} opacity files")
    
    return opacity_files

def main(args=None):
    """
    Main function to run the script.
    
    Parameters:
    -----------
    args : list, optional
        Command line arguments (for testing purposes)
    
    Returns:
    --------
    int
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(description='Generate dust opacity files using Optool for RADMC-3D')
    
    parser.add_argument('--material', type=str, default='E40R',
                       help='Dust material identifier (e.g., E40R)')
    parser.add_argument('--grain-size', type=float, default=0.3,
                       help='Grain size in microns')
    parser.add_argument('--nk-dir', type=str, default='nk_optool',
                       help='Directory containing the .lnk files')
    parser.add_argument('--output-dir', type=str, default='radmc3d_model',
                       help='Directory to save the final output files (e.g., RADMC-3D model directory)')
    parser.add_argument('--target-dir', type=str, default=None,
                       help='Alias for --output-dir, maintained for backward compatibility')
    parser.add_argument('--temperatures', type=str, default='10,100,200,300',
                       help='Comma-separated list of temperatures in K')
    parser.add_argument('--no-temp-dependent', action='store_true',
                       help='Generate a single opacity file instead of temperature-dependent ones')
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    # For backward compatibility, use target-dir if provided, otherwise use output-dir
    target_dir = args.target_dir if args.target_dir else args.output_dir
    
    # Parse temperatures
    if args.no_temp_dependent:
        temperatures = []
    else:
        temperatures = [int(t) for t in args.temperatures.split(',')]
    
    # Check if material is valid
    if args.material not in DENSITIES:
        print(f"Warning: Material '{args.material}' not found in density database. Available materials:")
        for mat in DENSITIES:
            print(f"  - {mat} (density: {DENSITIES[mat]} g/cm³)")
        print("Continuing with the specified material, but density information may be inaccurate.")
    
    # Check for nk_dir
    if not os.path.exists(args.nk_dir):
        print(f"Error: nk directory '{args.nk_dir}' not found.")
        print(f"Current directory: {os.getcwd()}")
        print("Please make sure the directory exists and contains the .lnk files.")
        return 1
    
    # Generate opacity files
    if args.no_temp_dependent:
        # Make sure target_dir exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Use output_dir/optool_output for intermediate files
        optool_output = os.path.join(target_dir, "optool_output")
        os.makedirs(optool_output, exist_ok=True)
        
        # Generate a single opacity file (no temperature)
        opacity_file = run_optool(args.material, args.grain_size, 
                                 nk_dir=args.nk_dir, output_dir=optool_output)
        
        if opacity_file:
            # File is already in optool_output, copy to target_dir
            target_file = os.path.join(target_dir, os.path.basename(opacity_file))
            if not os.path.exists(target_file) or not os.path.samefile(opacity_file, target_file):
                shutil.copy2(opacity_file, target_file)
                print(f"Copied {os.path.basename(opacity_file)} to {target_dir}")
            return 0
        else:
            return 1
    else:
        # Generate temperature-dependent opacity files, specifying target_dir
        opacity_files = generate_temperature_dependent_opacities(
            args.material, args.grain_size, temperatures,
            args.nk_dir, "optool_output", target_dir
        )
        
        return 0 if len(opacity_files) == len(temperatures) else 1

if __name__ == "__main__":
    sys.exit(main()) 