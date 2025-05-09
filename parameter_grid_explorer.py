#!/usr/bin/env python3
"""
Parameter Grid Explorer for RADMC-3D Water Fountain Models

This script runs main_radmc3d.py with combinations of parameters specified in a JSON file.
It allows exploring parameters in ranges (linear or logarithmic) while keeping others fixed.
"""

import os
import json
import numpy as np
import subprocess
import itertools
import time
import argparse
import multiprocessing
from functools import partial
from datetime import datetime
from pathlib import Path


def generate_parameter_values(param_config):
    """
    Generate parameter values from the configuration.
    
    Parameters:
    -----------
    param_config : dict
        Parameter configuration with ranges and single values
        
    Returns:
    --------
    dict
        Dictionary with parameter names as keys and lists of values as values
    """
    param_values = {}
    
    # Process ranges
    for param, config in param_config.get('ranges', {}).items():
        min_val = config['min']
        max_val = config['max']
        points = config['points']
        scale = config.get('scale', 'linear')
        
        if scale == 'log':
            if min_val <= 0 or max_val <= 0:
                raise ValueError(f"Log scale requires positive values for parameter {param}")
            values = np.logspace(np.log10(min_val), np.log10(max_val), points)
        else:  # linear
            values = np.linspace(min_val, max_val, points)
            
        param_values[param] = values.tolist()
    
    # Process single values
    for param, value in param_config.get('single_values', {}).items():
        param_values[param] = [value]
    
    return param_values


def create_param_combinations(param_values):
    """
    Create all combinations of parameter values.
    
    Parameters:
    -----------
    param_values : dict
        Dictionary with parameter names and their possible values
        
    Returns:
    --------
    list
        List of dictionaries, each representing a parameter combination
    tuple
        Ordered parameter names (to keep track of indices)
    """
    param_names = list(param_values.keys())
    value_lists = [param_values[name] for name in param_names]
    
    combinations = []
    for values in itertools.product(*value_lists):
        combination = {name: value for name, value in zip(param_names, values)}
        combinations.append(combination)
    
    return combinations, param_names


def create_independent_param_combinations(param_values, single_values=None):
    """
    Create parameter combinations where only one parameter varies at a time.
    
    Parameters:
    -----------
    param_values : dict
        Dictionary with parameter names and their possible values
    single_values : dict, optional
        Dictionary with fixed parameter values for the base configuration
        
    Returns:
    --------
    list
        List of dictionaries, each representing a parameter combination
    tuple
        Ordered parameter names
    dict
        Mapping from parameter name to its index in node directory naming
    """
    if single_values is None:
        single_values = {}
        
    param_names = list(param_values.keys())
    
    # Create a base configuration with the first value of each parameter
    base_config = {name: param_values[name][0] for name in param_names}
    
    # Update with any fixed single values provided
    base_config.update(single_values)
    
    # Start with the base configuration
    combinations = [base_config.copy()]
    
    # Map each RANGE parameter to its exploration index (for node directory naming)
    # Only include parameters that are in ranges, not single values
    param_indices = {name: idx for idx, name in enumerate(param_names)}
    
    # For each parameter, create variations while keeping others at base values
    for param_idx, param_name in enumerate(param_names):
        values = param_values[param_name]
        
        # Skip the first value as it's already in the base configuration
        for val_idx, value in enumerate(values[1:], 1):
            # Create a new configuration based on the base
            config = base_config.copy()
            # Change only this parameter
            config[param_name] = value
            combinations.append(config)
    
    return combinations, param_names, param_indices


def get_param_indices(params, param_values, param_names):
    """
    Get the indices of parameter values in their respective ranges.
    
    Parameters:
    -----------
    params : dict
        Parameter values for the current combination
    param_values : dict
        Dictionary with parameter names and their possible values
    param_names : list
        Ordered list of parameter names
    
    Returns:
    --------
    list
        Indices of parameter values in their ranges
    """
    indices = []
    for name in param_names:
        value = params[name]
        values_list = param_values[name]
        # Find the index of this value in the list
        # For floats, we need to find the closest match due to potential precision issues
        if isinstance(value, float):
            idx = min(range(len(values_list)), key=lambda i: abs(values_list[i] - value))
        else:
            idx = values_list.index(value)
        indices.append(idx)
    
    return indices


def get_node_name_independent(params, base_config, param_indices, param_values, range_param_names):
    """
    Generate a node name for independent parameter exploration mode.
    
    Parameters:
    -----------
    params : dict
        Parameter values for the current combination
    base_config : dict
        Base configuration (reference point)
    param_indices : dict
        Mapping from parameter name to its index in node naming
    param_values : dict
        Dictionary with parameter names and their possible values
    range_param_names : list
        List of parameter names that are in ranges
        
    Returns:
    --------
    str
        Node directory name
    list
        Indices for the modified parameters
    """
    # Use only range parameters for node naming
    range_param_indices = {name: idx for idx, name in enumerate(range_param_names)}
    
    # If this is the base configuration, use a special name with all zeros
    if params == base_config:
        return "node_" + "_".join(["0"] * len(range_param_names)), [0] * len(range_param_names)
    
    # Find which parameter was modified from the base
    modified_param = None
    modified_value = None
    for name, value in params.items():
        if name in base_config and value != base_config[name] and name in range_param_names:
            modified_param = name
            modified_value = value
            break
    
    if modified_param is None:
        # Should never happen, but just in case
        return "node_unknown", []
    
    # Find the index of the parameter and its value
    param_idx = range_param_indices[modified_param]
    values_list = param_values[modified_param]
    
    # Find value index
    if isinstance(modified_value, float):
        val_idx = min(range(len(values_list)), key=lambda i: abs(values_list[i] - modified_value))
    else:
        val_idx = values_list.index(modified_value)
    
    # Create indices list where all are 0 except the modified parameter
    indices = [0] * len(range_param_names)
    indices[param_idx] = val_idx
    
    # Format as node_0_0_1_0 etc.
    node_name = f"node_{'_'.join(map(str, indices))}"
    
    return node_name, indices


def run_radmc3d_with_params(run_info, base_output_dir, param_values, param_names, range_param_names, 
                          independent=False, base_config=None, param_indices=None):
    """
    Run main_radmc3d.py with the specified parameters.
    
    Parameters:
    -----------
    run_info : tuple
        (run_index, params) - Index of this run and parameter values
    base_output_dir : str
        Base directory for outputs
    param_values : dict
        Dictionary with parameter names and their possible values
    param_names : list
        Ordered list of parameter names
    range_param_names : list
        List of parameter names that are in ranges
    independent : bool
        Whether to use independent parameter exploration mode
    base_config : dict
        Base configuration for independent mode
    param_indices : dict
        Mapping from parameter name to its index in node naming for independent mode
        
    Returns:
    --------
    dict
        Result dictionary with run information
    """
    run_index, params = run_info
    
    if independent:
        # Independent mode node naming
        node_name, node_indices = get_node_name_independent(params, base_config, param_indices, param_values, range_param_names)
        
        # For metadata, determine which parameter was varied (if any)
        varied_param = None
        param_value = None
        if params != base_config:
            for name, value in params.items():
                if name in base_config and value != base_config[name]:
                    varied_param = name
                    param_value = value
                    break
    else:
        # Original grid mode node naming
        # Get indices of parameter values in their ranges
        indices = get_param_indices(params, param_values, param_names)
        
        # Create the node directory name with indices for range parameters only
        # For range parameters, track their indices
        range_indices = []
        for name in range_param_names:
            if name in params:
                idx = param_names.index(name)
                range_indices.append(indices[idx])
        
        # Create node name in the format node_i1_i2_i3...
        node_name = f"node_{'_'.join(map(str, range_indices))}"
        node_indices = range_indices
    
    # Create a unique output directory for this run
    run_dir = os.path.join(base_output_dir, node_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create a param summary string for the directory name
    param_summary = []
    for key, value in params.items():
        if isinstance(value, float):
            # Format floats with appropriate precision
            if abs(value) < 0.001 or abs(value) >= 1000:
                param_summary.append(f"{key}={value:.2e}")
            else:
                param_summary.append(f"{key}={value:.4f}")
        else:
            param_summary.append(f"{key}={value}")
    
    # Save parameter values to a JSON file in the run directory
    param_file = os.path.join(run_dir, "params.json")
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    # Build command using python3 instead of python
    cmd = ["python3", "main_radmc3d.py", f"--output_dir={run_dir}"]
    
    # Add all parameters
    for param, value in params.items():
        # Handle boolean parameters specifically
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{param}")
        else:
            cmd.append(f"--{param}={value}")
    
    # Print and run command
    print(f"\nRunning process {run_index + 1}: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return_code = process.returncode
    except Exception as e:
        print(f"Error executing process {run_index + 1}: {e}")
        return_code = -1
        stdout = str(e).encode()
        stderr = b""
    
    elapsed_time = time.time() - start_time
    
    # Save stdout and stderr to files
    with open(os.path.join(run_dir, "stdout.log"), 'wb') as f:
        f.write(stdout)
    
    with open(os.path.join(run_dir, "stderr.log"), 'wb') as f:
        f.write(stderr)
    
    # Print status
    if return_code == 0:
        print(f"✓ Process {run_index + 1} completed successfully in {elapsed_time:.1f} seconds")
    else:
        print(f"✗ Process {run_index + 1} failed with return code {return_code}")
    
    # For independent mode, create result with exploration metadata
    if independent and params != base_config:
        result = {
            "run_index": run_index,
            "node_dir": node_name,
            "node_indices": node_indices,
            "varied_parameter": varied_param,
            "parameter_value": param_value,
            "status": "success" if return_code == 0 else "failed",
            "return_code": return_code,
            "execution_time": elapsed_time
        }
    else:
        # Get only the range parameters (varying parameters)
        range_params = {name: params[name] for name in range_param_names if name in params}
        
        # Create result dictionary (original format)
        result = {
            "run_index": run_index,
            "node_dir": node_name,
            "node_indices": node_indices,
            "parameters": range_params,
            "parameter_indices": {name: indices[param_names.index(name)] for name in range_param_names if name in params}
            if not independent else {},
            "status": "success" if return_code == 0 else "failed",
            "return_code": return_code,
            "execution_time": elapsed_time
        }
    
    return result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run RADMC-3D with parameter grid')
    parser.add_argument('config_file', help='JSON configuration file')
    parser.add_argument('--output_dir', default='parameter_grid_results', 
                      help='Base output directory (not used when --base_name is provided)')
    parser.add_argument('--base_name', type=str, default=None,
                      help='Custom name for the folder containing all node_* directories (default: timestamp in output_dir)')
    parser.add_argument('--start_index', type=int, default=0,
                      help='Start index for runs (to resume from a specific point)')
    parser.add_argument('--dry_run', action='store_true',
                      help='Only print commands without running them')
    parser.add_argument('--max_processes', type=int, default=None,
                      help='Maximum number of parallel processes (default: number of CPU cores)')
    parser.add_argument('--independent', action='store_true',
                      help='Explore parameters independently (one at a time) rather than as a full grid')
    args = parser.parse_args()
    
    # Determine the number of processes to use
    max_processes = args.max_processes if args.max_processes is not None else multiprocessing.cpu_count()
    
    # Read configuration file
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # Generate parameter values
    param_values = generate_parameter_values(config)
    
    # Get names of parameters that are in ranges
    range_param_names = list(config.get('ranges', {}).keys())
    
    # Create parameter combinations based on mode
    if args.independent:
        # Independent mode - vary one parameter at a time
        single_values = config.get('single_values', {})
        combinations, param_names, param_indices = create_independent_param_combinations(param_values, single_values)
        base_config = {name: param_values[name][0] for name in param_names}
        base_config.update(single_values)
        
        print(f"Independent parameter exploration with {len(combinations)} combinations")
        print(f"Base configuration: {base_config}")
        print(f"Range parameters: {range_param_names}")
    else:
        # Original full grid mode
        combinations, param_names = create_param_combinations(param_values)
        base_config = None
        param_indices = None
        
        print(f"Parameter grid exploration with {len(combinations)} combinations")
    
    num_combinations = len(combinations)
    
    # Create base output directory - if base_name is provided, use it directly as the output directory
    if args.base_name:
        base_output_dir = args.base_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.join(args.output_dir, timestamp)
        
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save configuration to output directory, including exploration mode
    config_with_mode = config.copy()
    config_with_mode['exploration_mode'] = 'independent' if args.independent else 'full_grid'
    with open(os.path.join(base_output_dir, "grid_config.json"), 'w') as f:
        json.dump(config_with_mode, f, indent=2)
    
    # Initialize combinations data with pending status
    combinations_data = []
    
    for i, params in enumerate(combinations):
        if args.independent:
            # Independent mode
            node_name, node_indices = get_node_name_independent(params, base_config, param_indices, param_values, range_param_names)
            
            # For metadata, determine which parameter was varied (if any)
            varied_param = None
            param_value = None
            if params != base_config:
                for name, value in params.items():
                    if name in base_config and value != base_config[name]:
                        varied_param = name
                        param_value = value
                        break
            
            # Create entry for independent mode
            if params == base_config:
                entry = {
                    "run_index": i,
                    "node_dir": node_name,
                    "node_indices": node_indices,
                    "is_base_config": True,
                    "status": "pending"
                }
            else:
                entry = {
                    "run_index": i,
                    "node_dir": node_name,
                    "node_indices": node_indices,
                    "varied_parameter": varied_param,
                    "parameter_value": param_value,
                    "status": "pending"
                }
        else:
            # Original grid mode
            indices = get_param_indices(params, param_values, param_names)
            
            range_indices = []
            for name in range_param_names:
                if name in params:
                    idx = param_names.index(name)
                    range_indices.append(indices[idx])
                    
            # Create the node name
            node_name = f"node_{'_'.join(map(str, range_indices))}"
            
            # Create parameter-to-index mapping for range parameters only
            param_indices_map = {name: indices[param_names.index(name)] for name in range_param_names if name in params}
            
            # Get only the range parameters (varying parameters)
            range_params = {name: params[name] for name in range_param_names if name in params}
            
            # Create entry with pending status (original format)
            entry = {
                "run_index": i,
                "node_dir": node_name,
                "node_indices": range_indices,
                "parameters": range_params,
                "parameter_indices": param_indices_map,
                "status": "pending"
            }
            
        combinations_data.append(entry)
    
    # Save initial combinations to output directory
    combinations_file = os.path.join(base_output_dir, "all_combinations.json")
    with open(combinations_file, 'w') as f:
        enhanced_json = {
            "exploration_mode": "independent" if args.independent else "full_grid",
            "range_parameter_names": range_param_names,
            "combinations": combinations_data,
            "base_config": base_config if args.independent else None,
            "execution_summary": {
                "total_runs": num_combinations,
                "successful_runs": 0,
                "failed_runs": 0,
                "total_time": 0,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "num_processes": max_processes
            }
        }
        json.dump(enhanced_json, f, indent=2)
    
    print(f"Output directory: {base_output_dir}")
    print(f"Using {max_processes} parallel processes")
    
    if args.dry_run:
        print("\nDRY RUN - Commands would be:")
        for i, entry in enumerate(combinations_data):
            node_name = entry["node_dir"]
            # Need the full parameter set for running commands, not just range params
            params = combinations[i]
            
            # Use python3 in the dry run output as well
            cmd = ["python3", "main_radmc3d.py", f"--output_dir={node_name}"]
            for param, value in params.items():
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{param}")
                else:
                    cmd.append(f"--{param}={value}")
            print(f"Run {i+1}/{num_combinations}: {' '.join(cmd)}")
        return
    
    # Run RADMC-3D with each parameter combination in parallel
    total_start_time = time.time()
    
    # Filter combinations based on start_index
    run_combinations = [(i, params) for i, params in enumerate(combinations) if i >= args.start_index]
    
    # Create partial function with fixed arguments
    run_func = partial(
        run_radmc3d_with_params,
        base_output_dir=base_output_dir,
        param_values=param_values,
        param_names=param_names,
        range_param_names=range_param_names,
        independent=args.independent,
        base_config=base_config,
        param_indices=param_indices
    )
    
    # Run in parallel
    results = []
    with multiprocessing.Pool(processes=max_processes) as pool:
        # Use imap_unordered to get results as they complete
        for result in pool.imap_unordered(run_func, run_combinations):
            results.append(result)
    
    # Total execution time
    total_time = time.time() - total_start_time
    
    # Count successful and failed runs
    successful_runs = sum(1 for r in results if r['status'] == 'success')
    failed_runs = sum(1 for r in results if r['status'] == 'failed')
    
    # Update the combinations file with results
    with open(combinations_file, 'r') as f:
        all_data = json.load(f)
    
    # Update combinations with results
    for result in results:
        idx = result['run_index']
        all_data['combinations'][idx].update({
            'status': result['status'],
            'return_code': result.get('return_code'),
            'execution_time': result.get('execution_time')
        })
    
    # Update execution summary
    all_data['execution_summary'].update({
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'total_time': total_time,
        'end_time': datetime.now().isoformat()
    })
    
    # Save updated combinations with results
    with open(combinations_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # Print summary
    print(f"\nParameter exploration completed ({all_data['exploration_mode']} mode):")
    print(f"Total runs: {len(run_combinations)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Parallel processes: {max_processes}")
    
    # Create a simple summary file with just the high-level results
    summary_file = os.path.join(base_output_dir, "summary.txt") 
    with open(summary_file, 'w') as f:
        f.write(f"Parameter exploration completed at {datetime.now().isoformat()}\n")
        f.write(f"Exploration mode: {all_data['exploration_mode']}\n")
        f.write(f"Total runs: {len(run_combinations)}\n")
        f.write(f"Successful: {successful_runs}\n")
        f.write(f"Failed: {failed_runs}\n")
        f.write(f"Total time: {total_time:.1f} seconds\n")
        f.write(f"Parallel processes: {max_processes}\n")


if __name__ == "__main__":
    main() 