import numpy as np
import os
import shutil

# Physical constants
au = 1.49598e13     # Astronomical Unit [cm]
Msun = 2e33         # Solar mass [g]
Rsun = 6.957e10     # Solar radius [cm]
G = 6.67259e-8      # Gravitational constant [cm³ g⁻¹ s⁻²]
sigma = 5.67051e-5  # Stefan-Boltzmann constant [erg cm⁻² s⁻¹ K⁻⁴]
c = 2.99792458e10   # Speed of light [cm/s]
h = 6.62607015e-27  # Planck constant [erg s]
kB = 1.380649e-16   # Boltzmann constant [erg K⁻¹]

def create_wavelength_grid(lam_min=0.1, lam_max=1000, nlam=100, water_fountain=False):
    """
    Create a wavelength grid file for RADMC-3D.
    
    Parameters:
    -----------
    lam_min : float
        Minimum wavelength in microns
    lam_max : float
        Maximum wavelength in microns
    nlam : int
        Number of wavelength points
    water_fountain : bool
        If True, uses the special water fountain wavelength grid
        
    Returns:
    --------
    wavelength : ndarray
        Array of wavelengths
    """
    if water_fountain:
        # Special wavelength grid for water fountain models
        lam1 = 0.1
        lam2 = 7.0
        lam3 = 25.0
        lam4 = 1.0e4
        n12 = 20
        n23 = 100
        n34 = 30
        lam12 = np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False)
        lam23 = np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False)
        lam34 = np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
        wavelength = np.concatenate([lam12, lam23, lam34])
    else:
        # Standard logarithmic wavelength grid
        wavelength = np.logspace(np.log10(lam_min), np.log10(lam_max), nlam)
    
    # Write the wavelength grid to file
    with open('wavelength_micron.inp', 'w') as f:
        f.write(f'{len(wavelength)}\n')
        for lam in wavelength:
            f.write(f'{lam:13.6e}\n')
    
    print(f"Created wavelength_micron.inp with {len(wavelength)} points")
    return wavelength

def create_stars(radius=1.0, temperature=5780.0, pos=[0.0, 0.0, 0.0], spectrum_file=None, water_fountain=False):
    """
    Create a stars.inp file for a single star.
    
    Parameters:
    -----------
    radius : float
        Stellar radius in solar radii
    temperature : float
        Stellar effective temperature in Kelvin
    pos : list
        Position of the star [x, y, z] in AU
    spectrum_file : str
        Optional file containing custom spectrum (overrides temperature)
    water_fountain : bool
        If True, uses the special water fountain star format
        
    Returns:
    --------
    None
        Writes the stars.inp file
    """
    # Read the wavelength grid to ensure the stellar spectrum matches
    try:
        with open('wavelength_micron.inp', 'r') as f:
            lines = f.readlines()
            nlam = int(lines[0].strip())
            wavelengths = np.array([float(line.strip()) for line in lines[1:]])
    except FileNotFoundError:
        print("Warning: wavelength_micron.inp not found. Creating default wavelength grid.")
        wavelengths = create_wavelength_grid(water_fountain=water_fountain)
        nlam = len(wavelengths)
    
    if water_fountain:
        # Format used in the water fountain model
        with open('stars.inp', 'w') as f:
            f.write('2\n')  # Format number
            f.write(f'1 {nlam}\n')  # Number of stars and wavelengths
            f.write(f'{radius*Rsun:13.6e} {Msun:13.6e} {pos[0]:13.6e} {pos[1]:13.6e} {pos[2]:13.6e}\n\n')
            
            # Write wavelength grid
            for lam in wavelengths:
                f.write(f'{lam:13.6e}\n')
            
            # Negative temperature indicates blackbody
            f.write(f'\n{-temperature:13.6e}\n')
    else:
        # Calculate blackbody spectrum for each wavelength
        # Constants for Planck function
        h_local = 6.62607015e-34  # Planck constant in J⋅s
        c_local = 2.99792458e8    # Speed of light in m/s
        k_local = 1.380649e-23    # Boltzmann constant in J/K
        
        # Convert wavelengths from microns to meters
        lam_m = wavelengths * 1e-6
        
        # Calculate Planck function B_λ(T) in W/m²/μm/sr
        exp_term = np.exp(h_local * c_local / (lam_m * k_local * temperature)) - 1.0
        B_lambda = (2.0 * h_local * c_local**2) / (lam_m**5 * exp_term)
        
        # Convert to erg/s/cm²/Hz/sr (RADMC-3D units)
        B_nu = (lam_m**2 / c_local) * B_lambda * 1e-3  # Convert W to erg/s
        
        # Write the stars.inp file with format 1 (includes spectrum)
        with open('stars.inp', 'w') as f:
            f.write('1\n')  # Format number
            f.write('1\n')  # Number of stars
            f.write('2\n')  # Star shape: 2 = sphere
            f.write(f'{radius} {pos[0]} {pos[1]} {pos[2]}\n')  # Radius and position
            f.write(f'{nlam}\n')  # Number of wavelength points
            
            # Write the spectrum
            for flux in B_nu:
                f.write(f'{flux}\n')
    
    print(f"Created stars.inp with star at position {pos}, radius={radius}R☉, T={temperature}K")

def torus_wind(r, theta, params):
    """
    Calculate the density of a torus wind model.
    
    Parameters:
    -----------
    r : ndarray
        Radius grid
    theta : ndarray
        Theta grid (colatitude)
    params : list
        [rhomin, Rmin, Rtorus, A, B, C, D, E, F, oangle]
        
    Returns:
    --------
    ndarray
        Density distribution
    """
    rhomin, Rmin, Rtorus, A, B, C, D, E, F, oangle = params
    
    part1 = (r/Rmin)
    exp1 = -B*(1+C*(np.sin(theta)**F)*(np.exp(-(r/Rtorus)**D))/(np.exp(-(Rmin/Rtorus)**D)))
    part2 = (1+A*((1-np.cos(theta))**F)*(np.exp(-(r/Rtorus)**E))/(np.exp(-(Rmin/Rtorus)**E)))
    torusdens = (rhomin*(part1**exp1) * part2)
    torusdens = np.where(r<Rmin, 0., torusdens)
    torusdens = np.where(np.abs(theta)<oangle, 0.001*torusdens, torusdens)
    
    return torusdens

def outflow_lobe(r, theta, params):
    """
    Calculate the density of an outflow lobe.
    
    Parameters:
    -----------
    r : ndarray
        Radius grid
    theta : ndarray
        Theta grid (colatitude)
    params : list
        [lobe_dens, Rlobe, lobe_angle, width]
        
    Returns:
    --------
    ndarray
        Density distribution
    """
    lobe_dens, Rlobe, lobe_angle, width = params
    
    scale = (r/Rlobe)**2 + (theta/lobe_angle)**2
    lobedens = lobe_dens * np.exp(-(1-scale)**2/(2*width))
    
    return lobedens

def intmass(r, theta, func, params):
    """
    Calculate the integrated mass of a density distribution.
    
    Parameters:
    -----------
    r : ndarray
        Radius grid
    theta : ndarray
        Theta grid (colatitude)
    func : function
        Density function
    params : list
        Parameters for the density function
        
    Returns:
    --------
    float
        Integrated mass in solar masses
    """
    dR = (r.max()-r.min())/(len(r)-1)
    dtheta = (theta.max()-theta.min())/(len(theta)-1)
    
    zz = func(r, theta, params)
    mass = 2.*np.pi*np.sum(r*r*np.sin(theta)*dR*dtheta*zz)
    mass *= (1/2.e33)*(2.)  # Convert to solar masses
    
    return mass

def create_water_fountain_grid(rin=100*au, rout=5000*au, nr=1000, ntheta=150, nphi=1):
    """
    Create a spherical grid specifically for water fountain models.
    
    Parameters:
    -----------
    rin : float
        Inner radius in cm
    rout : float
        Outer radius in cm
    nr : int
        Number of radial cells
    ntheta : int
        Number of theta cells (one-sided, so 0 to pi/2)
    nphi : int
        Number of phi cells (usually 1 for 2D)
        
    Returns:
    --------
    dict
        Grid information dictionary
    """
    # Create grid points
    ri = rin * (rout/rin)**(np.linspace(0., nr, nr+1)/(nr-1.0))  # Cell walls
    thetai = np.pi/2.0 * np.linspace(0., 1., ntheta+1)  # Cell walls
    phii = np.array([0., 2*np.pi])  # Cell walls
    
    # Cell centers
    rc = 0.5 * (ri[:-1] + ri[1:])
    thetac = 0.5 * (thetai[:-1] + thetai[1:])
    
    # Create meshgrid for calculations
    rr, tt = np.meshgrid(rc, thetac, indexing='ij')
    
    # Write the grid file
    with open('amr_grid.inp', 'w') as f:
        f.write('1\n')                       # iformat
        f.write('0\n')                       # AMR grid style (0=regular grid, no AMR)
        f.write('100\n')                     # Coordinate system (100=spherical)
        f.write('0\n')                       # gridinfo
        f.write('1 1 0\n')                   # Include r,theta,phi coordinate
        f.write(f'{nr} {ntheta} {nphi}\n')   # Size of grid
        
        for value in ri:
            f.write(f'{value:13.6e}\n')      # R coordinates (cell walls)
        for value in thetai:
            f.write(f'{value:13.6e}\n')      # Theta coordinates (cell walls)
        for value in phii:
            f.write(f'{value:13.6e}\n')      # Phi coordinates (cell walls)
    
    print(f"Created amr_grid.inp with {nr}x{ntheta}x{nphi} spherical grid from {rin/au:.1f} to {rout/au:.1f} AU")
    
    # Create a grid info dictionary
    grid_info = {
        'type': 'spherical',
        'nr': nr,
        'ntheta': ntheta,
        'nphi': nphi,
        'rmin': rin,
        'rmax': rout,
        'thetamin': 0.0,
        'thetamax': np.pi/2.0,
        'phimin': 0.0,
        'phimax': 2*np.pi,
        'ri': ri,
        'thetai': thetai,
        'phii': phii,
        'rc': rc,
        'thetac': thetac,
        'rr': rr,
        'tt': tt
    }
    
    return grid_info

def create_water_fountain_density(grid_info, torus_params, lobe_params):
    """
    Create the density structure for a water fountain model.
    
    Parameters:
    -----------
    grid_info : dict
        Grid information from create_water_fountain_grid
    torus_params : list
        Parameters for the torus: [rhomin, Rmin, Rtorus, A, B, C, D, E, F, oangle]
    lobe_params : list
        Parameters for the outflow lobes: [lobe_dens, Rlobe, lobe_angle, width]
        
    Returns:
    --------
    ndarray
        Density array
    """
    # Extract grid
    rr = grid_info['rr']
    tt = grid_info['tt']
    nr = grid_info['nr']
    ntheta = grid_info['ntheta']
    
    # Initialize density array
    rhod = np.zeros((nr, ntheta))
    
    # Calculate torus and lobe densities
    for i in range(nr):
        for j in range(ntheta):
            rhod[i, j] = torus_wind(rr[i, j], tt[i, j], torus_params) + \
                         outflow_lobe(rr[i, j], tt[i, j], lobe_params)
    
    # Write the density file
    with open('dust_density.inp', 'w') as f:
        f.write('1\n')                       # Format number
        f.write(f'{nr*ntheta*1}\n')          # Nr of cells
        f.write('1\n')                       # Nr of dust species
        data = rhod.ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
    
    print(f"Created dust_density.inp with water fountain model (torus + outflow lobes)")
    
    return rhod

def copy_dust_opacity(dust_file='dustkapscatmat_E40R_300K_a0.3.inp', 
                     new_name='dust_kapscatmat_1.inp'):
    """
    Copy the dust opacity file to the current directory with a new name.
    
    Parameters:
    -----------
    dust_file : str
        Original dust opacity file name
    new_name : str
        New name for the dust opacity file
        
    Returns:
    --------
    None
    """
    try:
        shutil.copy(dust_file, new_name)
        print(f"Copied {dust_file} to {new_name}")
    except FileNotFoundError:
        print(f"ERROR: Dust file {dust_file} not found.")
        print(f"Current directory: {os.getcwd()}")
        print("Files in current directory:")
        for file in os.listdir():
            print(f"  {file}")
        raise

def create_single_dust_opacity(extension='E40R_300K_a0.3'):
    """
    Create a dustopac.inp file for RADMC-3D with a single dust species.
    
    Parameters:
    -----------
    extension : str
        Extension of the dust opacity file (e.g., 'E40R_300K_a0.3')
        
    Returns:
    --------
    None
        Writes the dustopac.inp file
    """
    with open('dustopac.inp', 'w') as f:
        f.write('2\n')
        f.write('1\n')
        f.write('============================================================================\n')
        f.write('10\n')
        f.write('0\n')
        f.write(f'{extension}\n')
        f.write('----------------------------------------------------------------------------\n')
    
    print(f"Created dustopac.inp with single dust species: {extension}")

def create_dustopac(nspecies=1, scattering_mode=1, water_fountain=False, temp_dependent=False):
    """
    Create a dustopac.inp file for RADMC-3D.
    
    Parameters:
    -----------
    nspecies : int
        Number of dust species
    scattering_mode : int
        Scattering mode (0=no scattering, 1=isotropic, 2=anisotropic)
    water_fountain : bool
        If True, uses the special water fountain dust opacity format
    temp_dependent : bool
        If True, create temperature-dependent dust opacities for 4 temperature ranges
        
    Returns:
    --------
    None
        Writes the dustopac.inp file
    """
    if temp_dependent:
        # Temperature-dependent dust opacities for 4 temperature ranges
        with open('dustopac.inp', 'w') as f:
            # Using exact format from user example
            f.write('2\n')
            f.write('4\n')
            
            # Only the first species has the ============= line
            f.write('============================================================================\n')
            
            # 10K opacities (for T < 50K)
            f.write('10\n')
            f.write('0\n')
            f.write('E40R_10K_a0.3\n')
            f.write('----------------------------------------------------------------------------\n')
            
            # 100K opacities (for 50K <= T < 150K) - no ============= separator
            f.write('10\n')
            f.write('0\n')
            f.write('E40R_100K_a0.3\n')
            f.write('----------------------------------------------------------------------------\n')
            
            # 200K opacities (for 150K <= T < 250K) - no ============= separator
            f.write('10\n')
            f.write('0\n')
            f.write('E40R_200K_a0.3\n')
            f.write('----------------------------------------------------------------------------\n')
            
            # 300K opacities (for T >= 250K) - no ============= separator
            f.write('10\n')
            f.write('0\n')
            f.write('E40R_300K_a0.3\n')
            f.write('----------------------------------------------------------------------------\n')
    elif water_fountain:
        # Water fountain specific dust opacity
        with open('dustopac.inp', 'w') as f:
            f.write('2\n')
            f.write('1\n')
            f.write('============================================================================\n')
            f.write('10\n')
            f.write('0\n')
            f.write('E40R_300K_a0.3\n')
            f.write('----------------------------------------------------------------------------\n')
    else:
        # Standard dust opacity setup
        with open('dustopac.inp', 'w') as f:
            f.write('2\n')  # Format number
            f.write(f'{nspecies}\n')  # Number of species
            
            # Only write the separator line once at the beginning
            f.write('============================================================================\n')
            
            # First species
            if scattering_mode > 0:
                f.write('10\n')  # Method for this dust species (10=full scattering matrix)
            else:
                f.write('1\n')   # Method for this dust species (1=opacity only)
            f.write('0\n')  # 0=thermal grain
            f.write('dust_kapscatmat_1\n')  # Extension of dust opacity file
            f.write('----------------------------------------------------------------------------\n')
            
            # Additional species without the separator line
            for i in range(1, nspecies):
                if scattering_mode > 0:
                    f.write('10\n')  # Method for this dust species (10=full scattering matrix)
                else:
                    f.write('1\n')   # Method for this dust species (1=opacity only)
                f.write('0\n')  # 0=thermal grain
                f.write(f'dust_kapscatmat_{i+1}\n')  # Extension of dust opacity file
                f.write('----------------------------------------------------------------------------\n')
    
    print(f"Created dustopac.inp with {'temperature-dependent' if temp_dependent else str(nspecies)} dust species, scattering mode {scattering_mode}")

def create_radmc3d_control(nphot_therm=1000000, scattering_mode_max=1, 
                          modified_random_walk=1, istar_sphere=1, water_fountain=False):
    """
    Create a radmc3d.inp control file.
    
    Parameters:
    -----------
    nphot_therm : int
        Number of photon packages for thermal Monte Carlo
    scattering_mode_max : int
        Maximum scattering mode
    modified_random_walk : int
        Enable/disable modified random walk
    istar_sphere : int
        Treat star as sphere (1) or point source (0)
    water_fountain : bool
        If True, use the water fountain specific settings
        
    Returns:
    --------
    None
        Writes the radmc3d.inp file
    """
    with open('radmc3d.inp', 'w') as f:
        f.write(f'nphot = {nphot_therm}\n')
        f.write(f'scattering_mode_max = {scattering_mode_max}\n')
        f.write('mc_scat_maxtauabs = 5.d0\n')
        
        if modified_random_walk:
            f.write('modified_random_walk = 1\n')
            
        if water_fountain:
            f.write(f'istar_sphere = {istar_sphere}\n')
    
    print(f"Created radmc3d.inp with nphot={nphot_therm}")

def setup_water_fountain_model(rin=100*au, rout=5000*au, nr=1000, ntheta=150,
                             stellar_radius=288*Rsun, stellar_mass=1*Msun, stellar_temp=3000,
                             Mdtorus=1.0/200, Mdlobe=0.1/200, 
                             Rtorus=1000*au, A=1, B=3, C=0, D=10, E=3, F=2,
                             Rlobe=2500*au, oangle=np.pi/10, width=0.005):
    """
    Set up a complete water fountain model with torus and outflow lobes.
    
    Parameters:
    -----------
    rin : float
        Inner radius in cm
    rout : float
        Outer radius in cm
    nr : int
        Number of radial grid cells
    ntheta : int
        Number of theta grid cells
    stellar_radius : float
        Radius of the central star in cm
    stellar_mass : float
        Mass of the central star in g
    stellar_temp : float
        Temperature of the central star in K
    Mdtorus : float
        Mass of the torus in solar masses
    Mdlobe : float
        Mass of the outflow lobes in solar masses
    Rtorus : float
        Characteristic radius of the torus in cm
    A, B, C, D, E, F : float
        Shape parameters for the torus
    Rlobe : float
        Characteristic radius of the lobes in cm
    oangle : float
        Opening angle of the outflow cavity in radians
    width : float
        Width parameter for the outflow lobes
        
    Returns:
    --------
    tuple
        (grid_info, density_array)
    """
    # Create initial parameter arrays
    torus_params = [1.0, rin, Rtorus, A, B, C, D, E, F, oangle]
    lobe_params = [1.0, Rlobe, oangle, width]
    
    # Create grid for mass calculations
    R_calc = np.linspace(rin, rout, 2000, endpoint=True)
    theta_calc = np.linspace(0, np.pi/2, 2000, endpoint=False)
    xx, yy = np.meshgrid(R_calc, theta_calc)
    
    # Calculate mass scaling factors
    scaletorus = intmass(xx, yy, torus_wind, torus_params)
    scalelobe = intmass(xx, yy, outflow_lobe, lobe_params)
    
    # Scale the density to match the desired masses
    torus_params[0] = (Mdtorus / scaletorus)
    lobe_params[0] = (Mdlobe / scalelobe)
    
    # Print the actual masses after scaling
    print(f'Mass torus: {intmass(xx, yy, torus_wind, torus_params):.6f} Msun')
    print(f'Mass lobe: {intmass(xx, yy, outflow_lobe, lobe_params):.6f} Msun')
    
    # Create the wavelength grid
    create_wavelength_grid(water_fountain=True)
    
    # Create the grid
    grid_info = create_water_fountain_grid(rin, rout, nr, ntheta)
    
    # Create the density structure
    rho = create_water_fountain_density(grid_info, torus_params, lobe_params)
    
    # Create the stellar source
    create_stars(radius=stellar_radius/Rsun, temperature=stellar_temp, 
                pos=[0.0, 0.0, 0.0], water_fountain=True)
    
    # Copy the dust opacity file
    copy_dust_opacity()
    
    # Create the dust opacity control file
    create_dustopac(water_fountain=True)
    
    # Create the RADMC-3D control file
    create_radmc3d_control(nphot_therm=1000000, istar_sphere=1, water_fountain=True)
    
    print(f"Water fountain model setup complete with masses: torus={Mdtorus:.6f} Msun, lobe={Mdlobe:.6f} Msun")
    
    return grid_info, rho

def create_amr_grid(grid_type='cartesian', **kwargs):
    """
    Create a grid file for RADMC-3D.
    
    Parameters:
    -----------
    grid_type : str
        Type of grid: 'cartesian', 'spherical', 'cylindrical'
    **kwargs : dict
        Grid parameters depending on grid_type
    
    For cartesian grid:
        nx, ny, nz : int
            Number of cells in each dimension
        xbound, ybound, zbound : list
            Boundaries of the grid in each dimension [min, max] in AU
            
    For spherical grid:
        nr, ntheta, nphi : int
            Number of cells in each dimension
        rbound : list
            Radial boundaries [rmin, rmax] in AU
        thetabound : list
            Polar angle boundaries [thetamin, thetamax] in radians (0 to pi)
        phibound : list
            Azimuthal angle boundaries [phimin, phimax] in radians (0 to 2pi)
        log_r : bool
            Whether to use logarithmic spacing for r (default: True)
            
    Returns:
    --------
    tuple
        Grid cell walls in each dimension
    """
    if grid_type.lower() == 'cartesian':
        return _create_cartesian_grid(**kwargs)
    elif grid_type.lower() == 'spherical':
        return _create_spherical_grid(**kwargs)
    elif grid_type.lower() == 'cylindrical':
        raise NotImplementedError("Cylindrical grid not yet implemented")
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")

def _create_cartesian_grid(nx=64, ny=64, nz=64, xbound=[-100, 100], ybound=[-100, 100], zbound=[-100, 100]):
    """Create a Cartesian grid for RADMC-3D."""
    # Create cell walls
    x = np.linspace(xbound[0], xbound[1], nx+1)
    y = np.linspace(ybound[0], ybound[1], ny+1)
    z = np.linspace(zbound[0], zbound[1], nz+1)
    
    # Write the AMR grid to file
    with open('amr_grid.inp', 'w') as f:
        f.write('1\n')                # iformat
        f.write('0\n')                # AMR grid style (0 = regular grid)
        f.write('0\n')                # Coordinate system (0 = cartesian)
        f.write('0\n')                # Gridinfo (0 = flash-style)
        f.write('1 1 1\n')            # Include x, y, z coordinates
        f.write(f'{nx} {ny} {nz}\n')  # Grid sizes
        for i in range(nx+1):
            f.write(f'{x[i]}\n')      # X-grid walls
        for i in range(ny+1):
            f.write(f'{y[i]}\n')      # Y-grid walls
        for i in range(nz+1):
            f.write(f'{z[i]}\n')      # Z-grid walls
    
    print(f"Created amr_grid.inp with Cartesian grid of {nx}x{ny}x{nz} cells")
    return x, y, z

def _create_spherical_grid(nr=100, ntheta=32, nphi=64, 
                         rbound=[1, 1000], thetabound=[0, np.pi], phibound=[0, 2*np.pi],
                         log_r=True, refinement_at_poles=True):
    """Create a Spherical grid for RADMC-3D."""
    # Create radial grid with optional logarithmic spacing
    if log_r:
        r = np.logspace(np.log10(rbound[0]), np.log10(rbound[1]), nr+1)
    else:
        r = np.linspace(rbound[0], rbound[1], nr+1)
    
    # Create theta grid with optional refinement near poles
    if refinement_at_poles:
        # Use finer gridding near the poles
        theta = np.zeros(ntheta+1)
        for i in range(ntheta+1):
            theta[i] = thetabound[0] + (thetabound[1]-thetabound[0]) * (np.sin(np.pi*i/ntheta))**2
    else:
        theta = np.linspace(thetabound[0], thetabound[1], ntheta+1)
    
    # Create phi grid (regular spacing)
    phi = np.linspace(phibound[0], phibound[1], nphi+1)
    
    # Write the AMR grid to file
    with open('amr_grid.inp', 'w') as f:
        f.write('1\n')                # iformat
        f.write('0\n')                # AMR grid style (0 = regular grid)
        f.write('100\n')              # Coordinate system (100 = spherical)
        f.write('0\n')                # Gridinfo (0 = flash-style)
        f.write('1 1 1\n')            # Include r, theta, phi coordinates
        f.write(f'{nr} {ntheta} {nphi}\n')  # Grid sizes
        for i in range(nr+1):
            f.write(f'{r[i]}\n')      # r-grid walls
        for i in range(ntheta+1):
            f.write(f'{theta[i]}\n')  # theta-grid walls
        for i in range(nphi+1):
            f.write(f'{phi[i]}\n')    # phi-grid walls
    
    print(f"Created amr_grid.inp with Spherical grid of {nr}x{ntheta}x{nphi} cells")
    return r, theta, phi

def create_density_from_function(grid, density_function, nspecies=1, grid_type='cartesian'):
    """
    Create a dust density file from a user-defined function.
    
    Parameters:
    -----------
    grid : tuple
        Grid cell walls as returned by create_amr_grid
    density_function : callable
        Function that takes grid coordinates and returns density
        For Cartesian: f(x, y, z) -> density
        For Spherical: f(r, theta, phi) -> density
    nspecies : int
        Number of dust species
    grid_type : str
        Type of grid: 'cartesian' or 'spherical'
        
    Returns:
    --------
    ndarray
        Density array
    """
    if grid_type.lower() == 'cartesian':
        x, y, z = grid
        nx, ny, nz = len(x)-1, len(y)-1, len(z)-1
        
        # Create cell centers
        xc = 0.5 * (x[:-1] + x[1:])
        yc = 0.5 * (y[:-1] + y[1:])
        zc = 0.5 * (z[:-1] + z[1:])
        
        # Create meshgrid for 3D coordinates
        xx, yy, zz = np.meshgrid(xc, yc, zc, indexing='ij')
        
        # Create density distribution using the provided function
        rho = np.zeros((nspecies, nx, ny, nz))
        for i in range(nspecies):
            rho[i] = density_function(xx, yy, zz)
        
    elif grid_type.lower() == 'spherical':
        r, theta, phi = grid
        nr, ntheta, nphi = len(r)-1, len(theta)-1, len(phi)-1
        
        # Create cell centers
        rc = 0.5 * (r[:-1] + r[1:])
        thetac = 0.5 * (theta[:-1] + theta[1:])
        phic = 0.5 * (phi[:-1] + phi[1:])
        
        # Create meshgrid for spherical coordinates
        rr, tt, pp = np.meshgrid(rc, thetac, phic, indexing='ij')
        
        # Create density distribution using the provided function
        rho = np.zeros((nspecies, nr, ntheta, nphi))
        for i in range(nspecies):
            rho[i] = density_function(rr, tt, pp)
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")
    
    # Write the density file
    with open('dust_density.inp', 'w') as f:
        f.write('1\n')  # iformat
        if grid_type.lower() == 'cartesian':
            f.write(f'{nx} {ny} {nz}\n')  # Grid size
        else:
            f.write(f'{nr} {ntheta} {nphi}\n')  # Grid size
        
        f.write(f'{nspecies}\n')  # Number of dust species
        
        for ispec in range(nspecies):
            if grid_type.lower() == 'cartesian':
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            f.write(f'{rho[ispec, ix, iy, iz]}\n')
            else:
                for ir in range(nr):
                    for itheta in range(ntheta):
                        for iphi in range(nphi):
                            f.write(f'{rho[ispec, ir, itheta, iphi]}\n')
    
    print(f"Created dust_density.inp with {nspecies} species")
    return rho

def setup_model(grid_type='cartesian', model_type='water_fountain', **kwargs):
    """
    Create all input files for a RADMC-3D model.
    
    Parameters:
    -----------
    grid_type : str
        Type of grid: 'cartesian' or 'spherical'
    model_type : str
        Type of model: 'power_law', 'envelope', 'disk', or 'water_fountain'
    **kwargs : dict
        Additional parameters for grid and model setup
        
    Returns:
    --------
    tuple
        (grid, density_array)
    """
    if model_type.lower() == 'water_fountain':
        # Extract parameters for water fountain model
        rin = kwargs.get('rin', 100*au)
        rout = kwargs.get('rout', 5000*au)
        nr = kwargs.get('nr', 1000)
        ntheta = kwargs.get('ntheta', 150)
        
        stellar_radius = kwargs.get('stellar_radius', 288*Rsun)
        stellar_mass = kwargs.get('stellar_mass', 1*Msun)
        stellar_temp = kwargs.get('stellar_temp', 3000)
        
        Mdtorus = kwargs.get('Mdtorus', 1.0/200)
        Mdlobe = kwargs.get('Mdlobe', 0.1/200)
        
        Rtorus = kwargs.get('Rtorus', 1000*au)
        A = kwargs.get('A', 1)
        B = kwargs.get('B', 3)
        C = kwargs.get('C', 0)
        D = kwargs.get('D', 10)
        E = kwargs.get('E', 3)
        F = kwargs.get('F', 2)
        
        Rlobe = kwargs.get('Rlobe', 2500*au)
        oangle = kwargs.get('oangle', np.pi/10)
        width = kwargs.get('width', 0.005)
        
        return setup_water_fountain_model(
            rin=rin, rout=rout, nr=nr, ntheta=ntheta,
            stellar_radius=stellar_radius, stellar_mass=stellar_mass, stellar_temp=stellar_temp,
            Mdtorus=Mdtorus, Mdlobe=Mdlobe,
            Rtorus=Rtorus, A=A, B=B, C=C, D=D, E=E, F=F,
            Rlobe=Rlobe, oangle=oangle, width=width
        )
    else:
        # Extract common parameters
        nspecies = kwargs.get('nspecies', 1)
        nphot_therm = kwargs.get('nphot_therm', 1000000)
        stellar_radius = kwargs.get('stellar_radius', 1.0)
        stellar_temp = kwargs.get('stellar_temp', 5780.0)
        dust_file = kwargs.get('dust_file', 'dustkapscatmat_E40R_300K_a0.3.inp')
        scattering_mode = kwargs.get('scattering_mode', 1)
        
        # Create wavelength grid
        lam_min = kwargs.get('lam_min', 0.1)
        lam_max = kwargs.get('lam_max', 1000)
        nlam = kwargs.get('nlam', 100)
        create_wavelength_grid(lam_min, lam_max, nlam)
        
        # Create grid based on type
        if grid_type.lower() == 'cartesian':
            nx = kwargs.get('nx', 64)
            ny = kwargs.get('ny', 64)
            nz = kwargs.get('nz', 64)
            xbound = kwargs.get('xbound', [-100, 100])
            ybound = kwargs.get('ybound', [-100, 100])
            zbound = kwargs.get('zbound', [-100, 100])
            grid = create_amr_grid(grid_type='cartesian', nx=nx, ny=ny, nz=nz,
                                  xbound=xbound, ybound=ybound, zbound=zbound)
        elif grid_type.lower() == 'spherical':
            nr = kwargs.get('nr', 100)
            ntheta = kwargs.get('ntheta', 32)
            nphi = kwargs.get('nphi', 64)
            rbound = kwargs.get('rbound', [1, 1000])
            thetabound = kwargs.get('thetabound', [0, np.pi])
            phibound = kwargs.get('phibound', [0, 2*np.pi])
            log_r = kwargs.get('log_r', True)
            refinement_at_poles = kwargs.get('refinement_at_poles', True)
            grid = create_amr_grid(grid_type='spherical', nr=nr, ntheta=ntheta, nphi=nphi,
                                  rbound=rbound, thetabound=thetabound, phibound=phibound,
                                  log_r=log_r, refinement_at_poles=refinement_at_poles)
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        if model_type.lower() == 'custom':
            density_function = kwargs.get('density_function')
            if density_function is None:
                raise ValueError("Must provide 'density_function' for custom model type")
            rho = create_density_from_function(grid, density_function, nspecies, grid_type)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Setup dust opacities
        for i in range(nspecies):
            copy_dust_opacity(dust_file, f'dust_kapscatmat_{i+1}.inp')
        create_dustopac(nspecies, scattering_mode)
        
        # Create star
        create_stars(stellar_radius, stellar_temp)
        
        # Create control file
        modified_random_walk = kwargs.get('modified_random_walk', 1)
        create_radmc3d_control(nphot_therm=nphot_therm, 
                              scattering_mode_max=scattering_mode,
                              modified_random_walk=modified_random_walk,
                              istar_sphere=1, water_fountain=False)
        
        print(f"Model setup complete. Type: {model_type} on {grid_type} grid. Ready to run RADMC-3D.")
        return grid, rho 

def copy_temp_dependent_dust_opacities(base_path='.'):
    """
    Copy temperature-dependent dust opacity files to the current directory.
    
    Parameters:
    -----------
    base_path : str
        Path where the dust opacity files are located
        
    Returns:
    --------
    bool
        True if all files were copied successfully, False otherwise
    """
    # List of required dust opacity files
    opacity_files = [
        'dustkapscatmat_E40R_10K_a0.3.inp',
        'dustkapscatmat_E40R_100K_a0.3.inp',
        'dustkapscatmat_E40R_200K_a0.3.inp',
        'dustkapscatmat_E40R_300K_a0.3.inp'
    ]
    
    # First check if any of the files already exist in the current directory
    existing_files = []
    for file in opacity_files:
        if os.path.exists(file):
            existing_files.append(file)
    
    if existing_files:
        print(f"Found {len(existing_files)} opacity files already in the current directory:")
        for file in existing_files:
            print(f"  - {file}")
    
    # Check which files exist in the source directory
    source_files = []
    for file in opacity_files:
        source = os.path.join(base_path, file)
        if os.path.exists(source):
            source_files.append(file)
    
    print(f"Found {len(source_files)} opacity files in {base_path}:")
    for file in source_files:
        print(f"  - {file}")
    
    # Copy the files
    success = True
    for file in opacity_files:
        source = os.path.join(base_path, file)
        try:
            if os.path.exists(source):
                shutil.copy(source, file)
                print(f"Copied {file}")
            else:
                print(f"Warning: Dust file {file} not found in {base_path}")
                success = False
        except Exception as e:
            print(f"Error copying {file}: {e}")
            success = False
    
    # Final verification
    if success:
        print("All opacity files were copied successfully.")
    else:
        print("WARNING: Not all opacity files were found or copied.")
        print("Current directory contents:")
        for file in os.listdir('.'):
            if file.startswith('dustkap'):
                print(f"  - {file}")
    
    return success 

def create_single_dust_opacity(extension='E40R_300K_a0.3'):
    """
    Create a dustopac.inp file for RADMC-3D with a single dust species.
    
    Parameters:
    -----------
    extension : str
        Extension of the dust opacity file (e.g., 'E40R_300K_a0.3')
        
    Returns:
    --------
    None
        Writes the dustopac.inp file
    """
    with open('dustopac.inp', 'w') as f:
        f.write('2\n')  # Format number
        f.write('1\n')  # Number of dust species
        f.write('============================================================================\n')
        f.write('10\n')  # Method for this dust species (10=full scattering matrix)
        f.write('0\n')   # 0=thermal grain
        f.write(f'{extension}\n')  # Extension of dust opacity file
        f.write('----------------------------------------------------------------------------\n')
    
    print(f"Created dustopac.inp with single dust species: {extension}") 

def verify_dust_temperature_file(filepath='dust_temperature.dat'):
    """
    Verify if a dust_temperature.dat file exists and is usable for direct spectrum calculation.
    
    Parameters:
    -----------
    filepath : str
        Path to the dust_temperature.dat file
        
    Returns:
    --------
    bool
        True if the file exists and appears valid, False otherwise
    """
    if not os.path.exists(filepath):
        print(f"Error: Dust temperature file '{filepath}' not found.")
        return False
    
    # Check if the file has content (not empty)
    if os.path.getsize(filepath) == 0:
        print(f"Error: Dust temperature file '{filepath}' is empty.")
        return False
    
    # Try to read the first few bytes to verify it's readable
    try:
        with open(filepath, 'rb') as f:
            header = f.read(16)  # Read just enough to check it's a binary file
        if len(header) == 0:
            print(f"Error: Could not read data from '{filepath}'.")
            return False
        
        print(f"Found valid dust temperature file: {filepath}")
        return True
    except Exception as e:
        print(f"Error verifying dust temperature file: {e}")
        return False 