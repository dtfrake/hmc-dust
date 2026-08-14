
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import astropy.units as u
from astropy.coordinates import SkyCoord
from glob import glob
import numpy as np
import h5py 

      
def plot_param_distribution(d):
    # Extract Teff, [Fe/H], logg, extinction, parallax
    # Units: 10^3 K, dex, dex, mag, mas
    teff, feh, logg, E, plx = d['stellar_params_est'].T
    
    # teff vs logg
    print('Plotting T_eff vs logg ...')
    idx = (
        # Basic reliability cut
        (d['quality_flags'] < 8) 
        # Confidence in teff > 0.5
        & (d['teff_confidence'] > 0.5) 
        # Confidence in logg > 0.5
        & (d['logg_confidence'] > 0.5) 
        # Uncertainties are small
        & (d['stellar_params_err'][:,0] < 0.500) # 500 K in T_eff
        & (d['stellar_params_err'][:,2] < 0.5)  # 0.5 dex in logg
    )
    print(f' -> {np.count_nonzero(idx)} stars selected.')
    fig,ax = plt.subplots(1, 1, figsize=(6,6))
    # Convert units of teff to K
    ax.hist2d(teff[idx]*1000, logg[idx],
              range=((3300, 11000), (-1, 5)), 
              bins=100, norm=LogNorm(),
              rasterized=True)
    ax.set_xlim(11000-1, 3300)
    ax.set_ylim(5.2, 0.5)
    ax.tick_params(axis='both', direction='in',
                   width=1, length=4, which='both')
    ax.set_xlabel(r'$T_{\mathrm{eff}} \ \left(\mathrm{K}\right)$')
    ax.set_ylabel(r'$\log g \ \left(\mathrm{dex}\right)$')
    fig.savefig('catalog_teff_vs_logg.png')
    plt.close(fig)
    
    # teff vs [Fe/H]
    print('Plotting T_eff vs [Fe/H] ...')
    idx = (
        # Basic reliability cut
        (d['quality_flags'] < 8) 
        # Confidence in teff > 0.5
        & (d['teff_confidence'] > 0.5) 
        # Confidence in [Fe/H] > 0.5
        & (d['feh_confidence'] > 0.5) 
        # Uncertainties are small
        & (d['stellar_params_err'][:,0] < 0.500) # 500 K in T_eff
        & (d['stellar_params_err'][:,1] < 0.5)   # 0.5 dex in [Fe/H]
    )
    print(f' -> {np.count_nonzero(idx)} stars selected.')
    fig,ax = plt.subplots(1, 1, figsize=(6,6))
    # Convert units of teff to K
    ax.hist2d(teff[idx]*1000, feh[idx], range=((3300, 11000), (-3, 1)), 
                         bins=100, norm=LogNorm(), rasterized=True)
    ax.set_xlim(11000-1, 3300)
    ax.set_ylim(-3, 1)
    ax.tick_params(axis='both', direction ='in',
                   width=1, length=4, which='both')
    ax.set_xlabel(r'$T_{\mathrm{eff}} \ \left(\mathrm{K}\right)$')
    ax.set_ylabel(r'$\mathrm{[Fe/H]} \ \left(\mathrm{dex}\right)$')
    fig.savefig('catalog_teff_vs_feh.png')
    plt.close(fig)


def plot_extinction_skymap(d, nbins=90):
    # Calculate Galacic coordinates of the stars
    c = SkyCoord(d['ra']*u.deg, d['dec']*u.deg, frame='icrs')
    c = c.transform_to('galactic')
    l = c.l.deg
    b = c.b.deg
    l[l>180] = l[l>180]-360 # Wrap at l = 180 deg
    plx = d['stellar_params_est'][:,-1] # Estimated parallax in mas
    distance = 1./plx # Distance in kpc
    
    # Extinction and errors:
    E = d['stellar_params_est'][:,-2]
    E_err = d['stellar_params_err'][:,-2]
    
    # Select stars
    idx = (
        # Basic reliability cut
        (d['quality_flags'] < 8) 
        # Within 1 kpc
        & (distance < 1.)
    )  
    print(f' -> {np.count_nonzero(idx)} stars selected.')
    
    # Bin stars in (l,b)
    l_bins = np.linspace(-180, 180, 2*nbins+1)
    b_bins = np.linspace(-90, 90, nbins+1)
    
    l_idx = np.searchsorted(l_bins, l[idx])-1
    b_idx = np.searchsorted(b_bins, b[idx])-1
    
    # Weighted sum of extinction in each (l,b)-pixel
    E_ivar = 1 / (E_err**2 + 1e-4) # Inverse variance of extinction
    E_sum = np.zeros((2*nbins, nbins))
    np.add.at(E_sum, (l_idx, b_idx), E[idx]*E_ivar[idx])
    E_weights = np.zeros((2*nbins, nbins))    
    np.add.at(E_weights, (l_idx, b_idx), E_ivar[idx])
    extinction_map = E_sum / (E_weights+1e-9)
    
    fig,ax = plt.subplots(1, 1, figsize=(6,4))
    im = ax.imshow(extinction_map.T,
                   extent=(-180, 180, -90, 90), 
                   aspect='auto', origin='lower',
                   interpolation='nearest', cmap='inferno',
                   norm=PowerNorm(0.5, vmin=0, vmax=1))
    ax.invert_xaxis()
    cbar = fig.colorbar(im, ax=ax, label=r'$E$',
                        fraction=0.08, pad=0.03)
    cbar.ax.tick_params(axis='both', direction='in', which='major')
    ax.tick_params(axis='both', direction = 'in', which='major')
    ax.set_xlabel(r'$\ell \ (\mathrm{deg})$')
    ax.set_ylabel(r'$b \ (\mathrm{deg})$')
    ax.set_title(r'$\mathrm{Extinction\ to\ 1\, kpc}$')
    fig.subplots_adjust(
        left=0.11,
        right=0.94,
        bottom=0.13,
        top=0.90
    )
    fig.savefig('extinction_map_1_kpc.png')
    plt.close(fig) 


def main():
    catalog_h5_list = glob('stellar_params_catalog_*.h5')
    nfold = 100

    # Load catalog data
    d = {}
    for j, catalog_h5_loc in enumerate(catalog_h5_list):
        with h5py.File(catalog_h5_loc, 'r') as f:
            print(f"Loading file {catalog_h5_loc}")
            for i, key in enumerate(f.keys()):
                if key not in d:
                    d[key] = [] 
                print(f"Loading {i+1}/{len(f.keys())}: {key}")
                d[key].append(f[key][::nfold]) # Only keep 1 of every <nfold> stars                 
        
    for key in d:
        d[key] = np.concatenate(d[key], axis=0)
    print(f' -> Loaded {len(d["ra"])} stars.')

    # Make plots
    print('Plotting distribution of stellar atmospheric parameters ...')
    plot_param_distribution(d)

    print('Plotting sky map of average stellar extinctions ...')
    plot_extinction_skymap(d, nbins=90)


if __name__=="__main__":
    main()
