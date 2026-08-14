import h5py 
from astropy.table import Table
import os
from glob import glob
import numpy as np

# Input location
catalog_h5_list = glob('stellar_params_catalog_*.h5')
catalog_h5_list.sort()


def main():
    for catalog_h5_loc in catalog_h5_list:
        print(f"Loading {catalog_h5_loc}")
        output_fits = Table()
        with h5py.File(catalog_h5_loc, 'r') as f:
            for i, key in enumerate(f.keys()):
                print(f"Loading {i+1}/{len(f.keys())}: {key}")
                output_fits[key] = f[key][:]
        base_fn,_ = os.path.splitext(catalog_h5_loc)
        catalog_fits_loc = base_fn + '.fits'
        print(f"Saving to {catalog_fits_loc}")
        output_fits.write(catalog_fits_loc)



if  __name__ == "__main__":
    main()
