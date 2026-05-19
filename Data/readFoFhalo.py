import sys
sys.path.append('..')
from BettiCurveCalaulation.utils import read_fof_halos
import os
from multiprocessing import Pool, freeze_support

redshift = 0.5
fofdict = {0:4, 0.5:3, 1.0:2, 2.0:1, 3.0:0}

boxsize = 1000
subbox = False
per_file_end = '.per'
bc_file_end = '.bc'

if subbox:
    boxsize = 368
    per_file_end = '.persub'
    bc_file_end = '.bcsub'

def process(pair, snapnum=fofdict[redshift], masscut=1.31e13):
    rpath, spath = pair
    if not os.path.exists(os.path.dirname(spath)):
        os.makedirs(os.path.dirname(spath))
    print(f'Processing {rpath}...')
    read_fof_halos(rpath, spath, snapnum=snapnum, mass_cut=masscut)

if __name__ == '__main__':
    freeze_support()
    num_processes = 128
    
    sims = ['ns_p']

    for sim in sims:
        n_tot = 2000 if sim == 'nwLH' else 500
        data_root = "/work/shared_data/Cosmo_Sims/Quijote/Halos/FoF/latin_hypercube_nwLH" if sim == 'nwLH' else f"/work/ljy/Quijote/Halos/FoF/{sim}"
        # n_tot = 15000
        # data_root = "/work/shared_data/Cosmo_Sims/Quijote/Halos/FoF/fiducial"
        save_root = f'/hscratch/ljy/ProcessedFoFData/{sim}/z{redshift}'
        save_paths = [f'{save_root}/{i}/halos.txt' for i in range(n_tot)]
        data_paths = [f'{data_root}/{i}' for i in range(n_tot)]

        # check existence
        if not os.path.exists(f'{save_root}'):
            os.makedirs(f'{save_root}')

        with Pool(processes=num_processes) as pool:
            pool.map(process, zip(data_paths, save_paths))




