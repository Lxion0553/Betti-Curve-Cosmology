import numpy as np
import os
import sys
sys.path.append('..')
from BettiCurveCalaulation.utils import read_halos, periodic_alpha, rsd_correction
from multiprocessing import Pool, freeze_support
from functools import partial

redshift = 0.5
fofdict = {0.0:4, 0.5:3, 1.0:2, 2.0:1, 3.0:0}
a = 1/(1+redshift)

boxsize = 1000
subbox = True
per_file_end = '.per'
bc_file_end = '.bc'

if subbox:
    boxsize = 368
    per_file_end = '.persub'
    bc_file_end = '.bcsub'

overwrite = False

def process_file(halos, save_path):

    if subbox:
    # test volume should be smaller than train for the instance of cosmic variance. Here 1Gpc (sim) ~ 20Gpc (DESI)
        logic_x = (halos[...,0] < boxsize)
        logic_y = (halos[...,1] < boxsize)
        logic_z = (halos[...,2] < boxsize)
        logic_indices = np.logical_and(np.logical_and(logic_x,logic_y),logic_z)
        halos = halos[logic_indices]

    new_file_path_betti = save_path.replace(per_file_end, bc_file_end)

    try:
    # calculate and save persistence diagram 
        diag = periodic_alpha(halos, 
                            boxsize=boxsize,
                            save_dir=save_path)
        diag.cal_persi()
        print(f'Persistence diagram saved to {save_path}')
        print('-----------------------------------------')
        l = diag.ave_dist
        diag.cal_betti(resolution=25,sample_range=[0,2.5*l],scale=True,save_dir=new_file_path_betti)
    except:
        pass

def rsd(halo_file, Om_list, w_list=-1, a=a):
    index = int(os.path.split(os.path.dirname(halo_file))[-1])

    if isinstance(w_list, np.ndarray):
        w = w_list[index]
    else:
        w = w_list

    if isinstance(Om_list, np.ndarray):
        Om = Om_list[index]
    else:
        Om = Om_list

    # for los in ['x', 'y', 'z']:
    for los in ['z']:
        rsd_path = os.path.join(os.path.dirname(halo_file), f'rsd{los}.npy')
        if os.path.exists(rsd_path) and not overwrite:
            rsd_coords = np.load(rsd_path)
            print(f'Already calculated {rsd_path}')
        else:
            rsd_coords = rsd_correction(halo_file, direction=los, w=w, fmt='fof', a=a, Om=Om)
            np.save(rsd_path, rsd_coords)

        rsd_per_path = rsd_path.replace('.npy', per_file_end)
        print(f'{rsd_per_path} : {w}, {Om}')
        process_file(rsd_coords, rsd_per_path)


if __name__ == '__main__':
    freeze_support()

    num_processes = 4
    # sims = [ 'nwLH',
    #         'Om_p',
    #         'w_p',
    #         'Om_m',
    #         'w_m',
    #         ]
    # sims = ['fiducial_ZA','Ob2_p',
    #         'Mnu_pp',
    #         's8_p',
    #         'fiducial','s8_m','Ob2_m',
    #         'ns_m',
    #         'h_m',
    #         'h_p',
    #         'Mnu_p','ns_p']

    sims = ['s8_m']

    file_paths = []

    for sim in sims:
        n_tot = 2000 if sim == 'nwLH' else 500
        save_root = f'/hscratch/ljy/ProcessedFoFData/{sim}/z{redshift}'
        file_paths += [f'{save_root}/{i}/halos.txt' for i in range(n_tot)]


        if sim == 'Om_m':
            Om = 0.3075
        elif sim == 'Om_p':
            Om = 0.3275
        else:
            Om = 0.3175

        if sim == 'w_m':
            w = -1.05
        elif sim == 'w_p':
            w = -0.95
        else:
            w = -1.0

        if sim == 'nwLH':
            w = np.loadtxt('/hscratch/ljy/ProcessedFoFData/nwLH/latin_hypercube_params_nwLH.txt', skiprows=1)[:,-1]
            Om = np.loadtxt('/hscratch/ljy/ProcessedFoFData/nwLH/latin_hypercube_params_nwLH.txt', skiprows=1)[:,0]

    # check for bc file existence
        for f in file_paths.copy():
            files = os.listdir(os.path.dirname(f))
            if f'rsdz{bc_file_end}' in files:
                file_paths.remove(f)
                print(f'Already calculated {f}')

    with Pool(processes=num_processes) as pool:
        pool.map(partial(rsd, w_list=w, Om_list=Om, a=a), file_paths)