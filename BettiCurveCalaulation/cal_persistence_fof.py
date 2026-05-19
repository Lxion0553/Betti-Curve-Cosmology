import numpy as np
import os
import sys
sys.path.append('..')
from BettiCurveCalaulation.utils import periodic_alpha
from BettiCurveCalaulation.profiling import profile_halos_to_betti
from multiprocessing import Pool, freeze_support

redshift = 0.5
fofdict = {0.0:4, 0.5:3, 1.0:2, 2.0:1, 3.0:0}
a = 1/(1+redshift)

boxsize = 1000
subbox = True
per_file_end = '.per'
bc_file_end = '.bc'
enable_profiling = False

if subbox:
    boxsize = 368
    per_file_end = '.persub'
    bc_file_end = '.bcsub'

output_name = f'z{redshift}' + per_file_end

def process_file(halos, save_path):

    n_input = int(halos.shape[0])

    if subbox:
    # test volume should be smaller than train for the instance of cosmic variance. Here 1Gpc (sim) ~ 20Gpc (DESI)
        logic_x = (halos[...,0] < boxsize)
        logic_y = (halos[...,1] < boxsize)
        logic_z = (halos[...,2] < boxsize)
        logic_indices = np.logical_and(np.logical_and(logic_x,logic_y),logic_z)
        halos = halos[logic_indices]

    new_file_path_betti = save_path.replace(per_file_end, bc_file_end)

    # One shared log per redshift directory; safe for multiprocessing via file lock.
    redshift_dir = os.path.dirname(os.path.dirname(save_path))
    log_path = os.path.join(redshift_dir, f'profiling_{output_name}.jsonl') if enable_profiling else None

    result = profile_halos_to_betti(
        periodic_alpha_cls=periodic_alpha,
        halos=halos,
        boxsize=boxsize,
        per_save_path=save_path,
        bc_save_path=new_file_path_betti,
        resolution=25,
        sample_range_factor=2.5,
        scale=True,
        periodic=True,
        n_halos_input=n_input,
        log_path=log_path,
        enable_profiling=enable_profiling,
    )

    if result.ok:
        print(f'Persistence diagram saved to {save_path}')
        print('-----------------------------------------')


if __name__ == '__main__':
    freeze_support()

    num_processes = 128
    sims = ['nwLH']
    file_paths = []

    for sim in sims:
        n_tot = 2000 if sim == 'nwLH' else 500
        save_root = f'/hscratch/ljy/ProcessedFoFData/{sim}/z{redshift}'
        file_paths += [f'{save_root}/{i}/halos.txt' for i in range(n_tot)]

    # check for bc file existence
    for f in file_paths.copy():
        d = os.path.dirname(f)
        files = os.listdir(d)
        if output_name.replace(per_file_end, bc_file_end) in files:
            file_paths.remove(f)
            print(f'Already calculated {f}')

    print(f'Number of files to process: {len(file_paths)}')

    def pool_wrapper(file_path):
        d = os.path.dirname(file_path)
        halos = np.loadtxt(file_path, skiprows=1)[:,1:4]
        save_path = os.path.join(d, output_name)
        process_file(halos, save_path)

    with Pool(processes=num_processes) as pool:
        pool.map(pool_wrapper, file_paths)