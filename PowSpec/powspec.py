import os
import numpy as np
from multiprocessing import Pool, freeze_support
from functools import partial

os.chdir('/home/ljy/project/powerspec')

def files_in_directory(root_dir, target):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files: 
            if file == target: 
                file_path = os.path.join(root, file)   
                file_paths.append(file_path)
    return file_paths

def powspec(in_out_pair, w_list, Om_list, kmin=0, kmax=0.3, los=None, conf='powspec.conf'):
    file_path, out_path = in_out_pair
    if los is None:
        los = 'z'
        
    los_map = {'x': '[1,0,0]',
           'y': '[0,1,0]',
           'z': '[0,0,1]'}

    # multipole 0, 2, 4

    if isinstance(w_list, np.ndarray):
        index = int(os.path.split(os.path.dirname(file_path))[-1])
        w = w_list[index]
    else:
        w = w_list

    if isinstance(Om_list, np.ndarray):
        index = int(os.path.split(os.path.dirname(file_path))[-1])
        Om = Om_list[index]
    else:
        Om = Om_list

    os.system('./POWSPEC -d {} -c {} --omega-m {} --eos-w {} --line-of-sight {} -k {} -K {} -a {}'.format(file_path, conf, Om, w, los_map[los], kmin, kmax, out_path))

def read_spec(file, max_order=2):
    """
    return (k, P0, ..., Pmax_order)
    """
    pk = []
    if max_order == 0:
        return np.loadtxt(file, comments='#')[:,[0,5]]
    else:
        pk.append(np.loadtxt(file, comments='#')[:,0])
        for order in range(0, max_order+1, 2):
            pk.append(np.loadtxt(file, comments='#')[:,5+order//2])
        return np.stack(pk)

def load_pk(root_dir, target, max_order=4, save_dir=None):
    dirs = files_in_directory(root_dir,target=target)

    spec = np.array([read_spec(file, max_order) for file in dirs])

    if save_dir is not None:
        np.save(save_dir,spec)

    return spec

def sub_cut(halos, boxsize=368):
    logic_x = (halos[...,0] < boxsize)
    logic_y = (halos[...,1] < boxsize)
    logic_z = (halos[...,2] < boxsize)
    logic_indices = np.logical_and(np.logical_and(logic_x,logic_y),logic_z)
    halos = halos[logic_indices]
    return halos

if __name__ == '__main__':
    freeze_support()
    n_processes = 4
    # paras = np.loadtxt('/hscratch/ljy/ProcessedData/nwLH/latin_hypercube_params_nwLH.txt', skiprows=1)
    # kmin = 0.018
    # kmax = 0.3
    # for path in files_in_directory('/hscratch/ljy/ProcessedData/nwLH/z0.5', 'coords.npy'):
    #     npy = np.load(path)
    #     np.savetxt(path.replace('.npy', '.txt'), npy, fmt='%.4f')
    #     path = path.replace('.npy', '.txt')
    #     powspec(path, paras, kmin=kmin, kmax=kmax)
    # for path in files_in_directory('/hscratch/ljy/ProcessedData/fiducial_ZA/z0.5', 'coords.npy'):
    #     npy = np.load(path)
    #     np.savetxt(path.replace('.npy', '.txt'), npy, fmt='%.4f')
    #     path = path.replace('.npy', '.txt')
    #     powspec(path, None, kmin=kmin, kmax=kmax)

    # load_pk('/hscratch/ljy/ProcessedData/fiducial_ZA/z0.5',f'[{kmin},{kmax}].spec',4,f'/home/ljy/BettiCurveCosmo/PowSpec/data/Phh_fZA_[{kmin},{kmax}].npy')
    # load_pk('/hscratch/ljy/ProcessedData/nwLH/z0.5',f'[{kmin},{kmax}].spec',4,f'/home/ljy/BettiCurveCosmo/PowSpec/data/Phh_nwLH_[{kmin},{kmax}].npy')

    # coord.spec : kmin=0, kmax=0.3
    sub = False
    boxsize = 1000 if not sub else 368
    fof = True
    # rsd = None
    rsd = 'z'
    redshift = 'z0.5'

    if fof:
        save_root = '/hscratch/ljy/ProcessedFoFData'
    else:
        save_root = '/hscratch/ljy/ProcessedData'

    kmin = 0.018
    kmax = 0.3

    out_name = f'[{kmin},{kmax}]'
    if rsd is not None:
        out_name += f'_rsd{rsd}'
    if sub:
        out_name += '_sub'
    out_name += '.spec'

    # sims = ['nwLH', 'Ob2_p', 'Mnu_pp', 's8_p', 'fiducial_ZA', 'fiducial', 'Om_p', 'w_p', 's8_m', 'Om_m', 'Ob2_m', 'ns_m', 'h_m', 'h_p', 'Mnu_p', 'w_m', 'ns_p']
    sims = ['s8_m']
    indicies = ['251', '315', '281', '329', '267', '336', '343', '274', '260', '237']

    for sim in sims:
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

        n_tot = 2000 if sim == 'nwLH' else 500
        sim_root = f'{save_root}/{sim}/{redshift}'

        if rsd is not None:
            # coord_paths = [f'{sim_root}/{i}/rsd{rsd}.npy' for i in range(n_tot)]
            coord_paths = [f'{sim_root}/{i}/rsd{rsd}.npy' for i in indicies]
            for path in coord_paths:
                npy = np.load(path)
                if sub:
                    npy = sub_cut(npy, boxsize=boxsize)
                np.savetxt(os.path.join(os.path.dirname(path), 'coords.tmp'), npy, fmt='%.4f')
        else:
            # coord_paths = [f'{sim_root}/{i}/halos.txt' for i in range(n_tot)]
            coord_paths = [f'{sim_root}/{i}/halos.txt' for i in indicies]
            for path in coord_paths:
                halos = np.loadtxt(path, skiprows=1)[:,1:4]
                if sub:
                    halos = sub_cut(halos, boxsize=boxsize)
                np.savetxt(os.path.join(os.path.dirname(path), 'coords.tmp'), halos, fmt='%.4f')
            
        # file_paths = [f'{sim_root}/{i}/coords.tmp' for i in range(n_tot)]
        file_paths = [f'{sim_root}/{i}/coords.tmp' for i in indicies]
        out_paths = [os.path.join(os.path.dirname(f), out_name) for f in file_paths]

        with Pool(n_processes) as pool:
            powspec_partial = partial(powspec, w_list=w, Om_list=Om, kmin=kmin, kmax=kmax, los=rsd, conf='./powspec.conf')
            pool.map(powspec_partial, zip(file_paths, out_paths))

        load_pk(sim_root, out_name, 4, f'/home/ljy/BettiCurveCosmo/PowSpec/data/fof/Phh_{sim}_{os.path.splitext(out_name)[0]}.npy')

        for path in file_paths:
            os.remove(path)
