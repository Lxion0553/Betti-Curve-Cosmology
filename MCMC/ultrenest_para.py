import os, sys
import argparse
from multiprocessing import freeze_support, Pool
from functools import partial
from unittest import result
import numpy as np
import joblib
from ultranest import ReactiveNestedSampler
import numpy as np
import shutil

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.compose import TransformedTargetRegressor
import joblib 
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import fmin_l_bfgs_b
from ultranest.stepsampler import RegionSliceSampler

sys.path.append('/home/ljy/BettiCurveCosmo')
from Emulator.GPRemulator import optimizer, custom_optimizer, PCAForY, Emulator
from Data.DataProcess import cut_data

def save_plots(sampler, result_dir):
    """Save the result, trace, and corner plots."""
    os.makedirs(result_dir, exist_ok=True)
    sampler.plot()
    sampler.plot_trace()
    sampler.plot_corner()

# def load_emulator(emulator_path):
#     emulator = Emulator()
#     emulator.load(emulator_path)
#     return emulator

def prior(cube):
    params = cube.copy()
    params[0] = cube[0] * 0.4 + 0.1
    params[1] = cube[1] * 0.04 + 0.03
    params[2] = cube[2] * 0.4 + 0.5
    params[3] = cube[3] * 0.4 + 0.8
    params[4] = cube[4] * 0.4 + 0.6
    # params[5] = cube[5] * 100
    params[5] = cube[5]
    params[6] = cube[6] * 0.6 - 1.3
    params[7] = cube[7] * 100
    return params

def unfinished_index(result_dir, target_dir, total=500):    
 
    full_index = list(range(total))
    
    for index_dir in os.listdir(result_dir):
    
        check_dir = os.path.join(result_dir, index_dir, target_dir)
    
        if os.path.exists(os.path.join(check_dir, 'chains', 'weighted_post.txt')):
            # finished.append(check_dir)
            full_index.remove(int(index_dir))
        # elif os.path.exists(check_dir):
        #     shutil.rmtree(check_dir)
    
    return full_index

if __name__ == "__main__":
    freeze_support()
    n_processes = 24
    # n_cosmo = 128

    emulator_path = '/home/ljy/BettiCurveCosmo/Emulator/model-fof'
    # emulator_path = '/home/ljy/BettiCurveCosmo/Emulator/model-fof-rsdz'
    # observed_data = np.load('/home/ljy/BettiCurveCosmo/Data/sens_testdata/z0.5/fiducialZA-bc.npy') 
    observed_data = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/DimlessBC/fiducial_ZA-fof-bc.npy') # !!! Normalization
    # observed_data = np.load('/home/ljy/BettiCurveCosmo/Data/TestData/DimlessBC/fiducial_ZA-fof-rsdz-bc.npy') # !!! Normalization
    result_dir = '/home/ljy/BettiCurveCosmo/Emulator/model-fof/mcmc_result/RobustAnalysis-FZA'
    # result_dir = '/home/ljy/BettiCurveCosmo/Emulator/model-fof-rsdz/mcmc_result/RobustAnalysis-FZA'
    cuts = [range(1, 6), range(2, 15), range(9, 19)]

    for dim in np.arange(2, -1, -1):
    # for dim in [0]:

        obs = observed_data[dim][:, cuts[dim]]

        n = obs.shape[0]
        H_factor = (n-len(cuts[dim])-2)/(n-1)

        cov = np.cov(obs.T)
        parameters = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu', 'w', 'scale_factor']

        def Run_ultrenest(observed_data, result_dir):

            emulator = joblib.load(os.path.join(emulator_path, f'autosklearn_model_{dim}.pkl'))

            def log_likelihood(params):
                predicted_BettiCurve = emulator.predict(params.reshape(1,-1))
                residual = observed_data - predicted_BettiCurve.squeeze()
                chi_squared = H_factor * np.dot(residual.T, np.linalg.solve(cov, residual))
                return -0.5 * chi_squared

            save_dir = os.path.join(result_dir, f'dim{dim}')
            os.makedirs(save_dir, exist_ok=True)

            # Define the sampler
            sampler = ReactiveNestedSampler(parameters, log_likelihood, prior, log_dir=save_dir, resume=True)

            sampler.run(
            min_num_live_points=400,
            dlogz=0.5,
            min_ess=400,
            # update_interval_volume_fraction=0.4,
            # max_num_improvement_loops=3,
            # frac_remain=0.05,  # Terminate earlier to avoid inefficiency
            )

            # Print and save results
            sampler.print_results()

            # Save plots
            save_plots(sampler, save_dir)

        index = unfinished_index(result_dir, f'dim{dim}', total=n)

        data = [obs[j,:].squeeze() for j in index]#[:n_cosmo]
        # emulator = [joblib.load(os.path.join(emulator_path, f'autosklearn_model_{dim}.pkl')) for _ in index]#[:n_cosmo]
        results_dir = [os.path.join(result_dir, f'{j}') for j in index]#[:n_cosmo]

        with Pool(n_processes) as pool:
            pool.starmap(Run_ultrenest, zip(data, results_dir))


