import os, sys, shutil
import argparse
from multiprocessing import freeze_support, Pool
from functools import partial
from unittest import result
import numpy as np
import joblib
from ultranest import ReactiveNestedSampler
import numpy as np

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

def load_data(data_path, cuts):

    observed_data = cut_data(np.load(data_path), cuts)
    observed_data = np.concatenate([obs for obs in observed_data], axis=1)
    cov_matrix = np.cov(observed_data.T)
    H_factor = (observed_data.shape[0]-observed_data.shape[1]-2)/(observed_data.shape[0]-1)

    return observed_data, cov_matrix, H_factor

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
    n_processes = 11

    emulator_path = '/home/ljy/BettiCurveCosmo/Emulator/model-fof'
    data_path = '/home/ljy/BettiCurveCosmo/Data/TestData/DimlessBC/fiducial_ZA-fof-bc.npy'
    result_dir = '/home/ljy/BettiCurveCosmo/Emulator/model-fof/mcmc_result//RobustAnalysis-FZA'
    cuts = [range(1, 6), range(2, 15), range(9, 19)]

    obs, cov, H_factor = load_data(data_path, cuts)

    n = obs.shape[0]

    parameters = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu', 'w', 'scale_factor']

    def Run_ultrenest(observed_data, result_dir):

        emulator = [joblib.load(os.path.join(emulator_path, f'autosklearn_model_{dim}.pkl')) for dim in range(3)]

        def log_likelihood(params):
            predicted_BettiCurve = np.concatenate([(e.predict(params.reshape(1,-1))).squeeze() for e in emulator]) 
            residual = observed_data - predicted_BettiCurve
            chi_squared = H_factor * np.dot(residual.T, np.linalg.solve(cov, residual))
            return -0.5 * chi_squared
        
        # Check if the result directory exists
        result_dir = os.path.join(result_dir, 'combined')
        os.makedirs(result_dir, exist_ok=True)

        # Define the sampler  
        sampler = ReactiveNestedSampler(parameters, log_likelihood, prior, log_dir=result_dir, resume=True)
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
        save_plots(sampler, result_dir)

    index = unfinished_index(result_dir, 'combined', total=n)

    data = [obs[j,:].squeeze() for j in index]
    results_dir = [os.path.join(result_dir, f'{j}') for j in index]
    # emulator = [[joblib.load(os.path.join(emulator_path, f'autosklearn_model_{dim}.pkl')) for dim in range(3)] for _ in index]

    with Pool(n_processes) as pool:
        # pool.starmap(Run_ultrenest, zip(emulator, data, results_dir))
        pool.starmap(Run_ultrenest, zip(data, results_dir))
    # for i in range(len(index)):
    #     Run_ultrenest(data[i], results_dir[i])
        

