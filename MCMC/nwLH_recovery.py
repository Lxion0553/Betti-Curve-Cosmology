import numpy as np
from ultranest import ReactiveNestedSampler
import os, sys
sys.path.append('..')
import joblib 
from Data.DataProcess import cut_data
from Emulator.GPRemulator import load_emulator_data
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support, Pool


EMULATOR = None
L_COV = None


def save_plots(sampler, result_dir):
    """Save the result, trace, and corner plots."""
    os.makedirs(result_dir, exist_ok=True)
    sampler.plot()
    sampler.plot_trace()
    sampler.plot_corner()

def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _psd_project(C: np.ndarray, eps_rel: float = 1e-12) -> np.ndarray:
    """Project symmetric matrix to PSD by eigenvalue clipping."""
    C = _sym(C)
    w, V = np.linalg.eigh(C)
    w_max = float(np.max(w)) if w.size else 0.0
    floor = max(0.0, eps_rel * w_max)
    w_clip = np.maximum(w, floor)
    return _sym((V * w_clip) @ V.T)

def chol_factor(C: np.ndarray, jitter0: float = 1e-12, max_tries: int = 10) -> np.ndarray:
    """Stable Cholesky for (near) SPD matrices; adds diagonal jitter if needed.

    Returns L where C ≈ L L^T.
    """
    C = _sym(C)
    d = C.shape[0]

    # First try without jitter
    try:
        L = np.linalg.cholesky(C)
        return L
    except np.linalg.LinAlgError:
        pass

    # Then try with increasing jitter
    jitter = float(jitter0)
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(C + jitter * np.eye(d))
            return L
        except np.linalg.LinAlgError:
            jitter *= 10.0

    C_psd = _psd_project(C)
    L = np.linalg.cholesky(C_psd)

    return L

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

def logging(parameters, result_dir, emulator_path,  cuts):
    log_path = os.path.join(result_dir, 'doc.txt')
    with open(log_path, 'w') as f:
        f.write(f"emulator_path: {emulator_path}\n")
        f.write(f"result_dir: {result_dir}\n")
        f.write(f"parameters: {parameters}\n")
        f.write(f"cuts: {cuts}\n")

def unfinished_index(result_dir, target_dir, total=500):    
 
    full_index = list(range(total))
    
    for index_dir in os.listdir(result_dir):
    
        check_dir = os.path.join(result_dir, index_dir, target_dir)
    
        if os.path.exists(os.path.join(check_dir, 'chains', 'weighted_post.txt')):
            # finished.append(check_dir)
            full_index.remove(int(index_dir))
    
    return full_index

if __name__ == "__main__":
    freeze_support()
    n_process = 13

    rsd = False
    cuts = [range(1, 6), range(2, 15), range(9, 19)]
    parameters = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu', 'w', 'scale_factor']
    result_dir_root = '/home/ljy/BettiCurveCosmo/Emulator/model-fof-rsdz/mcmc_result/nwLH-Recovery' if rsd else '/home/ljy/BettiCurveCosmo/Emulator/model-fof/mcmc_result/nwLH-Recovery'

    emulator_path = '/home/ljy/BettiCurveCosmo/Emulator/model-fof-rsdz' if rsd else '/home/ljy/BettiCurveCosmo/Emulator/model-fof'
    var_emu_path = '/home/ljy/BettiCurveCosmo/Emulator/model-fof-rsdz/mcmc_result/nwLH-Recovery/var_emu.npy' if rsd else '/home/ljy/BettiCurveCosmo/Emulator/model-fof/mcmc_result/nwLH-Recovery/var_emu.npy'

    dataset = load_emulator_data('../Data/EmulatorData/nwLH_fof_emulator_dimensionless_rsdz_[(1,6),(2,15),(9,19)].bc') if rsd else load_emulator_data('../Data/EmulatorData/nwLH_fof_emulator_dimensionless_[(1,6),(2,15),(9,19)].bc')
    input = np.array([np.array(d[0]) for d in dataset], dtype=np.float32)
    output = [np.array([d[1][dim] for d in dataset], dtype=np.float32) for dim in range(3)]
    y_test = []
    for dim in range(3):
        _, X_, _, y_ = train_test_split(input, output[dim], test_size=0.1, random_state=42)
        y_test.append(y_)
    X_test = X_
    y_test = np.concatenate(y_test, axis=1)
    cov = np.cov(np.concatenate(cut_data(np.load('../Data/TestData/DimlessBC/fiducial_ZA-fof-rsdz-bc.npy'), cuts), axis=1).T) if rsd else np.cov(np.concatenate(cut_data(np.load('../Data/TestData/DimlessBC/fiducial_ZA-fof-bc.npy'), cuts), axis=1).T)

    # Replace covariance diagonal with Var_emu
    Var_full = np.load(var_emu_path)
    if Var_full.ndim == 2:
        if Var_full.shape[0] != Var_full.shape[1]:
            raise ValueError(f'Var_emu 2D must be square, got shape={Var_full.shape}.')
        if np.allclose(Var_full, np.diag(np.diag(Var_full))):
            Var_full = np.diag(Var_full)
        else:
            raise ValueError('Var_emu must be diagonal (2D) or 1D array.')
    elif Var_full.ndim != 1:
        raise ValueError(f'Var_emu must be 1D or diagonal 2D array, got ndim={Var_full.ndim}.')

    if Var_full.shape[0] != cov.shape[0]:
        raise ValueError(
            'Var_emu size must match covariance dimension. '
            f'Got {Var_full.shape[0]}, expected {cov.shape[0]}.'
        )

    cov = np.array(cov, copy=True)
    np.fill_diagonal(cov, Var_full)

    H_factor = (500-y_test.shape[1]-2)/(500-1)

    # Global load emulator and Cholesky factor
    EMULATOR = [joblib.load(os.path.join(emulator_path, f'autosklearn_model_{dim}.pkl')) for dim in range(3)]
    L_COV = chol_factor(cov)

    def Run_ultrenest(observed_data, result_dir, p):

        def log_likelihood(params):
            predicted_BettiCurve = np.concatenate([(e.predict(params.reshape(1,-1))).squeeze() for e in EMULATOR]) 
            residual = observed_data - predicted_BettiCurve
            y = np.linalg.solve(L_COV, residual)
            chi_squared = H_factor * float(np.dot(y, y))
            return -0.5 * chi_squared
        
        # Check if the result directory exists
        result_dir = os.path.join(result_dir, 'combined')
        os.makedirs(result_dir, exist_ok=True)
        logging(p, result_dir, emulator_path, cuts)

        # Define the sampler  
        sampler = ReactiveNestedSampler(parameters, log_likelihood, prior, log_dir=result_dir, resume=True)
        sampler.run(
        min_num_live_points=1600,
        dlogz=0.5,
        min_ess=400,
        frac_remain=0.1,
        )
        # Print and save results
        sampler.print_results()
        # Save plots
        save_plots(sampler, result_dir)
    
    index = unfinished_index(result_dir_root, 'combined', total=200)

    observe_data = [y_test[j,:].squeeze() for j in index]
    results_dir = [os.path.join(result_dir_root, f'{j}') for j in index]
    params = [X_test[j] for j in index]

    with Pool(n_process) as pool:
        pool.starmap(Run_ultrenest, zip(observe_data, results_dir, params))

    # for i, obs, res_dir, p in zip(index, observe_data, results_dir, params):
    #     os.makedirs(res_dir, exist_ok=True)
    #     Run_ultrenest(obs, res_dir, p)