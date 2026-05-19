import numpy as np
from ultranest import ReactiveNestedSampler
import os, sys
sys.path.append('../..')
import joblib 
from Data.DataProcess import cut_data


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
    """Stable Cholesky for (near) SPD matrices; adds diagonal jitter if needed."""
    C = _sym(C)
    d = C.shape[0]

    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        pass

    jitter = float(jitter0)
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(C + jitter * np.eye(d))
        except np.linalg.LinAlgError:
            jitter *= 10.0

    C_psd = _psd_project(C)
    return np.linalg.cholesky(C_psd)


def _load_var_diag(var_path: str, expected_len: int, label: str, full_len: int = None):
    """Load variance diag from 1D or diagonal 2D.

    Accepts either:
    - exact selected length (expected_len), or
    - optional full length (full_len), then slices first expected_len entries.
    """
    var_full = np.load(var_path)
    if var_full.ndim == 2:
        if var_full.shape[0] != var_full.shape[1]:
            raise ValueError(f'{label}: 2D variance must be square, got shape={var_full.shape}.')
        if np.allclose(var_full, np.diag(np.diag(var_full))):
            var_full = np.diag(var_full)
        else:
            raise ValueError(f'{label}: variance must be diagonal (2D) or 1D array.')
    elif var_full.ndim != 1:
        raise ValueError(f'{label}: variance must be 1D or diagonal 2D array, got ndim={var_full.ndim}.')

    if var_full.shape[0] == expected_len:
        return np.asarray(var_full, dtype=float)

    if (full_len is not None) and (var_full.shape[0] == full_len):
        return np.asarray(var_full[:expected_len], dtype=float)

    if full_len is None:
        raise ValueError(f'{label}: expected diag length={expected_len}, got {var_full.shape[0]}.')

    raise ValueError(
        f'{label}: expected diag length={expected_len} (selected) or {full_len} (full), '
        f'got {var_full.shape[0]}.'
    )

def load_emulator(bc_path, pk_path):
    bc_model_files = {int(os.path.splitext(m)[0][-1]) : os.path.join(bc_path, m) for m in os.listdir(bc_path) if m.endswith('.pkl')}
    pk_model_files = {int(os.path.splitext(m)[0][-1]) : os.path.join(pk_path, m) for m in os.listdir(pk_path) if m.endswith('.pkl')}

    emulators = [[],[]]

    for i in range(0, len(bc_model_files)):
        emulator = joblib.load(bc_model_files[i])
        emulators[0].append(emulator)

    for i in range(0, len(pk_model_files)):
        emulator = joblib.load(pk_model_files[i])
        emulators[1].append(emulator)

    return emulators


def load_observed_data(bc_path, bc_cuts, pk_path, pk_multipoles):
    """
    pk_multipoles : The number of multipoles in the power spectrum. 1: pk0, 2: pk0+pk2, 3: pk0+pk2+pk4
    """
    observed_bc = cut_data(np.load(bc_path), bc_cuts)
    observed_bc = np.concatenate(observed_bc, axis=1)
    # print(observed_bc.shape)

    pk_data = np.load(pk_path)
    observed_pk = [pk_data[:,i+1] for i in range(pk_multipoles)]
    observed_pk = np.concatenate(observed_pk, axis=1)

    combined_data = np.concatenate([observed_bc, observed_pk], axis=1)
    # print(combined_data.shape)

    cov_matrix = np.cov(combined_data.T)
    H_factor = (combined_data.shape[0]-combined_data.shape[1]-2)/(combined_data.shape[0]-1)
    observed_data = np.mean(combined_data, axis=0)

    n_bc = observed_bc.shape[1]
    n_pk = observed_pk.shape[1]
    n_pk_full = None
    if pk_data.ndim == 3 and pk_data.shape[1] >= 2:
        # full multipoles available in file (excluding k column)
        n_pk_full = int((pk_data.shape[1] - 1) * pk_data.shape[2])

    return observed_data, cov_matrix, H_factor, n_bc, n_pk, n_pk_full

def save_plots(sampler, result_dir):
    """Save the result, trace, and corner plots."""
    os.makedirs(result_dir, exist_ok=True)
    sampler.plot()
    sampler.plot_trace()
    sampler.plot_corner()

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

def logging(bc_emulator_path, pk_emulator_path, observed_bc_path, observed_pk_path, var_emu_bc_path, var_emu_pk_path, result_dir, parameters, bc_cuts, pk_multipoles):
    log_path = os.path.join(result_dir, 'doc.txt')
    with open(log_path, 'w') as f:
        f.write('Emulator BC path: {}\n'.format(bc_emulator_path))
        f.write('Emulator Pk path: {}\n'.format(pk_emulator_path))
        f.write('Observed BC path: {}\n'.format(observed_bc_path))
        f.write('Observed Pk path: {}\n'.format(observed_pk_path))
        f.write('Var_emu BC path: {}\n'.format(var_emu_bc_path))
        f.write('Var_emu Pk path: {}\n'.format(var_emu_pk_path))
        f.write('Result directory: {}\n'.format(result_dir))
        f.write('Parameters: {}\n'.format(parameters))
        f.write('Cuts: {}\n'.format(bc_cuts))
        f.write('Multipoles: {}\n'.format(pk_multipoles))

def main(bc_emulator_path, pk_emulator_path, observed_bc_path, observed_pk_path, var_emu_bc_path, var_emu_pk_path, result_dir, parameters, bc_cuts, pk_multipoles):

    # Load the emulator
    emulator = load_emulator(bc_emulator_path, pk_emulator_path)

    # Load observed data
    observed_data, cov_matrix, H_factor, n_bc, n_pk, n_pk_full = load_observed_data(observed_bc_path, bc_cuts, observed_pk_path, pk_multipoles)

    # Build replaced-diagonal covariance: BC block from var_emu_BC, Pk block from var_emu_Pk
    C_tot = np.array(cov_matrix, copy=True)

    if var_emu_bc_path is not None:
        var_bc = _load_var_diag(var_emu_bc_path, expected_len=n_bc, label='var_emu_BC')
    else:
        var_bc = np.diag(cov_matrix)[:n_bc]

    if var_emu_pk_path is not None:
        var_pk = _load_var_diag(var_emu_pk_path, expected_len=n_pk, full_len=n_pk_full, label='var_emu_Pk')
    else:
        var_pk = np.diag(cov_matrix)[n_bc:n_bc + n_pk]

    var_diag = np.concatenate([var_bc, var_pk])
    np.fill_diagonal(C_tot, var_diag)
    L = chol_factor(C_tot)

    def log_likelihood(params):
        predicted_BC = np.concatenate([e.predict(params.reshape(1,-1)).squeeze() for e in emulator[0]])
        predicted_Pk = np.concatenate([e.predict(params.reshape(1,-1)).squeeze() for e in emulator[1]])
        predicted_curve = np.concatenate([predicted_BC, predicted_Pk]) 
        residual = observed_data - predicted_curve
        y = np.linalg.solve(L, residual)
        chi_squared = H_factor * float(np.dot(y, y))
        return -0.5 * chi_squared

    
    # Define the sampler
    os.makedirs(result_dir, exist_ok=True)
    logging(bc_emulator_path, pk_emulator_path, observed_bc_path, observed_pk_path, var_emu_bc_path, var_emu_pk_path, result_dir, parameters, bc_cuts, pk_multipoles) # Logging the parameters
    sampler = ReactiveNestedSampler(parameters, log_likelihood, prior, log_dir=result_dir, resume=True)
    
    sampler.run(
    min_num_live_points=400,
    # region_class=RobustEllipsoidRegion,
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

if __name__ == '__main__':
    rsd = False
    bc_emulator_path = '/home/ljy/BettiCurveCosmo/Emulator/model-fof-rsdz' if rsd else '/home/ljy/BettiCurveCosmo/Emulator/model-fof'
    pk_emulator_path = '/home/ljy/BettiCurveCosmo/PowSpec/model/model-fof-rsdz' if rsd else '/home/ljy/BettiCurveCosmo/PowSpec/model/model-fof'
    observed_bc_path = '/home/ljy/BettiCurveCosmo/Data/TestData/DimlessBC/fiducial_ZA-fof-rsdz-bc.npy' if rsd else '/home/ljy/BettiCurveCosmo/Data/TestData/DimlessBC/fiducial_ZA-fof-bc.npy'
    observed_pk_path = '/home/ljy/BettiCurveCosmo/PowSpec/data/fof/Phh_fiducial_ZA_[0.018,0.3]_rsdz.npy' if rsd else '/home/ljy/BettiCurveCosmo/PowSpec/data/fof/Phh_fiducial_ZA_[0.018,0.3].npy'
    var_emu_bc_path = '/home/ljy/BettiCurveCosmo/Emulator/model-fof-rsdz/mcmc_result/1/var_emu.npy' if rsd else '/home/ljy/BettiCurveCosmo/Emulator/model-fof/mcmc_result/1/var_emu.npy'
    var_emu_pk_path = '/home/ljy/BettiCurveCosmo/PowSpec/model/model-fof-rsdz/mcmc_result/1/var_rsd.npy' if rsd else '/home/ljy/BettiCurveCosmo/PowSpec/model/model-fof/mcmc_result/1/var_norsd.npy'
    result_dir = '/home/ljy/BettiCurveCosmo/MCMC/BC_Pk/result/fza-fof-rsdz' if rsd else '/home/ljy/BettiCurveCosmo/MCMC/BC_Pk/result/fza-fof'
    parameters = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu', 'w', 'scale_factor']
    bc_cuts = [range(1, 6), range(2, 15), range(9, 19)]
    pk_multipoles = 2 if rsd else 1

    main(bc_emulator_path, pk_emulator_path, observed_bc_path, observed_pk_path, var_emu_bc_path, var_emu_pk_path, result_dir, parameters, bc_cuts, pk_multipoles)