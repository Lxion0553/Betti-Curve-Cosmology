import argparse
import numpy as np
from ultranest import ReactiveNestedSampler
from ultranest.mlfriends import RobustEllipsoidRegion
from typing import List, Optional, Sequence, Tuple
from scipy.stats import poisson, norm
from scipy.special import gammaln
import os, sys
sys.path.append('..')

import joblib 
from Emulator.GPRemulator import optimizer, custom_optimizer
from Data.DataProcess import cut_data

PowSpec = False

def parse_range(range_str):
    start, end = range_str.split('-')
    return range(int(start), int(end))

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


def prepare_poisson_copula(C: np.ndarray, jitter: float = 1e-10) -> Tuple[np.ndarray, float]:
    """Build Gaussian-copula precision helper from covariance/correlation structure."""
    C = _sym(C)
    std = np.sqrt(np.clip(np.diag(C), 1e-30, None))
    R = C / np.outer(std, std)
    R = _sym(R)
    np.fill_diagonal(R, 1.0)

    for _ in range(8):
        R_try = R + jitter * np.eye(R.shape[0])
        sign, logdetR = np.linalg.slogdet(R_try)
        if sign > 0:
            invR_minus_I = np.linalg.inv(R_try) - np.eye(R.shape[0])
            return invR_minus_I, float(logdetR)
        jitter *= 10.0

    R_psd = _psd_project(R)
    sign, logdetR = np.linalg.slogdet(R_psd)
    if sign <= 0:
        raise ValueError('Failed to construct positive-definite correlation matrix for Poisson copula.')
    invR_minus_I = np.linalg.inv(R_psd) - np.eye(R_psd.shape[0])
    return invR_minus_I, float(logdetR)

def load_emulator(emulator_path: str, dim: int):
    """Load autosklearn (or GPR) emulators.

    The directory is expected to contain files ending with .pkl and with the last
    character before extension being the integer dimension index.

    dim = -1 : load all emulators
    dim = -2 : load emulators 1 and 2
    dim >= 0 : load that dimension
    """
    model_files = {
        int(os.path.splitext(m)[0][-1]): os.path.join(emulator_path, m)
        for m in os.listdir(emulator_path)
        if m.endswith('.pkl')
    }

    def _load_one(i: int):
        return joblib.load(model_files[i])


    if dim < 0:
        if dim == -2:
            return [_load_one(1), _load_one(2)]
        
        if dim == -1:
            return [_load_one(i) for i in sorted(model_files.keys())]
        
        raise ValueError(f'Unsupported dim={dim}')
    
    return _load_one(dim)

def _cuts_lengths(cuts: Sequence[range]) -> List[int]:
    return [len(list(c)) for c in cuts]


def _select_indices_from_cuts(cuts: Sequence[range], dim: int) -> np.ndarray:
    lens = _cuts_lengths(cuts)
    offsets = np.cumsum([0] + lens[:-1])

    if dim == -1:
        return np.arange(int(np.sum(lens)))

    if dim == -2:
        idx1 = np.arange(offsets[1], offsets[1] + lens[1])
        idx2 = np.arange(offsets[2], offsets[2] + lens[2])
        return np.concatenate([idx1, idx2])

    if dim >= 0:
        return np.arange(offsets[dim], offsets[dim] + lens[dim])

    raise ValueError(f'Unsupported dim={dim}')

def load_observed_data(data_path: str, cuts: Optional[Sequence[range]], dim: int):
    data = np.load(data_path)

    if PowSpec:
        # Supported PowSpec input formats (common cases):
        # 1) data.ndim == 3:
        #    a) (n_realizations, n_bins, >=3) with columns [k, P0, P2, ...]
        #    b) (n_realizations, n_fields, n_bins) e.g. (500, 4, 23) with fields [k, P0, P2, ...]
        # 2) data.ndim == 2: (n_realizations, 2*n_bins) where first half is P0 bins and second half is P2 bins
        # dim meaning in PowSpec mode:
        #   dim=0 -> P0
        #   dim=1 -> P2
        #   dim=-1 -> concatenate [P0, P2]
        if dim not in (-1, 0, 1):
            raise ValueError('PowSpec mode only supports dim=0, dim=1, or dim=-1 (combine P0+P2).')

        if data.ndim == 3:
            # Detect whether the second axis is bins or fields.
            # - If last axis is small (<=10) and middle is large, it's likely (n_real, n_bins, n_cols).
            # - If middle axis is small (<=10) and last axis is bins, it's likely (n_real, n_fields, n_bins).
            if data.shape[1] <= 10 and data.shape[2] >= 3:
                # (n_real, n_fields, n_bins)  e.g. (500, 4, 23)
                if data.shape[1] < 3:
                    raise ValueError(f'PowSpec array must contain [k, P0, P2] fields; got shape={data.shape}')
                # convention: field0=k, field1=P0, field2=P2
                X0 = np.asarray(data[:, 1, :], dtype=float)
                X1 = np.asarray(data[:, 2, :], dtype=float)
            else:
                # (n_real, n_bins, n_cols)
                if data.shape[2] < 3:
                    raise ValueError(f'PowSpec 3D array must have >=3 columns [k, P0, P2], got shape={data.shape}')
                X0 = np.asarray(data[:, :, 1], dtype=float)
                X1 = np.asarray(data[:, :, 2], dtype=float)
        elif data.ndim == 2:
            if data.shape[1] % 2 != 0:
                raise ValueError(
                    'PowSpec 2D array is expected to have even number of columns (P0 bins + P2 bins). '
                    f'Got shape={data.shape}.'
                )
            p = data.shape[1] // 2
            X0 = np.asarray(data[:, :p], dtype=float)
            X1 = np.asarray(data[:, p:], dtype=float)
        else:
            raise ValueError(f'Unsupported PowSpec array ndim={data.ndim}, shape={data.shape}')

        if dim == -1:
            X = np.concatenate([X0, X1], axis=1)
        else:
            X = X0 if dim == 0 else X1

        C_fid = np.cov(X.T)
        n, p = X.shape
        H_factor = (n - p - 2) / (n - 1)
        y = np.mean(X, axis=0)
        return y, C_fid, float(H_factor)

    if cuts is None:
        raise ValueError('cuts must be provided (e.g. -c 1-6 2-15 9-19).')

    blocks = cut_data(data, cuts)

    if dim < 0:
        if dim == -1:
            X = np.concatenate(blocks, axis=1)
        elif dim == -2:
            X = np.concatenate([blocks[1], blocks[2]], axis=1)
        else:
            raise ValueError(f'Unsupported dim={dim}')

        C_fid = np.cov(X.T)
        n, p = X.shape
        H_factor = (n - p - 2) / (n - 1)
        y = np.mean(X, axis=0)
        return y, C_fid, float(H_factor)

    # single dimension
    X = blocks[dim]
    C_fid = np.cov(X.T)
    n, p = X.shape
    H_factor = (n - p - 2) / (n - 1)
    y = np.mean(X, axis=0)
    return y, C_fid, float(H_factor)

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

def logging(emulator_path, observed_data_path, var_emu_path, cov_path, result_dir, parameters, cuts, likelihood_model):
    log_path = os.path.join(result_dir, 'doc.txt')
    with open(log_path, 'w') as f:
        f.write(f"emulator_path: {emulator_path}\n")
        f.write(f"observed_data_path: {observed_data_path}\n")
        f.write(f"var_emu_path: {var_emu_path}\n")
        f.write(f"cov_path: {cov_path}\n")
        f.write(f"likelihood_model: {likelihood_model}\n")
        f.write(f"result_dir: {result_dir}\n")
        f.write(f"parameters: {parameters}\n")
        f.write(f"cuts: {cuts}\n")

def main(    
    emulator_path: str,
    observed_data_path: str,
    var_emu_path: str,
    cov_path: Optional[str],
    result_dir: str,
    parameters: List[str],
    dim: int,
    cuts: Optional[Sequence[range]],
    likelihood_model: str,
    poisson_scale: float,
    use_hartlap: bool,
    powspec: bool,):

    global PowSpec
    PowSpec = bool(powspec)

    # Load emulators
    emulator = load_emulator(emulator_path, dim)

    # Load observed data (mean vector) and fid covariance
    observed_data, C_fid, H_factor = load_observed_data(observed_data_path, cuts, dim)
    p_obs = int(np.asarray(observed_data).shape[0])

    # Optional: load external covariance and override C_fid
    if cov_path is not None:
        Cov_full = np.load(cov_path)
        if Cov_full.ndim != 2:
            raise ValueError(f'Cov must be a 2D array, got ndim={Cov_full.ndim}.')
        if Cov_full.shape[0] != Cov_full.shape[1]:
            raise ValueError(f'Cov must be square, got shape={Cov_full.shape}.')

        if Cov_full.shape[0] == p_obs:
            C_fid = Cov_full
        else:
            if PowSpec:
                if dim not in (-1, 0, 1):
                    raise ValueError('PowSpec Cov selection requires dim=0, dim=1, or dim=-1.')
                if dim == -1:
                    raise ValueError(
                        'PowSpec dim=-1 expects Cov dimension to match combined size. '
                        f'Got Cov shape={Cov_full.shape}, expected {(p_obs, p_obs)}.'
                    )
                pbin = p_obs
                if Cov_full.shape[0] == 2 * pbin:
                    start = 0 if dim == 0 else pbin
                    idx = np.arange(start, start + pbin)
                    C_fid = Cov_full[np.ix_(idx, idx)]
                else:
                    raise ValueError(
                        'PowSpec expects Cov to be either (pbin,pbin) for the chosen dim, '
                        f'or (2*pbin,2*pbin) for both dims. Got Cov shape={Cov_full.shape}, pbin={pbin}.'
                    )
            else:
                if cuts is None:
                    raise ValueError('cuts must be provided to select Cov for BC.')
                total_len = int(np.sum([len(list(c)) for c in cuts]))
                if Cov_full.shape[0] != total_len:
                    raise ValueError(
                        'BC expects Cov dimension to match total combined size. '
                        f'Got Cov shape={Cov_full.shape}, expected {(total_len, total_len)}.'
                    )
                idx = _select_indices_from_cuts(cuts, dim)
                C_fid = Cov_full[np.ix_(idx, idx)]

        C_fid = _sym(C_fid)

    # Load Var_emu (variance) from file
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

    # Accept either full diagonal (combined) or already cut diagonal
    if Var_full.shape[0] == p_obs:
        Var_diag = Var_full
    else:
        if PowSpec:
            if dim not in (-1, 0, 1):
                raise ValueError('PowSpec Var selection requires dim=0, dim=1, or dim=-1.')
            if dim == -1:
                raise ValueError(
                    'PowSpec dim=-1 expects Var_emu diagonal to match combined dimension exactly. '
                    f'Got diag shape={Var_full.shape}, expected {(p_obs,)}.'
                )
            # dim=0/1: allow full diag (2*pbin,) and slice
            pbin = p_obs
            if Var_full.shape[0] == 2 * pbin:
                start = 0 if dim == 0 else pbin
                idx = np.arange(start, start + pbin)
                Var_diag = Var_full[idx]
            else:
                raise ValueError(
                    'PowSpec expects Var_emu diagonal to be either (pbin,) for the chosen dim, '
                    f'or (2*pbin,) for both dims. Got diag shape={Var_full.shape}, pbin={pbin}.'
                )
        else:
            if cuts is None:
                raise ValueError('cuts must be provided to select Var_emu diagonal for BC.')
            total_len = int(np.sum([len(list(c)) for c in cuts]))
            if Var_full.shape[0] != total_len:
                raise ValueError(
                    'BC expects Var_emu diagonal to match total combined dimension. '
                    f'Got diag shape={Var_full.shape}, expected {(total_len,)}.'
                )
            idx = _select_indices_from_cuts(cuts, dim)
            Var_diag = Var_full[idx]

    # Build diagonal covariance from diag elements
    Var_full = Var_diag
    C_tot = np.array(C_fid, copy=True)
    np.fill_diagonal(C_tot, Var_diag)

    # Poisson-like likelihood for normalized/non-integer data:
    # scale to pseudo-counts, then use continuous Poisson extension via gammaln.
    if poisson_scale <= 0:
        raise ValueError(f'poisson_scale must be > 0, got {poisson_scale}.')
    observed_eff = np.clip(np.asarray(observed_data, dtype=float) * poisson_scale, 0.0, None)
    invR_minus_I, logdetR = prepare_poisson_copula(C_tot)

    # likelihood
    def predict_curve(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta).reshape(1, -1)
        if dim < 0:
            return np.concatenate([e.predict(theta).squeeze() for e in emulator])
        return emulator.predict(theta).squeeze()
    
    def log_likelihood(params: np.ndarray) -> float:
        theta = np.asarray(params).ravel()

        predicted = predict_curve(theta)
        residual = observed_data - predicted

        if likelihood_model == 'gaussian':
            L = chol_factor(C_tot)
            y = np.linalg.solve(L, residual)
            quad = float(np.dot(y, y))
            quad_term = (H_factor * quad) if use_hartlap else quad
            return -0.5 * quad_term

        lam = np.clip(np.asarray(predicted, dtype=float) * poisson_scale, 1e-12, None)
        ll_marginal = float(np.sum(observed_eff * np.log(lam) - lam - gammaln(observed_eff + 1.0)))

        # PIT for continuous pseudo-counts via Normal approximation with continuity correction.
        z_cc = (observed_eff + 0.5 - lam) / np.sqrt(lam)
        u = norm.cdf(z_cc)
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        z = norm.ppf(u)
        ll_copula = -0.5 * (logdetR + float(z @ invR_minus_I @ z))
        return ll_marginal + ll_copula
    

    # Set up output directory
    if dim == -1:
        result_dir = os.path.join(result_dir, 'combined')
    elif dim == -2:
        result_dir = os.path.join(result_dir, 'combined12')
    else:
        result_dir = os.path.join(result_dir, f'dim{dim}')

    os.makedirs(result_dir, exist_ok=True)
    logging(emulator_path, observed_data_path, var_emu_path, cov_path, result_dir, parameters, cuts, likelihood_model)

    # Define the sampler
    sampler = ReactiveNestedSampler(parameters, log_likelihood, prior, log_dir=result_dir, resume=True)
    
    # Run the sampler
    sampler.run(
    min_num_live_points=1200,
    dlogz=0.5,
    min_ess=400,
    cluster_num_live_points=100,
    frac_remain=0.5, # higher value for simple posterior 
    )


    # Print and save results
    sampler.print_results()

    # Save plots
    save_plots(sampler, result_dir)


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run MCMC sampling with an emulator.")
    
    parser.add_argument(
        '-e',
        '--emulator', 
        type=str, 
        required=True, 
        help='Path to the trained emulator model (e.g., gpr1.pkl)'
    )
    parser.add_argument(
        '-o',
        '--observed-data', 
        type=str, 
        required=True, 
        help='Path to the observed data (e.g., fiducial_ZA_0.5bc.npy)'
    )
    parser.add_argument(
        '-ve',
        '--var-emu', 
        type=str, 
        required=True, 
        help='Path to var_emu.npy (Diagonal variance of emulator, 1d array or diagonal 2d).'
    )
    parser.add_argument(
        '--cov-path',
        type=str,
        default=None,
        help='Optional path to covariance matrix (must be full 2D). If provided, overrides C_fid.'
    )
    parser.add_argument(
        '-r',
        '--result-dir', 
        type=str, 
        default='results', 
        help='Directory to save the results and plots (default: results/)'
    )
    parser.add_argument(
        '-p',
        '--parameters', 
        type=str, 
        nargs='+',
        required=True, 
        help='List of parameters to sample (e.g., Om Ob h ns s8 Mnu w scale_factor)'
    )
    parser.add_argument(
        '-d',
        '--dimension', 
        type=int, 
        required=True, 
        help='Dimension of the observed data to use for MCMC sampling. dim = -1 : Combine all the emulators. dim = -2 : Combine dim1 and dim2 emulators.'
    )
    parser.add_argument(
        '-c',
        '--cuts', 
        type=parse_range, 
        nargs='+',
        required=False, 
        default=None,
        help='List of cuts to apply to the observed data (e.g., 1-6 3-12 13-18)'
    )
    parser.add_argument('--no-hartlap', action='store_true', help='Disable Hartlap factor on quadratic term.')
    parser.add_argument('--powspec', action='store_true', help='Treat observed data as PowSpec realizations (dim: 0=P0, 1=P2, -1=combine P0+P2).')
    parser.add_argument(
        '--likelihood-model',
        type=str,
        default='gaussian',
        choices=['gaussian', 'poisson'],
        help='Likelihood used in nested sampling: gaussian (default) or poisson (Poisson marginals + Gaussian copula).'
    )
    parser.add_argument(
        '--poisson-scale',
        type=float,
        default=1.0,
        help='Scale factor mapping normalized data to pseudo-counts for poisson likelihood (must be > 0).'
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Run the main function with provided arguments
    main(        
        emulator_path=args.emulator,
        observed_data_path=args.observed_data,
        var_emu_path=args.var_emu,
        cov_path=args.cov_path,
        result_dir=args.result_dir,
        parameters=args.parameters,
        dim=args.dimension,
        cuts=args.cuts,
        likelihood_model=args.likelihood_model,
        poisson_scale=args.poisson_scale,
        use_hartlap=(not args.no_hartlap),
        powspec=bool(args.powspec),)
