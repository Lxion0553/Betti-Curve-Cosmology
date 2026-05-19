import os
from typing import Optional
import numpy as np

# Globals for picklable scorer
_CHI2_COV: Optional[np.ndarray] = None
_CHI2_COV_PATH: Optional[str] = None


def set_chi2_cov(cov: np.ndarray) -> None:
    global _CHI2_COV
    _CHI2_COV = np.asarray(cov, dtype=np.float64)


def set_chi2_cov_path(path: str) -> None:
    """Set covariance file path for child processes (forkserver/spawn)."""
    global _CHI2_COV_PATH
    _CHI2_COV_PATH = path
    os.environ["CHI2_COV_PATH"] = path


def _get_chi2_cov() -> np.ndarray:
    global _CHI2_COV, _CHI2_COV_PATH
    if _CHI2_COV is not None:
        return _CHI2_COV

    if _CHI2_COV_PATH is None:
        _CHI2_COV_PATH = os.environ.get("CHI2_COV_PATH")

    if _CHI2_COV_PATH is None:
        raise ValueError("CHI2 covariance path not initialized. Call set_chi2_cov_path(path) first.")

    _CHI2_COV = np.load(_CHI2_COV_PATH)
    return _CHI2_COV


def reduced_chi2(y_true, y_pred, cov: np.ndarray) -> float:
    """Calculate reduced chi-squared for multi-output regression.

    y_true/y_pred: (n_samples, n_outputs) or (n_outputs,)
    cov: (n_outputs, n_outputs)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)

    cov = np.asarray(cov, dtype=np.float64)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Invalid covariance shape: {cov.shape}")

    if y_true.shape[1] != cov.shape[0]:
        raise ValueError(f"Shape mismatch: y has {y_true.shape[1]} outputs, cov is {cov.shape}")

    inv_cov = np.linalg.inv(cov)
    residual = y_true - y_pred
    chi2 = np.einsum("ij,jk,ik->i", residual, inv_cov, residual)
    # residual = y_true - y_pred
    # chi2 = residual**2 / cov.diagonal()
    dof = max(y_true.shape[1] - 1, 1)
    return float(np.mean(chi2 / dof))


def reduced_chi2_metric(y_true, y_pred) -> float:
    """Picklable scorer wrapper for Auto-sklearn multiprocessing."""

    cov = _get_chi2_cov()
    chi2 = reduced_chi2(y_true, y_pred, cov)

    return chi2

