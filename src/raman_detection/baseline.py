from scipy.signal import savgol_filter
import numpy as np

def smooth(X, cfg):
    """Savitzky-Golay smoothing of every spectrum (row) in X.

    Parameters
    ----------
    X: ndarray, shape (n_spectra, n_channels)
        The spectra to smooth.
    cfg: dict
        The configuration dictionary.

    Returns
    -------
    smooth: ndarray, shape (n_spectra, n_channels)
        The smoothed spectra.
    """
    baseline = cfg["baseline"]
    smooth = savgol_filter(X, window_length=baseline["savgol_window"], polyorder=baseline["savgol_polyorder"])
    return smooth

def subtract_background(X):
    """Remove the background from every spectrum (row) in X.

    Parameters
    ----------
    X: ndarray, shape (n_spectra, n_channels)
        The spectra to remove the background from.

    Returns
    -------
    X: ndarray, shape (n_spectra, n_channels)
        The spectra with the background removed.
    """
    background = np.median(X, axis=1, keepdims=True)
    return X - background

def estimate_noise(residual):
    """Robust noise estimate (scaled MAD) for every spectrum, shape (n_spectra, 1).

    Parameters
    ----------
    residual: ndarray, shape (n_spectra, n_channels)
        The residual spectra to estimate the noise from.

    Returns
    -------
    sigma_hat: ndarray, shape (n_spectra, 1)
        The noise estimate for every spectrum.
    """
    center = np.median(residual, axis=1, keepdims=True)
    mad = np.median(np.abs(residual - center), axis=1, keepdims=True)
    return 1.4826 * mad

def detection_statistic(X, nu, cfg):
    """Return the test statistic s for every spectrum in X (shape (n_spectra,)).

    Parameters
    ----------
    X: ndarray, shape (n_spectra, n_channels)
        The spectra to compute the test statistic for.
    nu: ndarray, shape (n_channels,)
        The wavenumbers corresponding to the spectra.
    cfg: dict
        The configuration dictionary.

    Returns
    -------
    s: ndarray, shape (n_spectra,)
        The test statistic for every spectrum.
    """
    lo, hi = cfg["baseline"]["search_range"]

    X_smooth = smooth(X, cfg)                    # step 1
    residual = subtract_background(X_smooth)     # step 2
    sigma_hat = estimate_noise(residual)         # step 3, shape (n_spectra, 1)

    mask = (nu >= lo) & (nu <= hi)               # step 4
    peak = residual[:, mask].max(axis=1, keepdims=True)
    return (peak / sigma_hat).ravel()

def calibrate_threshold(s_neg, fpr):
    """Return the threshold above which a fraction fpr of s_neg lies.

    Parameters
    ----------
    s_neg: ndarray, shape (n_spectra,)
        The test statistic for every spectrum.
    fpr: float
        The false positive rate.

    Returns
    -------
    threshold: float
        The threshold above which a fraction fpr of s_neg lies.
    """
    threshold = np.quantile(s_neg, 1 - fpr)
    return threshold