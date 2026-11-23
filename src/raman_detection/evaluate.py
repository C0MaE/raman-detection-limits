from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset(X, y, snr, test_size, seed):
    """Stratified train/test split, identical for every model.

    Parameters
    ----------
    X: ndarray, shape (n_spectra, n_features)
        The Raman spectra.
    y: ndarray, shape (n_spectra,)
        The labels.
    snr: ndarray, shape (n_spectra,)
        The signal-to-noise ratio.
    test_size: float
        The proportion of the dataset to include in the test split.
    seed: int
        The seed for the random number generator.

    Returns
    -------
    X_tr, X_te, y_tr, y_te, snr_te, snr_tr: tuple of ndarrays
        The train and test sets.
    """
    X_tr, X_te, y_tr, y_te, snr_tr, snr_te = train_test_split(X, y, snr, test_size=test_size, stratify=y, random_state=seed)
    return X_tr, X_te, y_tr, y_te, snr_tr, snr_te

def efficiency_vs_snr(snr, detected, bins):
    """Detection efficiency and its binomial error per SNR bin.

    Parameters
    ----------
    snr: ndarray
        The signal-to-noise ratio.
    detected: ndarray of bool
        Boolean mask, True where a spectrum was detected.
    bins: ndarray
        Bin edges (1D, increasing). Use the same edges for every model.

    Returns
    -------
    eps, sigma_eps: tuple of ndarrays
        The detection efficiency and its binomial error per SNR bin.
    """
    bins = np.asarray(bins)
    if bins.ndim != 1:
        raise ValueError("bins must be an array of bin edges")

    detected = np.asarray(detected, dtype=bool)

    n, _ = np.histogram(snr, bins=bins)
    k, _ = np.histogram(snr[detected], bins=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        eps = k / n
        sigma_eps = np.sqrt(eps * (1 - eps) / n)

    return eps, sigma_eps