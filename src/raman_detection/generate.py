import numpy as np

def generate_dataset(cfg, rng):
    return generate_spectrum()

def generate_spectrum():
    return

def sample_peaks():
    raise NotImplementedError

def lorentz(nu, center, amplitude, fwhm):
    """Lorentzian peak profile with height `amplitude` at `center`.

    Parameters
    ----------
    nu : float or ndarray
        Raman shift in cm^-1.
    center : float
        Peak position in cm^-1.
    amplitude : float
        Peak height at the center.
    fwhm : float
        Full width at half maximum in cm^-1.

    Returns
    -------
    float or ndarray
        Profile values, same shape as `nu`.
    """
    hwhm = fwhm / 2
    return amplitude * hwhm**2 / ((nu - center)**2 + hwhm**2)

def exposure_for_snr(snr, a, b, sigma_read):
    """Exposure time t that yields the target SNR.

    Solves SNR = t*a / sqrt(t*b + sigma_read**2) for t (positive root).

    Parameters
    ----------
    snr : float
        Target signal-to-noise ratio.
    a : float
        Peak height (count rate) of the reference band.
    b : float
        Background count rate at the reference band position.
    sigma_read : float
        Read-out noise in counts.

    Returns
    -------
    float
        Exposure time t, in the same time units as the rates a and b.
    """
    snr_sqr, sigma_sqr, a_sqr, b_sqr = snr**2, sigma_read**2, a**2, b**2
    root = np.sqrt(snr_sqr**2 * b_sqr + 4*a_sqr*snr_sqr*sigma_sqr)
    return (snr_sqr*b + root)/(2*a_sqr)