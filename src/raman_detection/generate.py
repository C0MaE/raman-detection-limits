import numpy as np

def generate_dataset(cfg, rng):
    axis = cfg["axis"]
    dataset = cfg["dataset"]
    n_spectra, n_points = dataset["n_spectra"], axis["n_points"]
    lo, hi = dataset["snr_range"]
    nu = np.linspace(axis["nu_min"], axis["nu_max"], n_points)
    n_signals = int(round(n_spectra * dataset["signal_fraction"]))
    y = rng.permutation(np.concatenate([np.ones(n_signals, dtype=int), np.zeros(n_spectra - n_signals, dtype=int)]))
    snr = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n_spectra))

    X = np.empty((n_spectra, n_points), dtype=np.float32)
    for i in range(n_spectra):
        counts, _ = generate_spectrum(nu=nu, rng=rng, cfg=cfg, has_signal=y[i], snr=snr[i])
        X[i] = counts

    return X, y, snr, nu

def generate_spectrum(nu, rng, cfg, has_signal, snr):
    """Generate a spectrum with a given SNR.

    Parameters
    ----------
    nu : ndarray
        Wavenumbers.
    rng : numpy.random.Generator
        Random number generator, created once by the caller.
    cfg : dict
        Configuration.
    has_signal : bool
        Whether the spectrum has signal.
    snr : float
        Signal-to-noise ratio.

    Returns
    -------
    counts : ndarray
        Counts.
    meta : dict
        Metadata.
    """
    sigma_read = cfg["noise"]["sigma_read"]
    peaks = sample_peaks(rng, cfg)
    g = [d for d in peaks if d.get("name") == "G"][0]
    C_G, A_G = g["center"], g["amplitude"]
    bg, b_ref = background(nu=nu, rng=rng, cfg=cfg, nu_ref=C_G, a_ref=A_G)
    t = exposure_for_snr(snr=snr, a=A_G, b=b_ref, sigma_read=sigma_read)
    signal = np.zeros_like(nu)
    if has_signal:
        for p in peaks: signal += lorentz(nu=nu, center=p["center"], amplitude=p["amplitude"], fwhm=p["fwhm"])
    expected = t * (signal + bg)
    counts = add_noise(rng, expected, sigma_read)
    meta = {"peaks": peaks, "t": t, "b_ref": b_ref, "expected": expected}
    return counts, meta

def add_noise(rng, expected, sigma_read):
    """Add Poisson noise and Gaussian read-out noise to the expected counts.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator, created once by the caller.
    expected : ndarray
        Expected counts.
    sigma_read : float
        Standard deviation of the Gaussian read-out noise.

    Returns
    -------
    noisy : ndarray
        Noisy counts.
    """
    poisson = rng.poisson(lam=expected, size=expected.shape)
    gauss = rng.normal(0, sigma_read, size=expected.shape)
    return poisson + gauss


def background(nu, rng, cfg, nu_ref, a_ref = 1.0):
    """Constant fluorescence background scaled to the reference peak.

    Draws the ratio r = B(nu_ref) / a_ref log-uniformly from
    cfg["background"]["ratio_range"] and returns a flat background
    of height r * a_ref.

    Parameters
    ----------
    nu: ndarray
        Raman shift axis in cm^-1.
    rng: numpy.random.Generator
        Random number generator, created once by the caller.
    cfg: dict
        Configuration; uses cfg["background"]["ratio_range"].
    nu_ref: float
        Position of the reference peak (G band) in cm^-1.
        Not used for the constant background, needed later for
        position-dependent backgrounds.
    a_ref: float, optional
        Height of the reference peak (G band). Default 1.0.

    Returns
    -------
    bg: ndarray
        Background values, same shape as `nu`.
    b_ref: float
        Background value at the reference position, B(nu_ref).
    """

    lo, hi = cfg["background"]["ratio_range"]
    r = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    b_ref = a_ref * r

    return np.full(nu.shape, b_ref), b_ref

def sample_peaks(rng, cfg):
    """Generate a random set of peaks.

    Parameters
    ----------
    rng: numpy.random.Generator
        randomized seeded number generator
    cfg: dict
        configuration file parameters

    Returns
    -------
    ndarray
        Peak positions in range
    """
    array = []

    for name, values in cfg['peaks'].items():
        center = rng.uniform(values['center_range'][0], values['center_range'][1])
        fwhm = rng.uniform(values['fwhm_range'][0], values['fwhm_range'][1])
        amplitude = values['amplitude']
        array.append({'name': name, 'center': center, 'fwhm': fwhm, 'amplitude': amplitude})

    return np.array(array, dtype=object)

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