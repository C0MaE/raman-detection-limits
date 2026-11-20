import pytest
import numpy as np
import copy

from raman_detection.generate import (lorentz,
                                      exposure_for_snr,
                                      sample_peaks,
                                      background,
                                      generate_spectrum,
                                      generate_dataset
                                      )

CFG = {'peaks': {'G': {'center_range': [1580, 1600],
                       'fwhm_range': [12.0, 30.0],
                       'amplitude': 1.0}},
       'background': {'ratio_range': [0.5, 10.0]},
       'noise': {'sigma_read': 3.0},
       'axis': {'nu_min': 100.0, 'nu_max': 3000.0, 'n_points': 1024},
       'dataset': {'n_spectra': 20000, 'signal_fraction': 0.5, 'snr_range': [0.5, 20.0]}
       }

def test_lorentz_amplitude_at_center():
    value = lorentz(nu=1590.0, center=1590.0, amplitude=2.5, fwhm=20.0)
    assert value == pytest.approx(2.5)


def test_lorentz_half_max_at_half_width():
    amplitude, center, fwhm = 2.5, 1590.0, 20.0
    x_target = center - fwhm / 2

    value = lorentz(nu=x_target, center=center, amplitude=amplitude, fwhm=fwhm)
    assert value == pytest.approx(amplitude / 2)

@pytest.mark.parametrize("offset", [1.0, 5.0, 10.0, 30.0, 200.0])
def test_lorentz_symmetric(offset):
    center = 1590.0
    left = lorentz(nu=center - offset, center=center, amplitude=2.5, fwhm=20.0)
    right = lorentz(nu=center + offset, center=center, amplitude=2.5, fwhm=20.0)
    assert left == pytest.approx(right)

def test_lorentz_numpy_length_preservation():
    amplitude, center, fwhm = 2.5, 1590.0, 20.0
    nu = np.linspace(100, 3000, 10000)
    value = lorentz(nu=nu, center=center, amplitude=amplitude, fwhm=fwhm)
    assert nu.shape == value.shape

@pytest.mark.parametrize("a, b, snr, expected", [
    (1, 4, 3, 36),
    (1, 2, 10, 200),
    (2, 4, 3, 9),
])
def test_exposure_without_read_noise(a, b, snr, expected):
    t = exposure_for_snr(snr=snr, a=a, b=b, sigma_read=0)
    assert t == pytest.approx(expected)

def test_snr_exposure_time_with_smooth_noise():
    a, b, sigma_read, snr = 1, 3, 4, 2
    value = exposure_for_snr(a=a, b=b, sigma_read=sigma_read, snr=snr)
    assert value == pytest.approx(16)

def test_snr_exposure_time_with_noise():
    a, b, sigma_read, snr = 1, 4, 3, 3
    value = exposure_for_snr(a=a, b=b, sigma_read=sigma_read, snr=snr)
    assert value == pytest.approx(38.1246)

@pytest.mark.parametrize("snr", [0.5, 1, 3, 20])
def test_exposure_roundtrip(snr):
    a, b, sigma_read = 1, 4, 3
    t = exposure_for_snr(snr=snr, a=a, b=b, sigma_read=sigma_read)
    snr_back = t * a / np.sqrt(t * b + sigma_read**2)
    assert snr_back == pytest.approx(snr)

def test_sample_peaks():
    rng = np.random.default_rng(1)
    peak = sample_peaks(rng=rng, cfg=CFG)[0]

    c_lo, c_hi = CFG['peaks']['G']['center_range']
    w_lo, w_hi = CFG['peaks']['G']['fwhm_range']

    assert peak['name'] == 'G'
    assert c_lo <= peak['center'] <= c_hi
    assert w_lo <= peak['fwhm'] <= w_hi
    assert peak['amplitude'] == pytest.approx(1.0)

def test_background_shape_and_constant():
    nu = np.linspace(100, 3000, 50)
    bg, b_ref = background(nu=nu, rng=np.random.default_rng(0), cfg=CFG, nu_ref=1590.0)
    assert bg.shape == nu.shape
    assert np.all(bg == b_ref)

def test_background_scales_with_reference_amplitude():
    nu = np.linspace(100, 3000, 50)
    _, b_ref = background(nu=nu, rng=np.random.default_rng(0), cfg=CFG, nu_ref=1590.0, a_ref=2.0)
    assert 0.5 <= b_ref / 2.0 <= 10.0

def test_background_ratio_log_uniform():
    lo, hi = CFG["background"]["ratio_range"]
    nu = np.linspace(100, 3000, 5)
    rng = np.random.default_rng(0)
    r = np.array([background(nu=nu, rng=rng, cfg=CFG, nu_ref=1590.0)[1] for _ in range(10_000)])
    assert np.all((r >= lo) & (r <= hi))
    assert np.median(r) == pytest.approx(np.sqrt(lo * hi), rel=0.05)

def test_generate_spectrum_shape():
    nu = np.linspace(100, 3000, 50)
    rng = np.random.default_rng(0)
    counts, meta = generate_spectrum(nu=nu, rng=rng, cfg=CFG, has_signal=True, snr=3)
    assert counts.shape == nu.shape

def test_generate_spectrum_same_seed():
    nu = np.linspace(100, 3000, 50)
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    counts1, meta1 = generate_spectrum(nu=nu, rng=rng1, cfg=CFG, has_signal=True, snr=3)
    counts2, meta2 = generate_spectrum(nu=nu, rng=rng2, cfg=CFG, has_signal=True, snr=3)
    assert np.array_equal(counts1, counts2)

def test_generate_spectrum_no_peak():
    nu = np.linspace(100, 3000, 50)
    rng = np.random.default_rng(0)
    counts, meta = generate_spectrum(nu=nu, rng=rng, cfg=CFG, has_signal=False, snr=3)
    assert (meta["expected"] == meta["t"] * meta["b_ref"]).all()

@pytest.mark.parametrize("snr", [0.5, 1, 3, 10])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generate_spectrum_snr_matches_target(snr, seed):
    axis = CFG["axis"]
    nu = np.linspace(axis["nu_min"], axis["nu_max"], axis["n_points"])
    rng = np.random.default_rng(seed)
    sigma_read = CFG["noise"]["sigma_read"]

    counts, meta = generate_spectrum(nu=nu, rng=rng, cfg=CFG, has_signal=True, snr=snr)

    g = [p for p in meta["peaks"] if p["name"] == "G"][0]
    idx = np.argmin(np.abs(nu - g["center"]))
    t, b = meta["t"], meta["b_ref"]
    snr_measured = (meta["expected"][idx] - t * b) / np.sqrt(t * b + sigma_read**2)

    # Grid offset of up to half a pixel lowers the sampled peak by at most ~5 %
    assert 0.94 * snr <= snr_measured <= snr * (1 + 1e-9)

def small_cfg(n=100):
    cfg = copy.deepcopy(CFG)
    cfg["dataset"]["n_spectra"] = n
    return cfg

def test_generate_dataset_shapes():
    cfg = small_cfg()
    X, y, snr, nu = generate_dataset(cfg, np.random.default_rng(0))
    assert X.shape == (100, cfg["axis"]["n_points"])
    assert y.shape == snr.shape == (100,)
    assert nu.shape == (cfg["axis"]["n_points"],)

def test_generate_dataset_signal_fraction():
    cfg = small_cfg()
    X, y, snr, nu = generate_dataset(cfg, np.random.default_rng(0))
    assert np.sum(y) == 50

def test_generate_dataset_snr_range():
    cfg = small_cfg()
    X, y, snr, nu = generate_dataset(cfg, np.random.default_rng(0))
    assert np.min(snr) >= cfg["dataset"]["snr_range"][0]
    assert np.max(snr) <= cfg["dataset"]["snr_range"][1]

def test_generate_dataset_same_seed():
    cfg = small_cfg()
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    X1, y1, snr1, nu1 = generate_dataset(cfg, rng1)
    X2, y2, snr2, nu2 = generate_dataset(cfg, rng2)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(snr1, snr2)
    assert np.array_equal(nu1, nu2)

def test_generate_dataset_even_distribution():
    cfg = small_cfg()
    X, y, snr, nu = generate_dataset(cfg, np.random.default_rng(0))
    assert not (y[:50] == 1).all()