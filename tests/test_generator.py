import pytest
import numpy as np
from raman_detection.generate import lorentz, exposure_for_snr


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