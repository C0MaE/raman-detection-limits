# Logbook

What I did and learned, one entry per session. Entries start with `## YYYY-MM-DD · Title`;
they also appear on [comae.dev](https://comae.dev/log).

## 2026-10-19 · Peak shapes and exposure time

Set up the project and worked through the theory: the Raman bands of carbon nanotubes
(radial breathing mode around 100–300 cm⁻¹, D band near 1350 cm⁻¹, G band near 1580 cm⁻¹,
2D band near 2700 cm⁻¹) and counting noise. For $N$ detected photons the Poisson noise is
$\sqrt{N}$, and the counts grow linearly with the exposure time $t$, so

$$\text{SNR} = \frac{N}{\sqrt{N}} \propto \sqrt{t}.$$

With a background under the peak and read-out noise $\sigma_\text{read}$ this becomes
$\text{SNR} = a t / \sqrt{b t + \sigma_\text{read}^2}$, which `exposure_for_snr` solves for $t$.
`lorentz` gives the peak shape. Both have tests.

Learned along the way: tests should be deterministic, and how to use pytest parametrize

## 2026-11-19 · Peak sampling, background and noise

Each spectrum now gets its own random parameters. `sample_peaks` draws the position and
width of the G band uniformly from ranges in `config.yaml` (version 0 has only the G band).
`background` adds a constant background whose level relative to the peak,
$r = B(\tilde\nu_G)/A_G$, is drawn log-uniformly between 0.5 and 10, so that every factor
is equally likely. `add_noise` turns the smooth expected counts $\lambda$ into a measurement:
a Poisson draw for every pixel plus Gaussian read-out noise.

The test for `add_noise` checks mean and variance over 100,000 draws:

$$\langle N \rangle = \lambda, \qquad \operatorname{Var}(N) = \lambda + \sigma_\text{read}^2.$$

Learned along the way: variances of independent noise sources add, standard deviations
don't. With $\lambda = 400$ and $\sigma_\text{read} = 3$ the total noise is
$\sqrt{409} \approx 20.2$, not $23$.

## 2026-11-20 · Generating spectra and the dataset

`generate_spectrum` puts the pieces together: draw the peaks, build the background, compute
the exposure time for the target SNR, add the Lorentzian signal only if the spectrum is
labelled as containing one, then add noise. Everything up to the exposure time is drawn the
same way for both classes, so background level and noise cannot give the label away; a model
has to find the peak. `generate_dataset` creates 20,000 spectra, exactly half of them with
signal (shuffled with `rng.permutation`), with the SNR drawn log-uniformly between 0.5 and 20.
`scripts/make_dataset.py` saves them to `data/dataset.npz`, which is not committed because
config and seed reproduce it exactly.

The SNR test recomputes the SNR from the noise-free spectrum at the grid point closest to
the G band centre. It comes out up to 5 % low: the axis spacing is about 2.8 cm⁻¹, and the
centre almost never falls on a grid point. For the narrowest G band the nearest point still
reaches at least $1/(1 + (1.4/6)^2) \approx 0.95$ of the peak height.

Looking at the first spectra: at SNR 3 the peak is barely visible by eye, at SNR 10 it is
obvious. That grey zone is where the models will have to prove themselves.

Learned along the way: a random generator is a stream of numbers, so a reproducibility test
needs two generators with the same seed rather than one used twice. And a notebook keeps the
old version of a module until the kernel restarts, unless `%autoreload 2` is on.

## 2026-11-23 · A baseline without machine learning

Before training any model I need a yardstick: a simple detector that every model has to
beat. `detection_statistic` computes one number per spectrum in four steps. Savitzky–Golay
smoothing (window 11, order 2) lowers the pixel noise to about 0.46 of its original value
while keeping the peak shape. The median of each spectrum estimates the constant background,
which a few peak pixels barely move. The noise is estimated robustly with the scaled median
absolute deviation, $\hat\sigma = 1.4826 \cdot \operatorname{MAD}$. The statistic is the
highest point of the residual inside the G band window, in units of the noise:

$$s = \frac{\max_{\tilde\nu_i \in [1560,\,1620]} r_i}{\hat\sigma}.$$

Searching only where the G band can be keeps the look-elsewhere effect small: the maximum
of pure noise grows with the number of places you look.

The histogram of $s$ shows what I hoped for. Spectra without signal sit in a narrow band
between about 0.5 and 2.5; spectra with signal reach up to $s \approx 40$, roughly twice
their SNR, because smoothing removes noise faster than it flattens the peak. At low SNR the
two distributions overlap, and that overlap is the grey zone the project is about.

For the evaluation, `split_dataset` makes one stratified train/test split shared by all
models, `calibrate_threshold` puts the threshold at the 99th percentile of the
signal-free training spectra (1 % false alarms), and `efficiency_vs_snr` counts the detected
fraction per logarithmic SNR bin with binomial errors
$\sigma_\varepsilon = \sqrt{\varepsilon(1-\varepsilon)/n}$.

Learned along the way: `keepdims=True` keeps a per-spectrum value as a column so it
broadcasts across each row. Two histograms drawn with their own bins are not comparable.
And two silent traps in NumPy: indexing with an integer array of 0s and 1s selects
positions instead of filtering, and `np.histogram` with a number of bins picks different
edges for different inputs.

## 2026-11-24 · First efficiency curve

`scripts/run_baseline.py` runs the full evaluation for the first time: split the dataset,
compute $s$ for train and test, set the threshold on the signal-free training spectra,
check the false positive rate on the test set, and count the detected fraction per SNR bin.
The threshold lands at $s = 3.24$, and on the test set 1.2 % of the signal-free spectra
exceed it, close to the target of 1 % (the statistical spread with about 3000 spectra is
roughly 0.2 %). So the calibration carries over to unseen data.

The baseline detects about 10 % of the signals at SNR 0.6, half of them at
$\text{SNR}_{50} \approx 1.3$, 90 % at $\text{SNR}_{90} \approx 2.2$, and everything above
SNR 4 (rough log interpolation between bins; a proper sigmoid fit comes later). That is
better than my eye: at SNR 3 I could barely see the peak in a raw spectrum. The SNR is
defined by the peak height at a single pixel, but the peak spans several pixels, and
smoothing combines them, which gains roughly a factor of two.

Two things to improve later. Above SNR 4 every bin sits at 100 %, so half of the bins
cannot separate models; the interesting range is 0.5 to 3. And the binomial error
$\sqrt{\varepsilon(1-\varepsilon)/n}$ is zero at $\varepsilon = 1$, which overstates the
certainty; a Wilson or Clopper–Pearson interval would be more honest.

Learned along the way: the first run gave a threshold of 34 and 0 % false alarms on the
test set, because I had calibrated on all training spectra instead of only the
signal-free ones. The false positive check caught it immediately. A second slip, passing
the statistic values instead of the detection decisions to `efficiency_vs_snr`, slipped
past the `dtype=bool` conversion, so the function now checks shapes and that the mask
contains only 0/1 or True/False.