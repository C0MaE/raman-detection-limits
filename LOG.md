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
