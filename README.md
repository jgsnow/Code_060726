# Angular Spectrum Analysis for Marine Magnetic Anomaly Classification and Estimation of their Dominant Orientations 

This repository contains Python modules for extracting, characterizing, and classifying directional signatures from two-dimensional anomaly fields using Fourier-domain angular spectrum analysis.

The workflow transforms 2D anomaly maps into angular magnitude spectra, groups moving-window spectra using unsupervised clustering, identifies dominant anomaly orientations from full-window (map scale) anomaly spectra and quantifies directional uncertainty.

---

## Repository Structure

### angular_spectra.py

Functions for extracting angular magnitude spectra from anomaly maps.

| Function | Description |
|-----------|-------------|
| `apply_window()` | Applies a two-dimensional Hann taper to reduce edge effects prior to Fourier transformation. |
| `fft_and_magnitude()` | Computes the 2-D Fourier transform and corresponding magnitude spectrum. |
| `polar_transform()` | Converts the Fourier magnitude spectrum from Cartesian to polar coordinates. |
| `find_dominant_orientations()` | Computes angular spectra and identifies dominant orientation peaks. |

---

### cluster_spectra.py

Functions for unsupervised classification of angular magnitude spectra.

| Function | Description |
|-----------|-------------|
| `cluster_angular_spectra()` | Performs K-means clustering of angular spectra and returns cluster labels and centroids. |

---

### orientation_estimation.py

Functions for quantifying orientation uncertainty of regional magnetic anomaly maps.

| Function | Description |
|-----------|-------------|
| `estimate_orientation_statistics()` | Computes dominant orientation, peak magnitude, magnitude-weighted RMS angular spread, and the corresponding 95% directional confidence interval. |

---

## Method Overview

The workflow consists of four steps:

1. Apply a Hann taper to the entire anomaly map (or to moving windows).
2. Compute the 2-D Fourier magnitude spectrum.
3. Transform the spectrum to polar coordinates and integrate amplitudes along radial profiles to obtain an angular magnitude spectrum.
4. Clustar moving window angular spectra using K-Means.
5. Identify dominant orientations and estimate directional uncertainty from map-scale spetra.

The angular magnitude spectrum is defined as

M(θ) = Σ A(r, θ)

where A(r, θ) is the Fourier-domain amplitude at radius r and orientation θ.

Dominant orientations correspond to peaks in the angular spectrum.

Directional uncertainty is quantified using a normalized magnitude-weighted root-mean-square (RMS) angular deviation around the dominant orientation.

---

## Example Workflow

```python
from angular_spectra import (
    apply_window,
    fft_and_magnitude,
    polar_transform,
    find_dominant_orientations
)

from orientation_estimation import estimate_orientation_statistics
from cluster_spectra import cluster_angular_spectra

# Apply taper
windowed = apply_window(anomaly_map)

# Fourier magnitude spectrum
magnitude = fft_and_magnitude(windowed)

# Polar transform
polar_img, theta, *_ = polar_transform(magnitude)

# Angular spectrum
dominant_angles, angular_profile, peaks = find_dominant_orientations(
    polar_img,
    theta
)

# K-Means Clustering (moving window spectra)
labels, centroids = cluster_angular_spectra(
    angular_spectra,
    n_clusters=2
)

# Orientation statistics (entire anomaly map)
theta_peak, peak_magnitude, sigma_theta, ci95 = \
    estimate_orientation_statistics(
        theta,
        angular_profile
    )
```

## Dependencies

- numpy
- scipy
- scikit-learn


## References

Blackman, R. B., & Tukey, J. W. (1958).
The Measurement of Power Spectra.

Bendat, J. S., & Piersol, A. G. (2010).
Random Data: Analysis and Measurement Procedures.

Fisher, N. I. (1993).
Statistical Analysis of Circular Data.

Mardia, K. V., & Jupp, P. E. (1999).
Directional Statistics.


## Citation

If you use this software, please cite:

Author et al. (2026), Nature Methods, DOI: xxx

and the corresponding Zenodo archive:

DOI: https://doi.org/10.5281/zenodo.20436903