import numpy as np


def estimate_orientation_statistics(theta, angular_spectrum, n_angles=180):
    """
    Estimate dominant orientation and angular uncertainty
    from an angular magnitude spectrum.

    Parameters
    ----------
    theta : ndarray
        Angular coordinates in radians.

    angular_spectrum : ndarray
        Angular magnitude spectrum.

    n_angles : int, optional
        Number of angular samples retained for analysis.
        Default is 180.

    Returns
    -------
    dominant_orientation_deg : float
        Peak orientation angle in degrees.

    peak_amplitude : float
        Magnitude of the dominant spectral peak.

    angular_rms_deg : float
        Magnitude-weighted RMS angular spread (degrees).

    rms_95_interval_deg : float
        Approximate 95% angular uncertainty estimated as ±1.96 × RMS.

    Notes
    -----
    Angular spread is computed using a magnitude-weighted
    root-mean-square (RMS) deviation relative to the dominant
    orientation, assuming 180° directional symmetry.

    The metric is defined on a discretized angular spectrum and
    approximates a continuous angular dispersion measure. Accuracy
    depends on angular sampling resolution.

    References
    ----------
    Bendat, J. S., & Piersol, A. G. (2010). 
    Random data: analysis and measurement procedures. 
    John Wiley & Sons.

    Mardia, K. V. (1999). 
    Directional statistics and shape analysis. 
    Journal of applied Statistics, 26, 949-957. 
    DOI: https://doi.org/10.1080/02664769921954
    """

    # Convert angles to degrees and retain half-spectrum
    theta_deg = np.degrees(theta[:n_angles])

    spectrum = np.asarray(angular_spectrum[:n_angles])

    if theta_deg.shape != spectrum.shape:
        raise ValueError(
            "theta and angular_spectrum must have matching dimensions."
        )

    # Dominant orientation
    peak_index = np.argmax(spectrum)

    dominant_orientation_deg = theta_deg[peak_index]

    # Angular difference with 180° symmetry
    angular_difference = np.abs(theta_deg - dominant_orientation_deg)

    angular_difference = np.minimum(
        angular_difference,
        180 - angular_difference
    )

    # Magnitude-weighted RMS angular spread
    angular_rms_deg = np.sqrt(
        (1 / n_angles) *
        np.sum(spectrum * angular_difference**2)
        / np.sum(spectrum)
    )

    # Approximate 95% uncertainty interval
    rms_95_interval_deg = 1.96 * angular_rms_deg

    peak_amplitude = spectrum[peak_index]

    return (
        dominant_orientation_deg,
        peak_amplitude,
        angular_rms_deg,
        rms_95_interval_deg
    )
