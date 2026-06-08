import numpy as np
from scipy.signal import windows


def apply_window(img, window_type="hann", taper_fraction=0.2):
    """
    Apply a 2D apodization window to reduce Fourier edge artifacts.

    The function supports either a single 2D image with shape
    ``(height, width)`` or a batch of images with shape
    ``(batch, height, width)``.

    Parameters
    ----------
    img : ndarray
        Input image or image batch.

    window_type : str, optional
        Type of window function. Currently only ``"hann"`` is supported.

    taper_fraction : float, optional
        Fraction of the image dimension used for edge tapering
        on each side. Must satisfy ``0 <= taper_fraction < 0.5``.

    Returns
    -------
    ndarray
        Windowed image array with the same shape as ``img``.

    Notes
    -----
    A stretched Hann window is used, consisting of:
        - a cosine taper near the image boundaries,
        - a central flat region with unit amplitude.

    This reduces spectral leakage and boundary discontinuities
    prior to Fourier-domain analysis.

    Example
    -------
    Apply Hann apodization to a single image:

    import numpy as np
    img = np.random.randn(256, 256)
    windowed = apply_window(img, window_type="hann", taper_fraction=0.2)

    References
    ----------
    Blackman, R. B., & Tukey, J. W. (1958).
    The Measurement of Power Spectra.
    Dover Publications.
    """

    if img.ndim not in (2, 3):
        raise ValueError(
            "Input array must have shape (H, W) or (N, H, W)."
        )

    if not (0 <= taper_fraction < 0.5):
        raise ValueError(
            "taper_fraction must satisfy 0 <= taper_fraction < 0.5."
        )

    if window_type.lower() != "hann":
        raise NotImplementedError(
            f"Unsupported window type: '{window_type}'."
        )

    # Extract spatial dimensions
    if img.ndim == 2:
        height, width = img.shape
    else:
        _, height, width = img.shape

    def stretched_hann(length, taper_fraction):
        """
        Construct a 1D stretched Hann window with a flat center.
        """

        taper = int(length * taper_fraction)

        # No taper requested
        if taper == 0:
            return np.ones(length)

        flat = length - 2 * taper

        hann = windows.hann(2 * taper)

        return np.concatenate([
            hann[:taper],
            np.ones(flat),
            hann[taper:]
        ])

    # Generate separable 2D window
    wy = stretched_hann(height, taper_fraction)
    wx = stretched_hann(width, taper_fraction)

    window_2d = np.outer(wy, wx)

    # Broadcasting automatically handles batch dimensions
    return img * window_2d


import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


def fft_and_magnitude(img):
    """
    Compute 2D Fourier transform magnitude spectrum.

    The function computes the FFT and shifts the zero-frequency
    component to the center of the spectrum.

    Parameters
    ----------
    img : ndarray (H, W)
        Input 2D image.

    Returns
    -------
    magnitude : ndarray (H, W)
        Shifted magnitude spectrum |FFT|.
    """
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    return np.abs(fft_shift)


def polar_transform(image, center=None, radius=None, angles=360, radii=177):
    """
    Map a 2D image from Cartesian to polar coordinates.

    Typically applied to FFT magnitude spectra to analyze
    angular energy distribution.

    Parameters
    ----------
    image : ndarray (H, W)
        Input 2D image (e.g., FFT magnitude spectrum).

    center : tuple of int, optional
        (x, y) center of transformation. Defaults to image center.

    radius : float, optional
        Maximum radial extent. Defaults to largest inscribed radius.

    angles : int, optional
        Number of angular samples (0–2π).

    radii : int, optional
        Number of radial samples.

    Returns
    -------
    polar : ndarray (radii, angles)
        Polar-sampled image.

    theta : ndarray
        Angular coordinates (radians).

    theta_grid, r, r_grid, x, y : ndarray
        Grids used for interpolation (for reproducibility/debugging).

    Notes
    -----
    Uses bilinear interpolation via scipy.ndimage.map_coordinates.
    Angular convention: theta increases counterclockwise.
    """
    if center is None:
        center = (image.shape[1] // 2, image.shape[0] // 2)

    if radius is None:
        radius = min(
            center[0], center[1],
            image.shape[1] - center[0],
            image.shape[0] - center[1]
        )

    r = np.linspace(0, radius, radii)
    theta = np.linspace(0, 2 * np.pi, angles, endpoint=False)

    theta_grid, r_grid = np.meshgrid(theta, r)

    x = r_grid * np.cos(theta_grid) + center[0]
    y = center[1] - r_grid * np.sin(theta_grid)

    coords = np.vstack((y.ravel(), x.ravel()))

    polar = map_coordinates(image, coords, order=1, mode="nearest")
    polar = polar.reshape((radii, angles))

    return polar, theta, theta_grid, r, r_grid, x, y


def find_dominant_orientations(polar_img, theta, threshold=0.2, endpoint=None):
    """
    Identify dominant الاتجاهs (orientations) from polar FFT spectrum.

    The angular energy profile is computed by integrating over radius,
    followed by peak detection.

    Parameters
    ----------
    polar_img : ndarray (radii, angles)
        Polar-transformed magnitude spectrum.

    theta : ndarray
        Angular coordinate array (radians).

    threshold : float, optional
        Relative peak height threshold (fraction of global max).

    endpoint : int, optional
        Limit search to half-spectrum (default: full angular range).

    Returns
    -------
    dominant_angles_deg : ndarray
        Detected dominant orientations in degrees.

    angular_profile : ndarray
        Radially summed angular energy distribution.

    peaks : ndarray
        Indices of detected peaks.
    """
    angular_profile = polar_img.sum(axis=0)

    if endpoint is None:
        endpoint = len(theta)

    peaks, _ = find_peaks(
        angular_profile[:endpoint],
        height=angular_profile.max() * threshold
    )

    dominant_angles_deg = np.degrees(theta[peaks])

    return dominant_angles_deg, angular_profile, peaks

