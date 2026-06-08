import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def cluster_angular_spectra(
    angular_spectra,
    n_clusters=2,
    n_angles=180,
    normalize=True,
    random_state=42
):
    """
    Cluster moving-window angular spectra using K-means.

    Parameters
    ----------
    angular_spectra : ndarray (n_samples, n_angles_total)
        Input angular spectra. Each row corresponds to the
        angular spectrum of a moving window.

    n_clusters : int, optional
        Number of K-means clusters.

    n_angles : int, optional
        Number of angular samples retained from the spectrum.
        Default is 180.

    normalize : bool, optional
        If True, apply Min-Max normalization prior to clustering.

    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    labels : ndarray (n_samples,)
        Cluster label assigned to each spectrum.

    centroids : ndarray (n_clusters, n_angles)
        Cluster centroid spectra in normalized feature space.

    Notes
    -----
    Angular spectra are reshaped into 2D feature vectors prior to 
    clustering. Min-Max normalization is applied independently to 
    each angular spectra to ensure equal weighting across angular bins.

    References
    ----------
    45.	MacQueen, J. (1967). 
    Multivariate observations. 
    Proceedings of the 5th Berkeley symposium on mathematical statistics and probability, 1, 281-297. Oakland, CA, USA: 
    University of California press.
    """

    # Ensure array format
    angular_spectra = np.asarray(angular_spectra)

    if angular_spectra.ndim != 2:
        raise ValueError(
            "Input angular_spectra must have shape "
            "(n_samples, n_angles)."
        )

    # Retain selected angular range
    spectra = angular_spectra[:, :n_angles]

    # Normalize features
    if normalize:
        scaler = MinMaxScaler()
        spectra_normalized = scaler.fit_transform(spectra)

    # K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        random_state=random_state
    )

    kmeans.fit(spectra_normalized)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids

