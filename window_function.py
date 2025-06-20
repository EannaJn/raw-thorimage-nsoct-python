import numpy as np
from scipy.special import i0

def select_window(apodize_method, index_length, width, depth_index=None, window_band=None):
    """
    Selects a windowing method and returns a 2D window matrix of shape (index_length, width).

    Parameters:
        apodizee_method (str): Name of the windowing method.
        index_length (int): Number of depth indices (rows).
        width (int): Number of lateral positions (columns).
        depth_index (np.ndarray, optional): Required for Cosine, Gaussian, and super_Gaussian windows.
        window_band (float, optional): Required for Gaussian and super_Gaussian windows.

    Returns:
        np.ndarray: 2D window array.
    """
    if apodize_method == 'Kaiser':
        alpha = 2
        index = np.arange(index_length)
        v = np.pi * alpha * np.sqrt(1 - (2 * index / (index_length - 1) - 1) ** 2)
        window_1d = i0(v) / i0(np.pi * alpha)

    elif apodize_method == 'Cosine':
        if depth_index is None:
            raise ValueError("depth_index must be provided for Cosine window")
        window_1d = np.cos(np.pi * depth_index / index_length)

    elif apodize_method == 'super_Gaussian':
        if depth_index is None or window_band is None:
            raise ValueError("depth_index and window_band must be provided for super_Gaussian window")
        window_1d = np.exp(-(depth_index / window_band) ** 4)

    elif apodize_method == 'Gaussian':
        if depth_index is None or window_band is None:
            raise ValueError("depth_index and window_band must be provided for Gaussian window")
        window_1d = np.exp(-(depth_index / window_band) ** 2)

    elif apodize_method == 'Hamming':
        n = np.arange(1, index_length + 1)
        window_1d = 0.53836 - 0.46164 * np.cos(2 * np.pi * n / (index_length - 1))

    elif apodize_method == 'Hann':
        n = np.arange(1, index_length + 1)
        window_1d = 0.5 - 0.5 * np.cos(2 * np.pi * n / (index_length - 1))

    elif apodize_method == 'Blackman':
        n = np.arange(1, index_length + 1)
        window_1d = (0.42 - 0.5 * np.cos(2 * np.pi * n / (index_length - 1)) +
                     0.08 * np.cos(4 * np.pi * n / (index_length - 1)))

    elif apodize_method == 'Nuttall':
        n = np.arange(1, index_length + 1)
        window_1d = (0.355768 - 0.487396 * np.cos(2 * np.pi * n / (index_length - 1)) +
                     0.144232 * np.cos(4 * np.pi * n / (index_length - 1)) -
                     0.012604 * np.cos(6 * np.pi * n / (index_length - 1)))

    elif apodize_method == 'Blackman_Harris':
        n = np.arange(1, index_length + 1)
        window_1d = (0.35875 - 0.48829 * np.cos(2 * np.pi * n / (index_length - 1)) +
                     0.14128 * np.cos(4 * np.pi * n / (index_length - 1)) -
                     0.01168 * np.cos(6 * np.pi * n / (index_length - 1)))

    else:
        window_1d = np.ones(index_length)

    # Ensure column vector and repeat across width
    window = np.outer(window_1d, np.ones(width))
    return window
