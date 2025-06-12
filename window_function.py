import numpy as np
from scipy.special import i0

def select_window(apodiz_method, index_length, width, depth_index=None, window_band=None):
    """
    Selects a windowing method and returns a 2D window matrix of shape (index_length, width).

    Parameters:
        apodiz_method (str): Name of the windowing method.
        index_length (int): Number of depth indices (rows).
        width (int): Number of lateral positions (columns).
        depth_index (np.ndarray, optional): Required for Cosine, Gaussian, and super_Gaussian windows.
        window_band (float, optional): Required for Gaussian and super_Gaussian windows.

    Returns:
        np.ndarray: 2D window array.
    """
    if apodiz_method == 'Kaiser':
        print('Kaiser window')
        alpha = 2
        index = np.arange(index_length)
        v = np.pi * alpha * np.sqrt(1 - (2 * index / (index_length - 1) - 1) ** 2)
        window_1d = i0(v) / i0(np.pi * alpha)

    elif apodiz_method == 'Cosine':
        if depth_index is None:
            raise ValueError("depth_index must be provided for Cosine window")
        window_1d = np.cos(np.pi * depth_index / index_length)

    elif apodiz_method == 'super_Gaussian':
        if depth_index is None or window_band is None:
            raise ValueError("depth_index and window_band must be provided for super_Gaussian window")
        window_1d = np.exp(-(depth_index / window_band) ** 4)

    elif apodiz_method == 'Gaussian':
        if depth_index is None or window_band is None:
            raise ValueError("depth_index and window_band must be provided for Gaussian window")
        window_1d = np.exp(-(depth_index / window_band) ** 2)

    elif apodiz_method == 'Hamming':
        print('Hamming window')
        n = np.arange(1, index_length + 1)
        window_1d = 0.53836 - 0.46164 * np.cos(2 * np.pi * n / (index_length - 1))

    elif apodiz_method == 'Hann':
        print('Hann window')
        n = np.arange(1, index_length + 1)
        window_1d = 0.5 - 0.5 * np.cos(2 * np.pi * n / (index_length - 1))

    elif apodiz_method == 'Blackman':
        print('Blackman window')
        n = np.arange(1, index_length + 1)
        window_1d = (0.42 - 0.5 * np.cos(2 * np.pi * n / (index_length - 1)) +
                     0.08 * np.cos(4 * np.pi * n / (index_length - 1)))

    elif apodiz_method == 'Nuttall':
        print('Nuttall window')
        n = np.arange(1, index_length + 1)
        window_1d = (0.355768 - 0.487396 * np.cos(2 * np.pi * n / (index_length - 1)) +
                     0.144232 * np.cos(4 * np.pi * n / (index_length - 1)) -
                     0.012604 * np.cos(6 * np.pi * n / (index_length - 1)))

    elif apodiz_method == 'Blackman_Harris':
        print('Blackman_Harris window')
        n = np.arange(1, index_length + 1)
        window_1d = (0.35875 - 0.48829 * np.cos(2 * np.pi * n / (index_length - 1)) +
                     0.14128 * np.cos(4 * np.pi * n / (index_length - 1)) -
                     0.01168 * np.cos(6 * np.pi * n / (index_length - 1)))

    else:
        print('Rectangular window (default)')
        window_1d = np.ones(index_length)

    # Ensure column vector and repeat across width
    window = np.outer(window_1d, np.ones(width))
    return window
