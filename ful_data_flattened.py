import numpy as np
from nsOCT_maximum_spatial_period import *

def full_data_flattened(nsoct, bins, do_compute=True, do_save=True):
    """
    Processes a 3D nsOCT dataset by flattening the selected region,
    generating a histogram, and calculating statistical metrics.

    Args:
        nsOCT (np.ndarray): 3D nsOCT dataset.
        bins (int or array-like): Number of bins or bin edges for the histogram.
        do_compute (bool): Whether to perform the computations.
        do_save (bool): Whether to save histogram and statistics to disk.

    Returns:
        dict or None: Dictionary with mean and std if computed, else None.
    """
    if not do_compute:
        return None

    params = Params
    threshold_spatial_period = params['threshold_spatial_period']
    pixels_selected = params['pixels_selected']
    full_data_flattened_dir = params['full_data_flattened']
    histograms_x_range = params['histograms_x_axis_range']
    histograms_y_limit = params['histograms_y_axis_limit']

    # Select relevant pixel range from the 3D nsOCT data
    try:
        three_d_nsOCT = nsoct[:, pixels_selected[0] - 1: pixels_selected[1], :]
    except Exception as e:
        print(f"Error selecting pixel range: {e}")
        return None

    print('The dimension of the selected 3D nsOCT data is', np.shape(three_d_nsOCT))

    # Flatten the 3D data into a 1D array
    flat_data = three_d_nsOCT.flatten()
    print('The dimension of flattened data is', np.shape(flat_data))

    # Generate histogram (before thresholding)
    plt.hist(flat_data, bins=bins, range=histograms_x_range, density=True, color='g', rwidth=0.8)
    plt.xlabel("Spatial period (nm)")
    plt.ylabel("Amplitude (a.u.)")
    plt.ylim(0, histograms_y_limit)
    plt.tight_layout()

    # Prepare output directory
    script_dir = os.path.dirname(__file__)
    full_flat_dir = os.path.join(script_dir, full_data_flattened_dir)
    if do_save:
        os.makedirs(full_flat_dir, exist_ok=True)
        plt.savefig(os.path.join(full_flat_dir, "Full_data_histo.png"))
    plt.clf()  # Clear the plot for future use

    # Filter the data by removing values below the threshold
    flat_data_masked = np.where(flat_data < threshold_spatial_period, np.nan, flat_data)

    # Calculate the mean and standard deviation of the valid data
    full_data_mean = np.nanmean(flat_data_masked)
    full_data_std = np.nanstd(flat_data_masked)

    if do_save:
        np.savetxt(os.path.join(full_flat_dir, "Full_data_mean.txt"),
                   np.array([full_data_mean]), fmt='%1.4f')
        np.savetxt(os.path.join(full_flat_dir, "Full_data_std.txt"),
                   np.array([full_data_std]), fmt='%1.4f')
        print("Full data mean and standard deviation saved successfully.")

    return {
        "mean": full_data_mean,
        "std": full_data_std
    }