import numpy as np
import matplotlib.pyplot as plt
from Params import *  # Import parameters from the Params module
import os  # For file and directory path handling
from nsOCT_maximum_spatial_period import *  # Import custom OCT processing module


def full_data_flattened(nsOCT, bins, do_compute):
    """
    Processes a 3D nsOCT dataset by flattening the entire dataset,
    generating a histogram, and calculating statistical metrics.

    Args:
        nsOCT (ndarray): 3D nsOCT dataset.
        bins (int): Number of bins for the histogram.
        do_compute (bool): Whether to perform the computations.

    Returns:
        None
    """
    # Load parameters
    params = Params
    threshold_spatial_period = params['threshold_spatial_period']
    pixels_selected = params['pixels_selected']
    full_data_flattened_dir = params['Full_data_flattened']
    histograms_x_range = params['histograms_x_axis_range']
    histograms_y_limit = params['histograms_y_axis_limit']

    if not do_compute:
        return

    # Step 1: Select relevant pixel range from the 3D nsOCT data
    three_d_nsOCT = nsOCT[:, pixels_selected[0] - 1: pixels_selected[1], :]
    print('The dimension of the selected 3D nsOCT data is', np.shape(three_d_nsOCT))

    # Step 2: Flatten the 3D data into a 1D array
    flat_data = three_d_nsOCT.flatten()
    print('The dimension of flattened data is', np.shape(flat_data))

    # Step 3: Generate and save a histogram of the flattened data
    plt.hist(flat_data, bins=bins, range=histograms_x_range, density=True,
             color='g', rwidth=0.8)
    plt.xlabel("Spatial period (nm)")
    plt.ylabel("Amplitude (a.u)")
    plt.ylim(0, histograms_y_limit)
    plt.tight_layout()

    # Create the directory if it doesn't exist and save the histogram
    script_dir = os.path.dirname(__file__)
    full_flat_dir = os.path.join(script_dir, full_data_flattened_dir)
    os.makedirs(full_flat_dir, exist_ok=True)
    plt.savefig(os.path.join(full_flat_dir, "Full_data_histo.png"))
    plt.clf()  # Clear the plot for future use

    # Step 4: Filter the data by removing values below the threshold
    flat_data[flat_data < threshold_spatial_period] = np.NaN

    # Step 5: Calculate the mean and standard deviation of the valid data
    full_data_mean = np.array([np.nanmean(flat_data)])
    full_data_std = np.array([np.nanstd(flat_data)])

    # Step 6: Save the calculated metrics to text files
    np.savetxt(os.path.join(full_flat_dir, "Full_data_mean.txt"),
               full_data_mean, fmt='%1.2f')
    np.savetxt(os.path.join(full_flat_dir, "Full_data_std.txt"),
               full_data_std, fmt='%1.2f')

    print("Full data mean and standard deviation saved successfully.")
