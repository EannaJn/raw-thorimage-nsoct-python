import numpy as np
import matplotlib.pyplot as plt
from Params import *  # Import parameter configurations
import os  # For file path management
from nsOCT_maximum_spatial_period import *  # Import custom modules


def Mean_spatial_period_for_enface(nsOCT, bins, do_compute, do_save):
    """
    Computes and analyzes the mean spatial period for en-face slices of a 3D nsOCT dataset.

    Args:
        nsOCT (ndarray): 3D nsOCT dataset.
        bins (int): Number of bins for histograms.
        do_compute (bool): Whether to perform the computations.
        do_save (bool): Whether to save the results.

    Returns:
        None
    """
    if not do_compute:
        return

    # Load necessary parameters from Params
    params = Params
    num_pixels = params['Number_of_pixels']
    depth_in_mm = params['Depth_in_mm']
    pixels_selected = params['pixels_selected']
    a_lines_to_process = params['A_lines_to_process']
    b_frames_to_process = params['B_frames_to_process']
    threshold_spatial_period = params['threshold_spatial_period']
    histograms_x_range = params['histograms_x_axis_range']
    histograms_y_limit = params['histograms_y_axis_limit']
    enface_histograms = params['enface_histograms']

    # Select the number of pixels
    selected_pixels = (pixels_selected[1] - pixels_selected[0]) + 1

    # Apply threshold to nsOCT data
    nsOCT[nsOCT <= threshold_spatial_period] = np.NaN

    # Rearrange axes for enface processing
    enface_nsOCT = np.swapaxes(np.swapaxes(nsOCT, 1, 0), 2, 1)
    enface_nsOCT = enface_nsOCT[
        pixels_selected[0] - 1: pixels_selected[1],
        a_lines_to_process[0] - 1: a_lines_to_process[1],
        b_frames_to_process[0] - 1: b_frames_to_process[1]
    ]

    # Compute the mean spatial period along pixels axis
    averaged_pixels = np.nanmean(nsOCT, axis=1)
    flat_enface = averaged_pixels.flatten()
    enface_mean = np.nanmean(flat_enface)
    enface_std = np.nanstd(flat_enface)

    # Plot histogram for all enface slices
    plt.hist(flat_enface, bins=bins, range=histograms_x_range, density=True,
             color='b', rwidth=0.8)
    plt.title("Mean spatial period for all enface images")
    plt.xlabel("Spatial period (nm)")
    plt.ylabel("Amplitude (a.u)")
    plt.ylim(0, histograms_y_limit)
    plt.tight_layout()
    plt.clf()  # Clear the plot for the next use

    # Save mean and standard deviation
    single_mean_std = np.array([enface_mean, enface_std])

    # Analyze individual enface slices
    enface_numbers = np.arange(pixels_selected[0], pixels_selected[1] + 1)
    enface_depths = np.linspace(
        (depth_in_mm * 2 / num_pixels) * pixels_selected[0],
        (depth_in_mm * 2 / num_pixels) * pixels_selected[1],
        selected_pixels
    )
    enface_means = []
    enface_stds = []

    for enface_slice, histo_name in zip(enface_nsOCT, enface_numbers):
        # Flatten and compute mean/std for the slice
        flattened_slice = enface_slice.flatten()
        slice_mean = np.nanmean(flattened_slice)
        slice_std = np.nanstd(flattened_slice)

        enface_means.append(slice_mean)
        enface_stds.append(slice_std)

        # Save histograms for individual slices if required
        if do_save:
            plt.hist(np.nanmean(enface_slice, axis=0), bins=bins, density=True,
                     range=histograms_x_range, rwidth=0.8, color='y')
            plt.title("Spatial period distribution for each enface")
            plt.xlabel("Spatial period (nm)")
            plt.ylabel("Amplitude")
            plt.ylim(0, histograms_y_limit)
            plt.tight_layout()
            plt.savefig(os.path.join(enface_histograms, f"{histo_name}.png"))
            plt.clf()

    # Convert lists to arrays for summary statistics
    enface_means = np.asarray(enface_means)
    enface_stds = np.asarray(enface_stds)
    mean_from_means = np.mean(enface_means)
    std_from_means = np.std(enface_means)

    # Plot mean spatial period vs enface depths
    plt.plot(enface_depths, enface_means, '.')
    plt.title("Mean spatial period vs enface depths")
    plt.xlabel("Enface depths")
    plt.ylabel("Mean spatial period (nm)")
    plt.tight_layout()
    plt.savefig(os.path.join(enface_histograms, "mean_vs_enface.png"))
    plt.clf()

    # Save computed data
    if do_save:
        np.save(os.path.join(enface_histograms, "en_face_means.npy"), enface_means)
        np.savetxt(os.path.join(enface_histograms, "mean_from_enface_means.txt"),
                   [mean_from_means], fmt='%1.2f')
        np.savetxt(os.path.join(enface_histograms, "std_from_enface_means.txt"),
                   [std_from_means], fmt='%1.2f')
        np.savetxt(os.path.join(enface_histograms, "enface_stds.txt"),
                   enface_stds, fmt='%1.2f')
        np.savetxt(os.path.join(enface_histograms, "enface_means.txt"),
                   enface_means, fmt='%1.1f')
