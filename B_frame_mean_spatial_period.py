import numpy as np
import matplotlib.pyplot as plt
from Params import *  # Import parameters
import os  # For file path management
from nsOCT_maximum_spatial_period import *  # Import custom modules


def Mean_spatial_period_for_B_frame(nsOCT, bins, do_compute, do_save):
    """
    Computes and analyzes the mean spatial period for B-frames in a 3D nsOCT dataset.

    Args:
        nsOCT (ndarray): 3D nsOCT dataset.
        bins (int): Number of bins for histograms.
        do_compute (bool): Whether to perform the computations.
        do_save (bool): Whether to save the results.

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 12})
    if not do_compute:
        return

    # Load necessary parameters
    params = Params
    pixels_selected = params['pixels_selected']
    threshold_spatial_period = params['threshold_spatial_period']
    b_frames_to_process = params['B_frames_to_process']
    histograms_x_range = params['histograms_x_axis_range']
    histograms_y_limit = params['histograms_y_axis_limit']
    b_frame_histograms = params['B_frame_histograms']

    # Define the range of B-frames to process
    processing_B_frames = np.arange(b_frames_to_process[0], b_frames_to_process[1] + 1)

    # Apply threshold to nsOCT data
    nsOCT[nsOCT <= threshold_spatial_period] = np.NaN
    nsOCT = nsOCT[:, pixels_selected[0] - 1: pixels_selected[1], :]

    # Compute mean spatial period for averaged B-frame
    averaged_B_frame = np.nanmean(nsOCT, axis=0)
    flattened_B_frame = averaged_B_frame.flatten()
    mean_spatial_period = np.nanmean(flattened_B_frame)
    std_spatial_period = np.nanstd(flattened_B_frame)

    # Plot and save histogram for averaged B-frame
    plt.hist(flattened_B_frame, bins=bins, range=histograms_x_range, density=True,
             color='c', rwidth=5)
    plt.title("Mean of all B-frames")
    plt.xlabel("Spatial period (nm)")
    plt.ylabel("Amplitude (a.u)")
    plt.xlim(590,700)
    plt.ylim(0, histograms_y_limit)
    plt.tight_layout()
    b_frame_hist_dir = os.path.join(os.path.dirname(__file__), b_frame_histograms)
    plt.savefig(os.path.join(b_frame_hist_dir, "Averaged_B_frame.png"))
    plt.clf()

    # Save mean and standard deviation of averaged B-frame
    np.savetxt(os.path.join(b_frame_histograms, "B_frame_single_data_mean_and_std.txt"),
               [mean_spatial_period, std_spatial_period], fmt='%1.2f')

    # Compute mean and standard deviation for individual B-frames
    b_frame_means = []
    b_frame_stds = []

    for b_frame, histo_name in zip(nsOCT, processing_B_frames):
        flattened_frame = b_frame.flatten()
        b_frame_mean = np.nanmean(flattened_frame)
        b_frame_std = np.nanstd(flattened_frame)
        b_frame_means.append(b_frame_mean)
        b_frame_stds.append(b_frame_std)

        # Save histogram for each B-frame if required
        if do_save:
            averaged_along_A_lines = np.nanmean(b_frame, axis=1)
            plt.hist(averaged_along_A_lines, bins=bins, range=histograms_x_range, density=True,
                     color='g', rwidth=0.8)
            plt.title("Spatial period distribution for a B-frame")
            plt.xlabel("Spatial period (nm)")
            plt.ylabel("Amplitude (a.u)")
            plt.ylim(0, histograms_y_limit)
            plt.tight_layout()
            plt.savefig(os.path.join(b_frame_hist_dir, f"{histo_name}.png"))
            plt.clf()

    # Convert lists to arrays for further processing
    b_frame_means = np.array(b_frame_means)
    b_frame_stds = np.array(b_frame_stds)
    mean_from_means = np.mean(b_frame_means)
    std_from_means = np.std(b_frame_means)

    # Plot and save mean spatial period vs B-frame number
    plt.plot(processing_B_frames, b_frame_means, '.')
    plt.title("B-frames vs Mean Spatial Period")
    plt.xlabel("B-frame number")
    plt.ylabel("Mean Spatial Period for B-frame (nm)")
    plt.tight_layout()
    plt.savefig(os.path.join(b_frame_hist_dir, "mean_vs_B_frames.png"))
    plt.clf()

    # Save results
    np.save(os.path.join(b_frame_histograms, "B_frame_means.npy"), b_frame_means)
    np.savetxt(os.path.join(b_frame_histograms, "std_from_B_frame_means.txt"),
               [std_from_means], fmt='%1.2f')
    np.savetxt(os.path.join(b_frame_histograms, "mean_from_B_frame_means.txt"),
               [mean_from_means], fmt='%1.2f')
    np.savetxt(os.path.join(b_frame_histograms, "B_frame_stds.txt"),
               b_frame_stds, fmt='%1.2f')
    np.savetxt(os.path.join(b_frame_histograms, "B_frame_means.txt"),
               b_frame_means, fmt='%1.2f')
