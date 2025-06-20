import numpy as np
import matplotlib.pyplot as plt
import os
from nsOCT_maximum_spatial_period import *

def mean_spatial_period_for_b_frame(nsoct, bins, do_compute=True, do_save=True):
    """
    Computes and optionally saves histograms, mean, and standard deviation
    of spatial periods across B-frames in a 3D nsOCT dataset.

    Args:
        nsoct (np.ndarray): 3D array of nsOCT spatial period values (shape: [b, y, x]).
        bins (int or array-like): Number of bins or bin edges for histogram plots.
        do_compute (bool): Whether to perform the analysis.
        do_save (bool): Whether to save figures and data to disk.

    Returns:
        dict: Dictionary containing per-frame means, stds, and overall statistics.
    """
    if not do_compute:
        return None

    import matplotlib
    matplotlib.rcParams.update({'font.size': 12})

    params = Params
    pixels_selected = params['pixels_selected']
    threshold_sp = params['threshold_spatial_period']
    b_frames_range = np.arange(params['b_frames_to_process'][0], params['b_frames_to_process'][1] + 1)
    hist_x_range = params['histograms_x_axis_range']
    hist_y_limit = params['histograms_y_axis_limit']
    save_dir = os.path.join(os.path.dirname(__file__), params['b_frame_histograms'])
    if do_save:
        os.makedirs(save_dir, exist_ok=True)

    # Apply threshold and crop to selected pixel range
    ns_oct_masked = np.where(nsoct <= threshold_sp, np.nan, nsoct)
    ns_oct_masked = ns_oct_masked[:, pixels_selected[0] - 1:pixels_selected[1], :]

    # Compute histogram for averaged B-frame (mean over B, then flatten)
    avg_b_frame = np.nanmean(ns_oct_masked, axis=0).flatten()
    mean_all = np.nanmean(avg_b_frame)
    std_all = np.nanstd(avg_b_frame)

    if do_save:
        plt.hist(avg_b_frame, bins=bins, range=hist_x_range, density=True, color='c', rwidth=0.8)
        plt.title("Mean of all B-frames")
        plt.xlabel("Spatial period (nm)")
        plt.ylabel("Amplitude (a.u.)")
        plt.xlim(hist_x_range)
        plt.ylim(0, hist_y_limit)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "averaged_b_frame.png"))
        plt.clf()

        np.savetxt(os.path.join(save_dir, "b_frame_single_data_mean_and_std.txt"),
                   [mean_all, std_all], fmt='%.4f', header="mean_all std_all")

    # Per-frame statistics
    frame_means = []
    frame_stds = []

    for idx, b_frame in zip(b_frames_range, ns_oct_masked):
        flat = b_frame.flatten()
        frame_means.append(np.nanmean(flat))
        frame_stds.append(np.nanstd(flat))

        if do_save:
            avg_along_a = np.nanmean(b_frame, axis=1)
            plt.hist(avg_along_a, bins=bins, range=hist_x_range, density=True,
                     color='g', rwidth=0.8)
            plt.title(f"Spatial period distribution for B-frame {idx}")
            plt.xlabel("Spatial period (nm)")
            plt.ylabel("Amplitude (a.u.)")
            plt.xlim(hist_x_range)
            plt.ylim(0, hist_y_limit)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"b_frame_{idx}.png"))
            plt.clf()

    frame_means = np.array(frame_means)
    frame_stds = np.array(frame_stds)
    mean_of_means = np.nanmean(frame_means)
    std_of_means = np.nanstd(frame_means)

    if do_save:
        # Save line plot of B-frame means
        plt.plot(b_frames_range, frame_means, '.', markersize=8)
        plt.title("B-frames vs Mean Spatial Period")
        plt.xlabel("B-frame number")
        plt.ylabel("Mean Spatial Period for B-frame (nm)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mean_vs_b_frames.png"))
        plt.clf()

        # Save all results
        np.save(os.path.join(save_dir, "b_frame_means.npy"), frame_means)
        np.savetxt(os.path.join(save_dir, "b_frame_means.txt"), frame_means, fmt='%.4f', header="b_frame_means")
        np.savetxt(os.path.join(save_dir, "b_frame_stds.txt"), frame_stds, fmt='%.4f', header="b_frame_stds")
        np.savetxt(os.path.join(save_dir, "mean_from_b_frame_means.txt"), [mean_of_means], fmt='%.4f', header="mean_of_means")
        np.savetxt(os.path.join(save_dir, "std_from_b_frame_means.txt"), [std_of_means], fmt='%.4f', header="std_of_means")

    return {
        "frame_means": frame_means,
        "frame_stds": frame_stds,
        "mean_of_means": mean_of_means,
        "std_of_means": std_of_means,
        "mean_all": mean_all,
        "std_all": std_all
    }