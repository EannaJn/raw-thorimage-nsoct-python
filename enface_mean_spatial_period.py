from nsOCT_maximum_spatial_period import *  # Custom logic (if needed)


def mean_spatial_period_for_enface(nsoct, bins, do_compute=True, do_save=True):
    """
    Compute and optionally save the mean spatial period across en-face slices in a 3D nsOCT dataset.

    Args:
        nsOCT (ndarray): 3D nsOCT dataset [B, A, P].
        bins (int or array-like): Number of histogram bins or bin edges.
        do_compute (bool): If False, function returns immediately.
        do_save (bool): If True, saves plots and statistics to disk.

    Returns:
        dict: Dictionary containing per-enface means, stds, and overall statistics.
    """
    if not do_compute:
        return None

    # Load Params
    px_start, px_end = Params['pixels_selected']
    al_start, al_end = Params['a_lines_to_process']
    bf_start, bf_end = Params['b_frames_to_process']
    threshold = Params['threshold_spatial_period']
    x_range = Params['histograms_x_axis_range']
    y_limit = Params['histograms_y_axis_limit']
    save_dir = Params['enface_histograms']
    depth_mm = Params['depth_in_mm']
    total_px = Params['number_of_pixels']

    # Ensure save directory exists
    if do_save:
        os.makedirs(save_dir, exist_ok=True)

    # Threshold filtering: set values below or equal to threshold to NaN
    nsOCT_masked = np.where(nsoct > threshold, nsoct, np.nan)

    # Swap axes to get enface slices: [P, A, B]
    enface_nsOCT = np.swapaxes(np.swapaxes(nsOCT_masked, 1, 0), 2, 1)
    # Select the pixel, A-line, and B-frame ranges
    enface_nsOCT = enface_nsOCT[px_start - 1:px_end, al_start - 1:al_end, bf_start - 1:bf_end]

    # Compute stats over all selected enface slices (flattened)
    all_px_avg = np.nanmean(enface_nsOCT, axis=0).flatten()
    global_mean = np.nanmean(all_px_avg)
    global_std = np.nanstd(all_px_avg)

    # Plot and save histogram for all enface slices
    plt.hist(all_px_avg, bins=bins, range=x_range, density=True, color='b', rwidth=0.8)
    plt.title("Mean spatial period for all enface images")
    plt.xlabel("Spatial period (nm)")
    plt.ylabel("Amplitude (a.u.)")
    plt.xlim(x_range)
    plt.ylim(0, y_limit)
    plt.tight_layout()
    if do_save:
        plt.savefig(os.path.join(save_dir, "All_enface_histogram.png"))
    plt.clf()

    # Initialize result lists
    enface_means = []
    enface_stds = []
    enface_indices = range(px_start, px_end + 1)
    # Compute enface depths in mm (center of each pixel)
    enface_depths = np.linspace(
        (depth_mm * 2 / total_px) * px_start,
        (depth_mm * 2 / total_px) * px_end,
        px_end - px_start + 1
    )

    for i, (depth_slice, idx) in enumerate(zip(enface_nsOCT, enface_indices)):
        flat_slice = depth_slice.flatten()
        mean_val = np.nanmean(flat_slice)
        std_val = np.nanstd(flat_slice)
        enface_means.append(mean_val)
        enface_stds.append(std_val)

        if do_save:
            # Histogram of mean values along A-lines for this enface slice
            hist_vals = np.nanmean(depth_slice, axis=0)
            plt.hist(hist_vals, bins=bins, range=x_range, density=True, rwidth=0.8, color='y')
            plt.title(f"Spatial period distribution - Enface {idx}")
            plt.xlabel("Spatial period (nm)")
            plt.ylabel("Amplitude (a.u.)")
            plt.xlim(x_range)
            plt.ylim(0, y_limit)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"enface_{idx}.png"))
            plt.clf()

    enface_means = np.array(enface_means)
    enface_stds = np.array(enface_stds)
    mean_of_means = np.nanmean(enface_means)
    std_of_means = np.nanstd(enface_means)

    # Final plot: enface depth vs mean spatial period
    plt.plot(enface_depths, enface_means, '.-', color='navy')
    plt.title("Mean spatial period vs enface depths")
    plt.xlabel("Depth (mm)")
    plt.ylabel("Mean spatial period (nm)")
    plt.tight_layout()
    if do_save:
        plt.savefig(os.path.join(save_dir, "mean_vs_enface.png"))
    plt.clf()

    # Save numeric results
    if do_save:
        np.save(os.path.join(save_dir, "enface_means.npy"), enface_means)
        np.savetxt(os.path.join(save_dir, "enface_means.txt"), enface_means, fmt='%1.4f', header="enface_means")
        np.savetxt(os.path.join(save_dir, "enface_stds.txt"), enface_stds, fmt='%1.4f', header="enface_stds")
        np.savetxt(os.path.join(save_dir, "mean_from_enface_means.txt"), [mean_of_means], fmt='%1.4f', header="mean_of_means")
        np.savetxt(os.path.join(save_dir, "std_from_enface_means.txt"), [std_of_means], fmt='%1.4f', header="std_of_means")
        np.savetxt(os.path.join(save_dir, "global_enface_mean_std.txt"), [global_mean, global_std], fmt='%1.4f', header="global_mean global_std")

    return {
        "enface_means": enface_means,
        "enface_stds": enface_stds,
        "mean_of_means": mean_of_means,
        "std_of_means": std_of_means,
        "global_mean": global_mean,
        "global_std": global_std,
        "enface_depths": enface_depths
    }