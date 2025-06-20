import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

from Params import *  # Ensure this file defines all necessary parameters
from nsOCT_maximum_spatial_period import *

matplotlib.use('Qt5Agg')
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12

def plot_b_frames(
    oct,
    nsoct,
    do_plot=True,
    do_save=False,
    oct_cmap='Greys_r',
    ns_cmap='jet',
    figsize=(10, 5),
    dpi=300
):
    """
    Plot and/or save B-frame slices of OCT and nsOCT datasets.

    Args:
        oct (np.ndarray): 3D OCT volume (b, d, a).
        nsoct (np.ndarray): 3D nsOCT volume (b, d, a).
        do_plot (bool): Whether to display plots interactively.
        do_save (bool): Whether to save plots as image files.
        oct_cmap (str): Colormap for OCT images.
        ns_cmap (str): Colormap for nsOCT images.
        figsize (tuple): Figure size for plots.
        dpi (int): Dots per inch for saved figures.
    """
    params = Params
    is_exist = params.get('is_exist', False)

    # Load all needed parameters
    number_of_pixels = params['number_of_pixels']
    pixels_selected = params['pixels_selected']
    number_of_a_lines = params['number_of_a_lines']
    number_of_b_frames = params['number_of_b_frames']
    lateral_dimension_x = params['lateral_dimension_x']
    depth_in_mm = params['depth_in_mm']
    b_frame_images = params['b_frame_images']
    b_frames_to_process = np.arange(params["b_frames_to_process"][0], params["b_frames_to_process"][1] + 1)
    a_lines_to_process = params["a_lines_to_process"]
    selected_x_dim_1 = (lateral_dimension_x / number_of_a_lines) * a_lines_to_process[0]
    selected_x_dim_2 = (lateral_dimension_x / number_of_a_lines) * a_lines_to_process[1]
    selected_depth_1 = (depth_in_mm * 2 / number_of_pixels) * pixels_selected[0]
    selected_depth_2 = (depth_in_mm * 2 / number_of_pixels) * pixels_selected[1]

    def _plot_pair(oct_img, nsoct_img, fign, suffix=""):
        fig, (ax1, ax2) = plt.subplots(1, 2, num='nsOCT_B-frame', figsize=figsize)

        # OCT
        im1 = ax1.imshow(
            oct_img[pixels_selected[0]:pixels_selected[1], :],
            cmap=oct_cmap,
            vmin=oct_image_intensity_range[0],
            vmax=oct_image_intensity_range[1],
            aspect=b_frame_aspect_ratio,
            interpolation='none',
            extent=[selected_x_dim_1, selected_x_dim_2, selected_depth_2, selected_depth_1]
        )
        ax1.set_title("OCT image")
        ax1.set_xlabel("Lateral (mm)")
        ax1.set_ylabel("Depth (mm)")
        plt.colorbar(im1, ax=ax1, shrink=0.3, label="Intensity (a.u)")

        # nsOCT
        im2 = ax2.imshow(
            nsoct_img[pixels_selected[0]:pixels_selected[1], :],
            cmap=ns_cmap,
            vmin=ns_oct_image_colorbar[0],
            vmax=ns_oct_image_colorbar[1],
            aspect=b_frame_aspect_ratio,
            interpolation='none',
            extent=[selected_x_dim_1, selected_x_dim_2, selected_depth_2, selected_depth_1]
        )
        ax2.set_title("nsOCT image")
        ax2.set_xlabel("Lateral (mm)")
        ax2.set_ylabel("Depth (mm)")
        plt.colorbar(im2, ax=ax2, shrink=0.3, label="Spatial Periods (nm)")

        plt.tight_layout()

        if do_save:
            try:
                save_dir = os.path.join(os.path.dirname(__file__), b_frame_images)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"{fign}{suffix}.jpg"), dpi=dpi)
                plt.close(fig)
            except Exception as e:
                print(f"Error saving figure {fign}{suffix}: {e}")
        elif do_plot:
            plt.show()
            plt.close(fig)

    def _plot_single(img, fign, mode, cmap, vmin, vmax, aspect, extent, colorbar_label, do_save, suffix=""):
        fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
        im = ax.imshow(
            img[pixels_selected[0]:pixels_selected[1], :],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
            interpolation='none',
            extent=extent
        )
        ax.set_title(f"{mode} image")
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
        plt.colorbar(im, ax=ax, shrink=0.5, label=colorbar_label)
        plt.tight_layout()
        if do_save:
            try:
                save_dir = os.path.join(os.path.dirname(__file__), b_frame_images)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"{fign}{suffix}_{mode}.jpg"), dpi=dpi)
                plt.close(fig)
            except Exception as e:
                print(f"Error saving {mode} figure {fign}{suffix}: {e}")
        elif do_plot:
            plt.show()
            plt.close(fig)

    # Plot currently processed frames
    if do_plot:
        print(f"The dimension of OCT is {oct.shape}")
        print(f"The dimension of nsOCT is {nsoct.shape}")

    if not is_exist:
        for i, fign in enumerate(b_frames_to_process):
            _plot_pair(oct[i], nsoct[i], fign)

    # If previously saved data exists, plot or save again from that as separate images
    if is_exist:
        print(f"The OCT saving location is {params.get('oct_b_frames', 'N/A')}")
        for i, fign in enumerate(b_frames_to_process):
            # Plot OCT image only
            _plot_single(
                oct[i], fign, "OCT", oct_cmap,
                oct_image_intensity_range[0], oct_image_intensity_range[1],
                b_frame_aspect_ratio,
                [selected_x_dim_1, selected_x_dim_2, selected_depth_2, selected_depth_1],
                "Intensity (a.u)", do_save, suffix=""
            )
            # Plot nsOCT image only
            _plot_single(
                nsoct[i], fign, "nsOCT", ns_cmap,
                ns_oct_image_colorbar[0], ns_oct_image_colorbar[1],
                b_frame_aspect_ratio,
                [selected_x_dim_1, selected_x_dim_2, selected_depth_2, selected_depth_1],
                "Spatial Periods (nm)", do_save, suffix="ns"
            )