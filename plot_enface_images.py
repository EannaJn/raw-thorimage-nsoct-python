import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib import rcParams
from pathlib import Path
from nsOCT_maximum_spatial_period import (
    Params, a_lines_to_process, b_frames_to_process,
    en_face_aspect_ratio, ns_oct_image_colorbar,
    oct_image_intensity_range
)
import matplotlib.font_manager as fm

matplotlib.use('Qt5Agg')
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12

def plot_enface_images(oct, nsoct, do_plot=True, do_save=False):
    """
    Process and visualize en-face OCT and nsOCT images in a single function, with X and Y scale bars.

    Parameters:
        oct (ndarray): 3D OCT volume (b_frames, a_lines, pixels).
        nsoct (ndarray): 3D nsOCT volume (b_frames, a_lines, pixels).
        do_plot (bool): Whether to display images.
        do_save (bool): Whether to save images to disk.
    """
    if not do_plot and not do_save:
        return

    # Load parameters
    px_start, px_end = Params['pixels_selected']
    n_alines = Params['number_of_a_lines']
    n_bframes = Params['number_of_b_frames']
    lat_x = Params['lateral_dimension_x']
    lat_y = Params['lateral_dimension_y']
    enface_dir = Path(__file__).resolve().parent / Params['enface_images']

    # Coordinate extents
    x1, x2 = (lat_x / n_alines) * np.array(a_lines_to_process)
    y1, y2 = (lat_y / n_bframes) * np.array(b_frames_to_process)

    # Calculate scale bar lengths (quarter of each dimension)
    x_extent = abs(x2 - x1)
    y_extent = abs(y2 - y1)
    x_scalebar_length = 0.25 * x_extent
    y_scalebar_length = 0.25 * y_extent

    img_shape = (a_lines_to_process[1] - a_lines_to_process[0], b_frames_to_process[1] - b_frames_to_process[0])
    x_pixel_size = x_extent / img_shape[0] if img_shape[0] > 0 else 1
    y_pixel_size = y_extent / img_shape[1] if img_shape[1] > 0 else 1

    # Reorient and crop both OCT and nsOCT
    def reorient_and_crop(data):
        data = np.swapaxes(data, 1, 0)
        data = np.swapaxes(data, 2, 1)
        return data[px_start - 1:px_end,
                    a_lines_to_process[0] - 1:a_lines_to_process[1],
                    b_frames_to_process[0] - 1:b_frames_to_process[1]]

    enface_nsoct = reorient_and_crop(nsoct)
    enface_oct = reorient_and_crop(oct)
    fig_numbers = np.arange(px_start, px_end + 1)

    if do_save:
        enface_dir.mkdir(parents=True, exist_ok=True)

    # Font properties for scale bar
    fontprops = fm.FontProperties(family='serif', size=10)

    for i, fign in enumerate(fig_numbers):
        # Plot nsOCT image
        fig, ax = plt.subplots()
        im_ns = ax.imshow(
            enface_nsoct[i], cmap='jet', aspect=en_face_aspect_ratio, interpolation='none',
            vmin=ns_oct_image_colorbar[0], vmax=ns_oct_image_colorbar[1],
            extent=[x1, x2, y2, y1]
        )
        plt.colorbar(im_ns, ax=ax, shrink=1, label='Spatial Periods (nm)', ticks=[600, 620, 650, 680])
        ax.set_title('nsOCT image')
        ax.set_xlabel("X axis (mm)")
        ax.set_ylabel("Y axis (mm)")

        if do_save:
            ns_save_path = enface_dir / f"{fign}ns.jpg"
            plt.savefig(ns_save_path, bbox_inches='tight', dpi=300)
            plt.clf()
        elif do_plot:
            plt.show()

        # Plot OCT image
        oct_img = enface_oct[i]
        oct_rescaled = oct_img - np.min(oct_img)
        oct_mean = np.mean(oct_rescaled, axis=0)
        oct_bg = oct_mean - savgol_filter(oct_mean, window_length=10, polyorder=2)
        oct_final = oct_rescaled - oct_bg

        fig, ax = plt.subplots()
        im_oct = ax.imshow(
            oct_final, cmap='Greys_r', aspect=en_face_aspect_ratio, interpolation='none',
            vmin=oct_image_intensity_range[0], vmax=oct_image_intensity_range[1],
            extent=[x1, x2, y2, y1]
        )
        plt.colorbar(im_oct, ax=ax, shrink=1, label="Intensity (a.u)")
        ax.set_title('OCT image')
        ax.set_xlabel("X axis (mm)")
        ax.set_ylabel("Y axis (mm)")

        if do_save:
            oct_save_path = enface_dir / f"{fign}.jpg"
            plt.savefig(oct_save_path, bbox_inches='tight', dpi=300)
            plt.clf()
        elif do_plot:
            plt.show()