import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

from Params import *  # Import parameters (ensure this file exists)
from nsOCT_maximum_spatial_period import *  # Import additional utilities

# Use the Qt5 backend for matplotlib
matplotlib.use('Qt5Agg')

# Configure global font settings for plots
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12


def plot_B_frames(OCT, nsOCT, do_plot, do_save):
    """
    Function to plot B-frame images for OCT and nsOCT datasets.

    Args:
        OCT: 3D OCT data.
        nsOCT: 3D nsOCT data.
        do_plot (bool): Whether to display the plots.
        do_save (bool): Whether to save the plots to files.
    """
    is_exist = Params['is_exist']

    plt.rcParams.update({'font.size': 7})

    if do_plot:  # Proceed only if plotting is enabled
        # Retrieve parameters
        Number_of_pixels = Params['Number_of_pixels']
        pixels_selected = Params['pixels_selected']
        Number_of_A_lines = Params['Number_of_A_lines']
        Number_of_B_frames = Params['Number_of_B_frames']
        Lateral_dimension_X = Params['Lateral_dimension_X']
        Lateral_dimension_Y = Params['Lateral_dimension_Y']
        Depth_in_mm = Params['Depth_in_mm']
        B_frame_images = Params['B_frame_images']
        B_frames_to_process = Params["B_frames_to_process"]
        A_lines_to_process = Params["A_lines_to_process"]

        # Processing ranges for B-frames and A-lines
        processing_B_frames = np.arange(B_frames_to_process[0], B_frames_to_process[1] + 1)
        processing_A_lines = np.arange(A_lines_to_process[0], A_lines_to_process[1] + 1)

        # Calculate dimensions for display
        selected_depth_1 = (Depth_in_mm * 2 / Number_of_pixels) * pixels_selected[0]
        selected_depth_2 = (Depth_in_mm * 2 / Number_of_pixels) * pixels_selected[1]
        selected_X_dim_1 = (Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[0]
        selected_X_dim_2 = (Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[1]

        # Retrieve data
        three_d_OCT = OCT
        three_d_nsOCT = nsOCT
        print(f"The dimension of three_d_OCT is {np.shape(three_d_OCT)}")
        print(f"The dimension of three_d_nsOCT is {np.shape(three_d_nsOCT)}")

        # Iterate over each B-frame to process
        for OCT_image, nsOCT_image, figN in zip(three_d_OCT, three_d_nsOCT, processing_B_frames):
            # Create side-by-side subplots for OCT and nsOCT
            fig, (ax1, ax2) = plt.subplots(1, 2, num='nsOCT_Anand')

            # Plot OCT image
            OCT_img = ax1.imshow(
                OCT_image[pixels_selected[0]:pixels_selected[1], :],
                cmap='Greys_r',
                vmin=OCT_image_intensity_range[0],
                vmax=OCT_image_intensity_range[1],
                aspect=B_frame_aspect_ratio,
                interpolation='none',
                extent=[selected_X_dim_1, selected_X_dim_2, selected_depth_2, selected_depth_1]
            )
            plt.colorbar(mappable=OCT_img, shrink=0.3, ax=ax1, label='Intensity (a.u)')
            ax1.set_title('OCT image')
            ax1.set_xlabel("Lateral (mm)")
            ax1.set_ylabel("Depth (mm)")

            # Plot nsOCT image
            nsOCT_img = ax2.imshow(
                nsOCT_image[pixels_selected[0]:pixels_selected[1], :],
                cmap='jet',
                vmin=nsOCT_image_colorbar[0],
                vmax=nsOCT_image_colorbar[1],
                aspect=B_frame_aspect_ratio,
                interpolation='none',
                extent=[selected_X_dim_1, selected_X_dim_2, selected_depth_2, selected_depth_1]
            )
            plt.colorbar(mappable=nsOCT_img, shrink=0.3, ax=ax2, label='Spatial Periods (nm)')
            ax2.set_title('nsOCT image')
            ax2.set_xlabel("Lateral (mm)")
            ax2.set_ylabel("Depth (mm)")

            # Save or display the plot
            if do_save:
                plt.tight_layout()
                script_dir = os.path.dirname(__file__)
                b_frame_dir = os.path.join(script_dir, B_frame_images)
                plt.savefig(os.path.join(b_frame_dir, f"{figN}.jpg"), dpi = 300)
                plt.clf()
            else:
                plt.tight_layout()
                plt.show()

    # Handle case where existing images should be plotted
    if is_exist:
        print(f"The OCT saving location is {Params['oct_B_frames']}")

        for nsOCT_img, figN in zip(three_d_nsOCT, processing_B_frames):
            # Plot nsOCT image
            nsOCT_plot = plt.imshow(
                nsOCT_img[pixels_selected[0]:pixels_selected[1], :],
                cmap='jet',
                vmin=nsOCT_image_colorbar[0],
                vmax=nsOCT_image_colorbar[1],
                aspect=B_frame_aspect_ratio,
                interpolation='none',
                extent=[selected_X_dim_1, selected_X_dim_2, selected_depth_2, selected_depth_1]
            )
            plt.colorbar(mappable=nsOCT_plot, shrink=0.3, label='Spatial Periods (nm)')
            plt.title('nsOCT image')
            plt.xlabel("Lateral (mm)")
            plt.ylabel("Depth (mm)")
            if do_save:
                script_dir = os.path.dirname(__file__)
                b_frame_dir = os.path.join(script_dir, B_frame_images)
                plt.savefig(os.path.join(b_frame_dir, f"{figN}ns.jpg"), dpi = 300)
                plt.clf()
            else:
                plt.show()

        for OCT_img, figN in zip(three_d_OCT, processing_B_frames):
            # Plot OCT image
            OCT_plot = plt.imshow(
                OCT_img[pixels_selected[0]:pixels_selected[1], :],
                cmap='Greys_r',
                vmin=OCT_image_intensity_range[0],
                vmax=OCT_image_intensity_range[1],
                aspect=B_frame_aspect_ratio,
                interpolation='none',
                extent=[selected_X_dim_1, selected_X_dim_2, selected_depth_2, selected_depth_1]
            )
            plt.colorbar(mappable=OCT_plot, shrink=0.3, label='Intensity (a.u)')
            plt.title('OCT image')
            plt.xlabel("Lateral (mm)")
            plt.ylabel("Depth (mm)")
            if do_save:
                script_dir = os.path.dirname(__file__)
                b_frame_dir = os.path.join(script_dir, B_frame_images)
                plt.savefig(os.path.join(b_frame_dir, f"{figN}.jpg"), dpi = 300 )
                plt.clf()
            else:
                plt.show()
