import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib_scalebar.scalebar import ScaleBar
from Params import *  # Import global parameters from an external file
import os
from matplotlib import rcParams
from nsOCT_maximum_spatial_period import *  # Import specific functions or constants
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for rendering

# Configure global font settings for plots
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12

def enface(OCT, nsOCT, do_plot, do_save):
    """
    Function to process and visualize en-face images from OCT and nsOCT data.

    Parameters:
    OCT (array): 3D array containing OCT data.
    nsOCT (array): 3D array containing nsOCT data.
    do_plot (bool): Whether to display plots.
    do_save (bool): Whether to save the generated plots.
    """
    is_exist = Params['is_exist']  # Check if a specific condition exists (not used later)

    if do_plot:
        # Load parameters for image processing and plotting
        Number_of_pixels = Params['Number_of_pixels']
        Number_of_A_lines = Params['Number_of_A_lines']
        Number_of_B_frames = Params['Number_of_B_frames']
        Lateral_dimension_X = Params['Lateral_dimension_X']
        Lateral_dimension_Y = Params['Lateral_dimension_Y']
        Depth_in_mm = Params['Depth_in_mm']
        enface_images = Params['enface_images']
        oct_enface_images = Params['oct_enface_images']
        pixels_selected = Params['pixels_selected']

        # Define ranges for processing B-frames and A-lines
        processing_B_frames = np.arange(B_frames_to_process[0], B_frames_to_process[1] + 1)
        Num_B_frames = len(processing_B_frames)

        processing_A_lines = np.arange(A_lines_to_process[0], A_lines_to_process[1] + 1)
        num_A_lines_processed = len(processing_A_lines)

        # Input 3D data
        three_d_OCT = OCT
        three_d_nsOCT = nsOCT

        print("The dimension of three_d_OCT is", np.shape(three_d_OCT))
        print("The dimension of three_d_nsOCT is", np.shape(three_d_nsOCT))

        # Reorder dimensions for easier en-face visualization
        swap_array_nsOCT = np.swapaxes(three_d_nsOCT, 1, 0)
        enface_nsOCT = np.swapaxes(swap_array_nsOCT, 2, 1)

        swap_array_OCT = np.swapaxes(three_d_OCT, 1, 0)
        enface_OCT = np.swapaxes(swap_array_OCT, 2, 1)

        # Calculate lateral dimensions of selected ranges
        selected_Y_dimension_1 = float((Lateral_dimension_Y / Number_of_B_frames) * B_frames_to_process[0])
        selected_Y_dimension_2 = float((Lateral_dimension_Y / Number_of_B_frames) * B_frames_to_process[1])
        selected_X_dim_1 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[0])
        selected_X_dim_2 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[1])

        # Crop data to selected pixels, A-lines, and B-frames
        enface_OCT = enface_OCT[pixels_selected[0] - 1:pixels_selected[1],
                                A_lines_to_process[0]-1:A_lines_to_process[1],
                                B_frames_to_process[0]-1:B_frames_to_process[1]]

        enface_nsOCT = enface_nsOCT[pixels_selected[0] - 1:pixels_selected[1],
                                    A_lines_to_process[0]-1:A_lines_to_process[1],
                                    B_frames_to_process[0]-1:B_frames_to_process[1]]

        # Generate figure numbers for saving images
        figure_number = np.arange(pixels_selected[0], pixels_selected[1] + 1)

        # Plot nsOCT images
        for nsOCT, figN in zip(enface_nsOCT, figure_number):
            # Display nsOCT image with appropriate settings
            nsOCT_image = plt.imshow(nsOCT, cmap='jet', aspect=en_face_aspect_ratio, interpolation='none',
                                     vmin=nsOCT_image_colorbar[0], vmax=nsOCT_image_colorbar[1],
                                     extent=[selected_X_dim_1, selected_X_dim_2,
                                             selected_Y_dimension_2, selected_Y_dimension_1])
            nsOCT_image_array_save = np.asarray(nsOCT, dtype=np.float32)

            # Add colorbar and labels
            plt.colorbar(mappable=nsOCT_image, shrink=1, label='Spatial Periods (nm)', ticks=[590, 620, 650, 680])
            plt.axis('on')
            plt.title('nsOCT image')
            plt.xlabel("X axis (mm)")
            plt.ylabel("Y axis (mm)")

            # Save or display the plot
            if do_save:
                script_dir = os.path.dirname(__file__)
                enface_dir = os.path.join(script_dir, enface_images)
                plt.savefig(os.path.join(enface_dir, "{}ns.jpg".format(figN)), bbox_inches='tight')
                plt.clf()  # Clear the figure after saving
            else:
                plt.show()

        # Plot OCT images
        for OCT, figN in zip(enface_OCT, figure_number):
            plt.figure()

            # Rescale and normalize OCT data
            OCTimage_rescaled = OCT - np.min(OCT)
            OCTimage_rescaled_mean = np.mean(OCTimage_rescaled, axis=0)
            OCTimage_rescaled_mean_savgol = savgol_filter(OCTimage_rescaled_mean, window_length=49, polyorder=2)
            OCTimage_rescaled_background = OCTimage_rescaled_mean - OCTimage_rescaled_mean_savgol
            OCT = OCTimage_rescaled - OCTimage_rescaled_background

            print("The minimum and maximum intensity are", np.min(OCT), np.max(OCT))

            # Display OCT image
            OCT_image = plt.imshow(OCT, cmap='Greys_r', aspect=en_face_aspect_ratio, interpolation='none',
                                   vmin=OCT_image_intensity_range[0], vmax=OCT_image_intensity_range[1],
                                   extent=[selected_X_dim_1, selected_X_dim_2,
                                           selected_Y_dimension_2, selected_Y_dimension_1])

            # Save or display the plot
            if do_save:
                script_dir = os.path.dirname(__file__)
                enface_dir = os.path.join(script_dir, enface_images)
                plt.savefig(os.path.join(enface_dir, "{}.jpg".format(figN)), bbox_inches='tight')
                plt.clf()  # Clear the figure after saving
            else:
                plt.show()