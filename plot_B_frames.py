import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from Params import*
matplotlib.use('Qt5Agg')
import os as os
from nsOCT_maximum_spatial_periodOCT import *

def plot_B_frames(OCT, nsOCT, do_plot,do_save):
    is_exist = Params['is_exist']

    if do_plot:

        if is_exist == False:

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

            processing_B_frames = np.arange(B_frames_to_process[0], B_frames_to_process[1] + 1)
            num_b_frames_processed = len(processing_B_frames)

            processing_A_lines = np.arange(A_lines_to_process[0], A_lines_to_process[1] + 1)
            num_A_lines_processed = len(processing_A_lines)

            selected_depth_1 = float((Depth_in_mm * 2 / Number_of_pixels) * pixels_selected[0])
            selected_depth_2 = float((Depth_in_mm * 2 / Number_of_pixels) * pixels_selected[1])

            selected_X_dim_1 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[0])
            selected_X_dim_2 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[1])

            three_d_OCT = OCT
            three_d_nsOCT = nsOCT


            print("The dimension of three_d_OCT is", np.shape(three_d_OCT))
            print("The dimension of three_d_nsOCT is", np.shape(three_d_nsOCT))


            for OCTimage, nsOCT, figN in zip(three_d_OCT,three_d_nsOCT, processing_B_frames):

                fig, (ax1,ax2) = plt.subplots(1,2,num='nsOCT_Anand')

                OCT_image = ax1.imshow(OCTimage[pixels_selected[0]:pixels_selected[1], :], cmap='Greys_r', vmin= OCT_image_intensity_range[0], vmax=OCT_image_intensity_range[1], aspect =B_frame_aspect_ratio,interpolation = 'none', extent=[selected_X_dim_1, selected_X_dim_2,selected_depth_2,selected_depth_1])

                plt.colorbar(mappable=OCT_image, shrink=0.5, ax=ax1, label='Intensity (a.u)')
                ax1.set_title('OCT image')
                ax1.set_xlabel("Lateral (mm)")
                ax1.set_ylabel("Depth (mm)")

                nsOCT_image = ax2.imshow(nsOCT[pixels_selected[0]:pixels_selected[1], :], cmap='jet', aspect =B_frame_aspect_ratio, vmin=nsOCT_image_colorbar[0], vmax=nsOCT_image_colorbar[1],interpolation = 'none', extent=[selected_X_dim_1, selected_X_dim_2,selected_depth_2,selected_depth_1 ])
                plt.colorbar(mappable=nsOCT_image,shrink = 0.5, ax=ax2, label='Spatial Periods (nm)')
                ax2.set_title('nsOCT image')
                ax2.set_xlabel("Lateral (mm)")
                ax2.set_ylabel("Depth (mm)")


                if do_save:
                    plt.tight_layout()
                    # plt.subplots_adjust(wspace= 0.6, right=0.62, hspace = 0.511)
                    script_dir = os.path.dirname(__file__)
                    b_frame_dir = os.path.join(script_dir, B_frame_images)
                    plt.savefig(os.path.join(b_frame_dir, "{}.jpg".format(figN)))
                    plt.clf()
                else:
                    plt.tight_layout()
                    plt.show()
                    # plt.waitforbuttonpress()
                    # plt.clf()

        if is_exist == True:

            three_d_nsOCT = nsOCT
            three_d_OCT = OCT

            Number_of_pixels = Params['Number_of_pixels']
            pixels_selected = Params['pixels_selected']
            Number_of_A_lines = Params['Number_of_A_lines']
            Number_of_B_frames = Params['Number_of_B_frames']
            Lateral_dimension_X = Params['Lateral_dimension_X']
            Lateral_dimension_Y = Params['Lateral_dimension_Y']
            Depth_in_mm = Params['Depth_in_mm']
            B_frame_images = Params['B_frame_images']
            oct_B_frames = Params['oct_B_frames']
            B_frames_to_process = Params["B_frames_to_process"]
            A_lines_to_process = Params["A_lines_to_process"]


            print("The oct saving location is", oct_B_frames)

            processing_B_frames = np.arange(B_frames_to_process[0], B_frames_to_process[1] + 1)
            num_b_frames_processed = len(processing_B_frames)

            processing_A_lines = np.arange(A_lines_to_process[0], A_lines_to_process[1] + 1)
            num_A_lines_processed = len(processing_A_lines)

            selected_depth_1 = float((Depth_in_mm * 2 / Number_of_pixels) * pixels_selected[0])
            selected_depth_2 = float((Depth_in_mm * 2 / Number_of_pixels) * pixels_selected[1])

            selected_X_dim_1 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[0])
            selected_X_dim_2 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[1])



            for nsOCT, figN in zip(three_d_nsOCT,processing_B_frames):
                nsOCT_image = plt.imshow(nsOCT[pixels_selected[0]:pixels_selected[1], :], cmap='jet', aspect =B_frame_aspect_ratio, vmin=nsOCT_image_colorbar[0], vmax=nsOCT_image_colorbar[1],interpolation = 'none', extent=[selected_X_dim_1, selected_X_dim_2,selected_depth_2,selected_depth_1 ])

                plt.colorbar(mappable=nsOCT_image, shrink=1, label='Spatial Periods (nm)')
                plt.title('nsOCT image')
                plt.xlabel("Lateral (mm)")
                plt.ylabel("Depth (mm)")

                if do_save:
                    # plt.tight_layout()
                    script_dir = os.path.dirname(__file__)
                    b_frame_dir = os.path.join(script_dir, B_frame_images)
                    plt.savefig(os.path.join(b_frame_dir, "{}ns.jpg".format(figN)))
                    plt.clf()
                else:
                    # plt.tight_layout()
                    plt.show()
                    # plt.waitforbuttonpress()
                    # plt.clf()

            for OCT, figN in zip(three_d_OCT, processing_B_frames):
                OCT_image = plt.imshow(OCT[pixels_selected[0]:pixels_selected[1], :], cmap='Greys_r', aspect=B_frame_aspect_ratio,
                                         vmin=OCT_image_intensity_range[0], vmax=OCT_image_intensity_range[1],
                                         interpolation='none',
                                         extent=[selected_X_dim_1, selected_X_dim_2, selected_depth_2,
                                                 selected_depth_1])

                plt.colorbar(mappable=OCT_image, shrink=1, label='Intensity (a.u)')
                plt.title('OCT image')
                plt.xlabel("Lateral (mm)")
                plt.ylabel("Depth (mm)")

                if do_save:
                    # plt.tight_layout()
                    pscript_dir = os.path.dirname(__file__)
                    b_frame_dir = os.path.join(script_dir, B_frame_images)
                    plt.savefig(os.path.join(b_frame_dir, "{}.jpg".format(figN)))
                    plt.clf()
                else:
                    # plt.tight_layout()
                    plt.show()
                    # plt.waitforbuttonpress()
                    # plt.clf()

