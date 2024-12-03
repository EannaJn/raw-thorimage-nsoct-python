import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib_scalebar.scalebar import ScaleBar
from Params import*
import os
from matplotlib import rcParams
from nsOCT_maximum_spatial_periodOCT import *
matplotlib.use('Qt5Agg')

rcParams['font.serif'] = 'Times New Roman'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12

def enface(OCT, nsOCT, do_plot, do_save):

    is_exist = Params['is_exist']

    if do_plot:
        if is_exist == False:
            Number_of_pixels = Params['Number_of_pixels']
            Number_of_A_lines = Params['Number_of_A_lines']
            Number_of_B_frames = Params['Number_of_B_frames']
            Lateral_dimension_X = Params['Lateral_dimension_X']
            Lateral_dimension_Y = Params['Lateral_dimension_Y']
            Depth_in_mm = Params['Depth_in_mm']
            enface_images = Params['enface_images']
            pixels_selected = Params['pixels_selected']

            processing_B_frames = np.arange(B_frames_to_process[0], B_frames_to_process[1] + 1)
            Num_B_frames = len(processing_B_frames)

            processing_A_lines = np.arange(A_lines_to_process[0], A_lines_to_process[1] + 1)
            num_A_lines_processed = len(processing_A_lines)



            three_d_OCT = OCT
            three_d_nsOCT = nsOCT

            print("The dimension of three_d_OCT is", np.shape(three_d_OCT))
            print("The dimension of three_d_nsOCT is", np.shape(three_d_nsOCT))
            swap_array_OCT = np.swapaxes(three_d_OCT, 1, 0)
            enface_OCT = np.swapaxes(swap_array_OCT, 2, 1)
            swap_array_nsOCT = np.swapaxes(three_d_nsOCT, 1, 0)
            enface_nsOCT = np.swapaxes(swap_array_nsOCT, 2, 1)


            selected_pixels = pixels_selected[1] - pixels_selected[0]

            selected_Y_dimension_1 = float((Lateral_dimension_Y / Number_of_B_frames) * B_frames_to_process[0])
            selected_Y_dimension_2 = float((Lateral_dimension_Y/ Number_of_B_frames) * B_frames_to_process[1])

            selected_X_dim_1 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[0])
            selected_X_dim_2 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[1])


            enface_OCT = enface_OCT[pixels_selected[0] - 1:pixels_selected[1],
                         A_lines_to_process[0]-1:A_lines_to_process[1], B_frames_to_process[0]-1:B_frames_to_process[1]]
            enface_nsOCT = enface_nsOCT[pixels_selected[0] - 1:pixels_selected[1],
                         A_lines_to_process[0]-1:A_lines_to_process[1], B_frames_to_process[0]-1:B_frames_to_process[1]]
            print("The dimensions of eface_OCT and enface_nsOCT are", np.shape(enface_OCT), np.shape(enface_nsOCT))


            figure_number = np.arange(pixels_selected[0],pixels_selected[1]+1)

            for OCTimage, nsOCT, figN in zip(enface_OCT,enface_nsOCT, figure_number):

                fig, (ax1,ax2) = plt.subplots(1,2,num='nsOCT_Anand')

                # OCTimage_rescaled = OCTimage[0]-np.min(OCTimage)
                # OCTimage_rescaled_mean = np.mean(OCTimage_rescaled, axis = 0)
                # OCTimage_rescaled_mean_savgol = savgol_filter(OCTimage_rescaled_mean, window_length= 101, polyorder= 2)
                # OCTimage_rescaled_background = OCTimage_rescaled_mean - OCTimage_rescaled_mean_savgol
                # OCTimage = OCTimage_rescaled - OCTimage_rescaled_background*1.0

                OCT_image = ax1.imshow(OCTimage, cmap='Greys_r',aspect = en_face_aspect_ratio, vmin= OCT_image_intensity_range[0],interpolation = 'none', vmax= OCT_image_intensity_range[1], extent=[selected_X_dim_1,selected_X_dim_2,selected_Y_dimension_2,selected_Y_dimension_1])
                plt.colorbar(mappable=OCT_image, shrink=0.40, ax=ax1, label='Intensity (a.u)')
                ax1.set_title('OCT image')
                ax1.set_xlabel("X-axis (mm)")
                ax1.set_ylabel("Y-axis (mm)")

                nsOCT_image = ax2.imshow(nsOCT, cmap='jet',aspect = en_face_aspect_ratio,interpolation = 'none',vmin = nsOCT_image_colorbar[0], vmax = nsOCT_image_colorbar[1] ,extent=[selected_X_dim_1,selected_X_dim_2,selected_Y_dimension_2,selected_Y_dimension_1])
                plt.colorbar(mappable=nsOCT_image,shrink = 0.40,ax = ax2, label='Spatial Periods (nm)')
                ax2.set_title('nsOCT image')
                ax2.set_xlabel("X-axis (mm)")
                ax2.set_ylabel("Y-axis (mm)")

                if do_save:
                    # plt.tight_layout()
                    plt.subplots_adjust(wspace=0.6)
                    script_dir = os.path.dirname(__file__)
                    enface_dir = os.path.join(script_dir, enface_images)
                    plt.savefig(os.path.join(enface_dir, "{}.jpg".format(figN)))
                    plt.clf()
                else:
                    # plt.tight_layout()
                    plt.waitforbuttonpress()
                    plt.clf()


        elif is_exist == True:

            Number_of_pixels = Params['Number_of_pixels']
            Number_of_A_lines = Params['Number_of_A_lines']
            Number_of_B_frames = Params['Number_of_B_frames']
            Lateral_dimension_X = Params['Lateral_dimension_X']
            Lateral_dimension_Y = Params['Lateral_dimension_Y']
            Depth_in_mm = Params['Depth_in_mm']
            enface_images = Params['enface_images']
            oct_enface_images = Params['oct_enface_images']
            pixels_selected = Params['pixels_selected']

            processing_B_frames = np.arange(B_frames_to_process[0], B_frames_to_process[1] + 1)
            Num_B_frames = len(processing_B_frames)

            processing_A_lines = np.arange(A_lines_to_process[0], A_lines_to_process[1] + 1)
            num_A_lines_processed = len(processing_A_lines)

            three_d_OCT = OCT
            three_d_nsOCT = nsOCT

            print("The dimension of three_d_OCT is", np.shape(three_d_OCT))
            print("The dimension of three_d_nsOCT is", np.shape(three_d_nsOCT))

            swap_array_nsOCT = np.swapaxes(three_d_nsOCT, 1, 0)
            enface_nsOCT = np.swapaxes(swap_array_nsOCT, 2, 1)

            selected_pixels = pixels_selected[1] - pixels_selected[0]

            swap_array_OCT = np.swapaxes(three_d_OCT, 1, 0)
            enface_OCT = np.swapaxes(swap_array_OCT, 2, 1)
            # enface_OCT = enface_OCT[pixels_selected[0] - 1:pixels_selected[1], :, :]



            selected_Y_dimension_1 = float((Lateral_dimension_Y / Number_of_B_frames) * B_frames_to_process[0])
            selected_Y_dimension_2 = float((Lateral_dimension_Y / Number_of_B_frames) * B_frames_to_process[1])

            selected_X_dim_1 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[0])
            selected_X_dim_2 = float((Lateral_dimension_X / Number_of_A_lines) * A_lines_to_process[1])

            enface_OCT = enface_OCT[pixels_selected[0] - 1:pixels_selected[1],
                         A_lines_to_process[0]-1:A_lines_to_process[1], B_frames_to_process[0]-1:B_frames_to_process[1]]
            enface_nsOCT = enface_nsOCT[pixels_selected[0] - 1:pixels_selected[1],
                         A_lines_to_process[0]-1:A_lines_to_process[1], B_frames_to_process[0]-1:B_frames_to_process[1]]

            figure_number = np.arange(pixels_selected[0], pixels_selected[1] + 1)

            for nsOCT, figN in zip(enface_nsOCT, figure_number):


                nsOCT_image = plt.imshow(nsOCT, cmap='jet',aspect = en_face_aspect_ratio,interpolation = 'none',vmin = nsOCT_image_colorbar[0],
                                         vmax = nsOCT_image_colorbar[1] ,extent=[selected_X_dim_1,selected_X_dim_2,
                                                                                 selected_Y_dimension_2,selected_Y_dimension_1])
                nsOCT_image_array_save = np.asarray((nsOCT), dtype = np.float32)
                #np.save(enface_images + r"\nsOCT_image_matrix{}.npy".format(figN), nsOCT_image_array_save)

                plt.colorbar(mappable=nsOCT_image, shrink=1, label='Spatial Periods (nm)', ticks=[590,620,650,680])
                plt.axis('on')
                plt.title('nsOCT image')
                plt.xlabel("X axis (mm)")
                plt.ylabel("Y axis (mm)")

                if do_save:
                    # plt.tight_layout()
                    script_dir = os.path.dirname(__file__)
                    enface_dir = os.path.join(script_dir, enface_images)
                    plt.savefig(os.path.join(enface_dir, "{}ns.jpg".format(figN)), bbox_inches = 'tight')
                    plt.clf()
                else:
                    # plt.tight_layout()
                    plt.show()
                    # plt.waitforbuttonpress()
                    # plt.clf()

            for OCT, figN in zip(enface_OCT, figure_number):

                plt.figure()
                OCTimage_rescaled = OCT - np.min(OCT)
                OCTimage_rescaled_mean = np.mean(OCTimage_rescaled, axis=0)
                OCTimage_rescaled_mean_savgol = savgol_filter(OCTimage_rescaled_mean, window_length=49, polyorder=2)
                OCTimage_rescaled_background = OCTimage_rescaled_mean - OCTimage_rescaled_mean_savgol
                OCT = OCTimage_rescaled - OCTimage_rescaled_background * 1.0
                print("the minimum and maximum intensity are", np.min(OCT), np.max(OCT))


                OCT_image = plt.imshow(OCT, cmap='Greys_r',aspect = en_face_aspect_ratio,interpolation = 'none',vmin = OCT_image_intensity_range[0],
                                         vmax = OCT_image_intensity_range[1] ,extent=[selected_X_dim_1,selected_X_dim_2,
                                                                                 selected_Y_dimension_2,selected_Y_dimension_1])

                OCT_image_array_save = np.asarray((enface_OCT), dtype=np.float32)
                #np.save(enface_images + r"\OCT_image_matrix{}.npy".format(figN), OCT_image_array_save)

                # scalebar = ScaleBar(dx= 0.001,label = None, fixed_value = 0.1,fixed_units= 'mm',color = 'tab:green',scale_formatter =None,frameon=False, location = 'lower right')  # 1 pixel = 0.1 mm
                # plt.gca().add_artist(scalebar)


                # x_line = [0.39, 0.49]
                # y_line = [0.48, 0.48]
                # plt.plot(x_line, y_line, color='white', linewidth=6.5, markeredgecolor = 'green')

                plt.axis('on')
                plt.title('OCT image')
                plt.xlabel("X axis (mm)")
                plt.ylabel("Y axis (mm)")

                if do_save:
                    # plt.tight_layout()
                    script_dir = os.path.dirname(__file__)
                    enface_dir = os.path.join(script_dir, enface_images)
                    plt.savefig(os.path.join(enface_dir, "{}.jpg".format(figN)), bbox_inches='tight')
                    plt.clf()
                else:
                    # plt.tight_layout()

                    plt.show()
                    # plt.waitforbuttonpress()
                    # plt.clf()