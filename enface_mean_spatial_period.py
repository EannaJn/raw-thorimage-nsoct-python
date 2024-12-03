import numpy as np
import matplotlib.pyplot as plt
from Params import*
import os
from nsOCT_maximum_spatial_periodOCT import *


def Mean_spatial_period_for_enface(nsOCT, bins,do_compute, do_save):

    if do_compute:

        Number_of_pixels = Params['Number_of_pixels']
        Number_of_A_lines = Params['Number_of_A_lines']
        Number_of_B_frames = Params['Number_of_B_frames']
        Lateral_dimension_X = Params['Lateral_dimension_X']
        Lateral_dimension_Y = Params['Lateral_dimension_Y']
        Depth_in_mm = Params['Depth_in_mm']
        # Num_B_frames = Params['Number_of_B_frames_to_process']
        enface_histograms = Params['enface_histograms']
        threshold_spatial_period = Params['threshold_spatial_period']
        enface_histograms = Params['enface_histograms']
        B_frames_to_process = Params["B_frames_to_process"]

        selected_pixels = (pixels_selected[1] - pixels_selected[0])+1

        three_d_nsOCT = nsOCT
        NaN_limit = threshold_spatial_period
        three_d_nsOCT[three_d_nsOCT <= NaN_limit] = np.NaN

        swap_array_nsOCT = np.swapaxes(three_d_nsOCT, 1, 0)
        enface_nsOCT = np.swapaxes(swap_array_nsOCT, 2, 1)

        enface_nsOCT = enface_nsOCT[pixels_selected[0] - 1:pixels_selected[1],
                         A_lines_to_process[0]-1:A_lines_to_process[1], B_frames_to_process[0]-1:B_frames_to_process[1]]

        enface_averaged_along_pixels = np.nanmean(three_d_nsOCT, axis=1)
        print("The dimension of averaged_along_pixels is", np.shape(enface_averaged_along_pixels))

        flat_enface = enface_averaged_along_pixels.flatten()
        enface_single_data_mean = np.nanmean(flat_enface)
        enface_std = np.nanstd(flat_enface)

        plt.hist(flat_enface, bins=bins, range=(histograms_x_axis_range[0], histograms_x_axis_range[1]), histtype='bar', density=True,color='b',rwidth=0.8)
        # sns.distplot(flat_enface)
        plt.title("Mean spatial period for all enface images")
        plt.xlabel("Spatial period (nm)")
        plt.ylabel("Amplitude (a.u)")
        plt.ylim(0, histograms_y_axis_limit)
        plt.tight_layout()
        # plt.savefig(enface_histograms + r"\averaged_enface_slice.png")
        plt.clf()

        Single_mean_and_std = np.array([enface_single_data_mean, enface_std])

        # np.savetxt(enface_histograms + r"\enface_single_data_mean_and_std.txt", Single_mean_and_std, fmt='%1.2f')

        """Average of a single enface slice"""

        enface_numbers = np.arange(pixels_selected[0], pixels_selected[1]+1)
        enface_depths = np.linspace((Depth_in_mm*2/Number_of_pixels)*pixels_selected[0], (Depth_in_mm*2/Number_of_pixels)*pixels_selected[1], selected_pixels)
        enface_means = []
        enface_stds = []

        for enface,histo_name in zip(enface_nsOCT,enface_numbers):
            flattened_enface = enface.flatten()
            enface_averaged_along_A_lines = np.nanmean(enface,axis = 0)
            enface_mean = np.nanmean(flattened_enface)
            print("The mean spatial period for the enface image  is",enface_mean )
            enface_means.append(enface_mean)
            enface_std = np.nanstd(flattened_enface)
            enface_stds.append(enface_std)

            if do_save:
                enface_histo = plt.hist(enface_averaged_along_A_lines, density=True,bins=bins, rwidth = 0.8, range=(histograms_x_axis_range[0], histograms_x_axis_range[1]), histtype='bar', color='y')
                plt.title('Spatial period distribution for each enface')
                plt.xlabel("Spatial period (nm)")
                plt.ylabel("Amplitude")
                plt.ylim(0, histograms_y_axis_limit)
                plt.tight_layout()
                script_dir = os.path.dirname(__file__)
                enface_hist_dir = os.path.join(script_dir, enface_histograms)
                plt.savefig(os.path.join(enface_hist_dir, "{}.png".format(histo_name)))
                plt.clf()

        enface_means = np.asarray(enface_means)
        enface_stds = np.asarray(enface_stds)
        mean_from_enface_means = np.mean(enface_means)
        mean_from_enface_means = np.asarray([mean_from_enface_means])
        std_from_enface_means = np.std(enface_means)
        std_from_enface_means = np.asarray([std_from_enface_means])
        mean_vs_enface = plt.plot(enface_depths,enface_means, '.')
        plt.title("Mean spatial period vs enface depths")
        plt.xlabel("enface depths")
        plt.ylabel("Mean spatial period (nm)")
        plt.tight_layout()
        script_dir = os.path.dirname(__file__)
        enface_hist_dir = os.path.join(script_dir, enface_histograms)
        plt.savefig(os.path.join(enface_hist_dir, "mean_vs_enface.png"))
        plt.clf()
        np.save(enface_histograms + r"\en_face_means.npy", enface_means)
        np.savetxt(enface_histograms + r"\mean_from_enface_means.txt", mean_from_enface_means, fmt='%1.2f')
        np.savetxt(enface_histograms + r"\std_from_enface_means.txt", std_from_enface_means, fmt='%1.2f')
        np.savetxt(enface_histograms + r"\enface_stds.txt" , enface_stds, fmt='%1.2f')
        np.savetxt(enface_histograms + r"\enface_means.txt", enface_means, fmt='%1.1f',newline=',')

