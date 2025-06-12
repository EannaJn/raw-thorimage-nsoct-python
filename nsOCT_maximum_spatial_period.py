import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import scipy.signal as sg
from scipy import interpolate as ip
from scipy.fftpack import fft, fftshift, ifft,ifftshift
import os as os
import os.path
import Params
import zipfile
import xmltodict
from Params import*
from scipy import ndimage as ndimage
from window_function import select_window


def OCT_nsOCT_processing(f_name):
    """ The path to the .OCT file """

    # Get the root directory and file name without extension
    root_direc = os.path.split(f_name)[0]
    file_base_name = os.path.splitext(os.path.split(f_name)[1])[0]

    # Create unique directories for each file based on the file name
    output_base_dir = os.path.join(root_direc, file_base_name)

    # Create unique subdirectories for the current file's outputs
    processed_data_dir = os.path.join(output_base_dir, "Processed_from_original_data")
    re_processed_data_dir = os.path.join(output_base_dir, "Re-processed_from_original_data")

    unzipped_folder_dir = os.path.join(output_base_dir, "unzipped_folder")
    b_frames_OCT_dir = os.path.join(output_base_dir, "b_frames_OCT")
    b_frames_nsOCT_dir = os.path.join(output_base_dir, "b_frames_nsOCT")

    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(unzipped_folder_dir, exist_ok=True)
    os.makedirs(b_frames_OCT_dir, exist_ok=True)
    os.makedirs(b_frames_nsOCT_dir, exist_ok=True)

    search_path = os.path.join(b_frames_nsOCT_dir, 'nsOCT_1.npy')
    is_exist = os.path.exists(search_path)

    if not is_exist:
        # Continue with file processing for this unique file
        output_subfolders = ['B_frame_images', 'enface_images', 'B_frame_histograms', 'enface_histograms',
                             'Full_data_flattened']
        for osf in output_subfolders:
            os.makedirs(os.path.join(processed_data_dir, osf), exist_ok=True)

        # Paths to output subdirectories for this file
        B_frame_images = os.path.join(processed_data_dir, output_subfolders[0])
        enface_images = os.path.join(processed_data_dir, output_subfolders[1])
        B_frame_histograms = os.path.join(processed_data_dir, output_subfolders[2])
        enface_histograms = os.path.join(processed_data_dir, output_subfolders[3])
        Full_data_flattened = os.path.join(processed_data_dir, output_subfolders[4])

        with zipfile.ZipFile(file=f_name, mode='r') as file:
            file.extractall(path=unzipped_folder_dir)

        Header = os.path.join(unzipped_folder_dir,"Header.xml")
        Chirp__Data = os.path.join(unzipped_folder_dir,"data\Chirp.data")
        Apodization__array = os.path.join(unzipped_folder_dir,"data\Spectral0.data")

        with open(Header, 'r') as telestoheader:
            xmlstring = telestoheader.read(-1)

        xmldict = xmltodict.parse(xmlstring)

        Number_of_pixels = 2048
        Number_of_A_lines = int(xmldict['Ocity']['DataFiles']['DataFile'][3]['@SizeX'])
        Number_of_B_frames = int(xmldict['Ocity']['Image']['SizePixel']['SizeY'])
        Lateral_dimension_X = float(xmldict['Ocity']['Image']['SizeReal']['SizeX'])
        Lateral_dimension_Y = float(xmldict['Ocity']['Image']['SizeReal']['SizeY'])
        Depth_in_mm = float(xmldict['Ocity']['Image']['SizeReal']['SizeZ'])

        Apodization_array = np.fromfile(Apodization__array, dtype='u2', count=-1)
        Apodization_A_lines = Apodization_array.reshape(Number_of_A_lines, Number_of_pixels)

        Chirp_Data = np.fromfile(Chirp__Data, dtype='float32', count=-1)

        print("Number of pixels is", Number_of_pixels,
              "\nNumber of A-lines is", Number_of_A_lines,
              "\nNumber of B-frames is", Number_of_B_frames,
              "\nLateral dimension X is", Lateral_dimension_X,
              "\nLateral dimension Y is", Lateral_dimension_Y,
              "\nThe depth in mm is", Depth_in_mm)

        B_frames_to_process = Params["B_frames_to_process"]
        A_lines_to_process = Params["A_lines_to_process"]
        processing_B_frames = np.arange(B_frames_to_process[0], B_frames_to_process[1] + 1)

        for b_frame_number in processing_B_frames:
            print('Processing B-frame {}...'.format(b_frame_number))

            b__frame = os.path.join(unzipped_folder_dir,"data\Spectral{}.data".format(b_frame_number))

            b_frame = np.fromfile(b__frame, dtype='u2', count=-1)
            b_frame = b_frame.reshape(Number_of_A_lines, Number_of_pixels)

            Mean_Apo_spectrum = np.array(np.mean(Apodization_A_lines, axis = 0))

            DC_Removed = b_frame - Mean_Apo_spectrum

            Chirp_Data_len = Chirp_Data.shape[0]
            x = np.linspace(0,int(Chirp_Data_len-1),Chirp_Data_len)
            interpolation_function = ip.interp1d(Chirp_Data, DC_Removed, kind = 'linear', axis = 1)
            Linear_Data = interpolation_function(x)

            # Apodization window
            depth_index = np.arange(Number_of_pixels) - Number_of_pixels // 2
            window_band = 0.3 * Number_of_pixels
            Apodization_window = select_window("Cosine",Number_of_pixels,Number_of_A_lines,depth_index,window_band)
            Apodized_data = np.multiply(Linear_Data, Apodization_window.T)

            data = Apodized_data.T
            #data = Linear_Data.T

            data_after_ifft = 20 * np.log10(abs(spfft.ifft(Apodized_data)))
            Prep_Normalization = data_after_ifft + abs(np.min(data_after_ifft))
            Normalized_data = ((Prep_Normalization / (np.amax(Prep_Normalization))) * 255).T

            Lower_wavelength = Params['Lower_wavelength']
            Upper_wavelength = Params['Upper_wavelength']
            Wavelength = np.linspace(Lower_wavelength,Upper_wavelength,Number_of_pixels)

            Threshold = Params['Threshold']
            Refractive_index = Params["Refractive_index"]
            Hz = Wavelength/(2*Refractive_index)*1000 ##unit: nm
            num_window= Params['Number_of_windows']
            Pixels_per_window = int(Number_of_pixels / num_window)

            index_max_N_win_fft_arr = []

            for Aline in data.T:
                N_window_fft = [] # list to use append
                tukey_window_short = sg.windows.tukey(Pixels_per_window, alpha=0.75) # create short tukey
                tukey_window_full = np.pad(tukey_window_short, pad_width=(0,Number_of_pixels - Pixels_per_window), mode = 'constant') # pad to full length
                for _ in range(0,num_window*oversampling_degree):
                    N_window_fft.append(np.log10(abs((fft(Aline * tukey_window_full)))))
                    tukey_window_full = np.roll(tukey_window_full, int(abs(Pixels_per_window/oversampling_degree)))
                N_window_fft_wo_med_filter=np.array(N_window_fft)
                N_window_fft = ndimage.median_filter(input=N_window_fft_wo_med_filter, size=(4, 4))
                #N_window_fft = np.array(N_window_fft)
                N_window_fft[N_window_fft < Threshold] = 0
                index_of_max_in_N_window_fft = np.argmax(N_window_fft,axis=0)
                index_max_N_win_fft_arr.append(index_of_max_in_N_window_fft)

            Win4 = np.array(index_max_N_win_fft_arr).T
            Win5a = []
            for w in Win4.T:
                #Win5a.append(Hz[np.round(w * int(abs(Pixels_per_window / oversampling_degree)) + int(
                    #abs(Pixels_per_window / (2 * oversampling_degree))))])
                Win5a.append(Hz[np.round(w * int(abs(Pixels_per_window)))])

            Win5 = np.array(Win5a).T

            b_frame_OCT = np.asarray((Normalized_data), dtype=np.float32)
            b_frame_nsOCT = np.asarray((Win5), dtype=np.float32)

            script_dir = os.path.dirname(__file__)
            b_frames_OCT_dir = os.path.join(script_dir, b_frames_OCT_dir)
            b_frames_nsOCT_dir = os.path.join(script_dir, b_frames_nsOCT_dir)

            np.save(os.path.join(b_frames_OCT_dir, r'OCT_{}.npy').format(b_frame_number), b_frame_OCT)
            np.save(os.path.join(b_frames_nsOCT_dir ,r'nsOCT_{}.npy').format(b_frame_number), b_frame_nsOCT)


        three_d_OCT_list = []
        three_d_nsOCT_list = []

        for i in processing_B_frames:
            OCT_bf = np.load(b_frames_OCT_dir + r'/OCT_{}.npy'.format(b_frame_number))
            three_d_OCT_list.append(OCT_bf)

        for i in processing_B_frames:
            nsOCT_bf = np.load(b_frames_nsOCT_dir + r'/nsOCT_{}.npy'.format(b_frame_number))
            three_d_nsOCT_list.append(nsOCT_bf)

        three_d_OCT = np. asarray((three_d_OCT_list), dtype = np.float32)
        three_d_nsOCT = np.asarray((three_d_nsOCT_list), dtype = np.float32)

        three_d_OCT = three_d_OCT[:, :, A_lines_to_process[0]-1:A_lines_to_process[1]]
        three_d_nsOCT = three_d_nsOCT[:, :, A_lines_to_process[0] - 1:A_lines_to_process[1]]

        #compute_bins_prep = np.unique(three_d_nsOCT)
        compute_bins_prep = np.linspace(np.min(three_d_nsOCT),np.max(three_d_nsOCT),101)
        compute_bins = np.sort(compute_bins_prep)[1:]

        threshold_spatial_period = np.min(three_d_nsOCT)


        print("The dimension of three_d_OCT is", np.shape(three_d_OCT))
        print("The dimension of three_d_nsOCT is", np.shape(three_d_nsOCT))
        print("The minimum spatial period is", np.min(three_d_nsOCT))
        print("The maximum spatial period is", np.max(three_d_nsOCT))
        print("the number of elements in the bins sequence", len(compute_bins))

    if is_exist == True:
        print("Data already processed")
        # Continue with file processing for this unique file
        output_subfolders = ['B_frame_images', 'enface_images', 'B_frame_histograms', 'enface_histograms',
                             'Full_data_flattened', 'OCT_B_frames', 'OCT_enface_images']
        for osf in output_subfolders:
            os.makedirs(os.path.join(re_processed_data_dir, osf), exist_ok=True)

        # Paths to output subdirectories for this file
        B_frame_images = os.path.join(re_processed_data_dir, output_subfolders[0])
        enface_images = os.path.join(re_processed_data_dir, output_subfolders[1])
        B_frame_histograms = os.path.join(re_processed_data_dir, output_subfolders[2])
        enface_histograms = os.path.join(re_processed_data_dir, output_subfolders[3])
        Full_data_flattened = os.path.join(re_processed_data_dir, output_subfolders[4])
        oct_B_frames = os.path.join(re_processed_data_dir, output_subfolders[5])
        oct_enface_images = os.path.join(re_processed_data_dir, output_subfolders[6])

        with zipfile.ZipFile(file=f_name, mode='r') as file:
            file.extractall(path=unzipped_folder_dir)

        Header = os.path.join(unzipped_folder_dir, "Header.xml")

        with open(Header, 'r') as telestoheader:
            xmlstring = telestoheader.read(-1)

        xmldict = xmltodict.parse(xmlstring)

        Number_of_pixels = 2048
        Number_of_A_lines = int(xmldict['Ocity']['DataFiles']['DataFile'][3]['@SizeX'])
        Number_of_B_frames = int(xmldict['Ocity']['Image']['SizePixel']['SizeY'])
        Lateral_dimension_X = float(xmldict['Ocity']['Image']['SizeReal']['SizeX'])
        Lateral_dimension_Y = float(xmldict['Ocity']['Image']['SizeReal']['SizeY'])
        Depth_in_mm = float(xmldict['Ocity']['Image']['SizeReal']['SizeZ'])

        for osf in output_subfolders:
            os.makedirs(os.path.join(re_processed_data_dir, osf), exist_ok=True)

        Params.update({"oct_B_frames": oct_B_frames,
                       "oct_enface_images": oct_enface_images})

        print("Number of pixels is", Number_of_pixels,
              "\nNumber of A-lines is", Number_of_A_lines,
              "\nNumber of B-frames is", Number_of_B_frames,
              "\nLateral dimension X is", Lateral_dimension_X,
              "\nLateral dimension Y is", Lateral_dimension_Y,
              "\nThe depth in mm is", Depth_in_mm)

        B_frames_to_process = Params["B_frames_to_process"]
        A_lines_to_process = Params["A_lines_to_process"]

        # three_d_OCT = np.load(file = path_to_OCT_file)
        # three_d_nsOCT = np.load(file = f_name)

        three_d_OCT_list = []
        three_d_nsOCT_list = []

        processing_B_frames = np.arange(B_frames_to_process[0], B_frames_to_process[1] + 1)
        b_frames_OCT = os.path.join(file_base_name, "b_frames_OCT")
        b_frames_nsOCT = os.path.join(file_base_name, "b_frames_nsOCT")

        for b_frame_number in processing_B_frames:
            OCT_bf = np.load(b_frames_OCT_dir + r'/OCT_{}.npy'.format(b_frame_number))
            three_d_OCT_list.append(OCT_bf)

        for b_frame_number in processing_B_frames:
            nsOCT_bf = np.load(b_frames_nsOCT_dir + r'/nsOCT_{}.npy'.format(b_frame_number))
            three_d_nsOCT_list.append(nsOCT_bf)

        three_d_OCT = np.asarray((three_d_OCT_list), dtype=np.float32)
        three_d_nsOCT = np.asarray((three_d_nsOCT_list), dtype=np.float32)

        three_d_OCT = three_d_OCT[:, :, A_lines_to_process[0] - 1:A_lines_to_process[1]]
        three_d_nsOCT = three_d_nsOCT[:, :, A_lines_to_process[0] - 1:A_lines_to_process[1]]

        print("The dimensions of OCT file loaded ", np.shape(three_d_OCT))
        print("The dimensions of nsOCT file loaded ", np.shape(three_d_nsOCT))

        threshold_spatial_period = np.min(three_d_nsOCT)

        print("The minimum spatial period is", np.min(three_d_nsOCT))
        print("The maximum spatial period is", np.max(three_d_nsOCT))

        compute_bins_prep = np.unique(three_d_nsOCT)
        compute_bins = np.sort(compute_bins_prep)[1:]

        print("the number of elements in the bins sequence", len(compute_bins))
        print("the minimum and maximum in the bins sequence are", np.min(compute_bins), np.max(compute_bins))

    Params.update({'Number_of_pixels': Number_of_pixels,
                   'Number_of_A_lines': Number_of_A_lines,
                   'Number_of_B_frames': Number_of_B_frames,
                   'Lateral_dimension_X': Lateral_dimension_X,
                   'Lateral_dimension_Y': Lateral_dimension_Y,
                   'Depth_in_mm': Depth_in_mm,
                   'B_frame_images': B_frame_images,
                   'enface_images': enface_images,
                   'Full_data_flattened': Full_data_flattened,
                   'B_frame_histograms': B_frame_histograms,
                   'threshold_spatial_period': threshold_spatial_period,
                   'enface_histograms': enface_histograms,
                   'is_exist': is_exist})

    return (three_d_OCT, three_d_nsOCT, compute_bins)



