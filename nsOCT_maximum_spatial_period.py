import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import scipy.signal as sg
from scipy import interpolate as ip
from scipy.fftpack import fft
import os
import Params
import zipfile
import xmltodict
from Params import *
from window_function import select_window
from scipy import ndimage as ndimage


def _make_dirs(dirs):
    """Utility to create directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def _extract_oct_zip(f_name, extract_path):
    """Extracts the .oct zip file to the specified directory."""
    with zipfile.ZipFile(file=f_name, mode='r') as file:
        file.extractall(path=extract_path)

def _parse_header(header_path):
    """Parses the XML header and returns relevant imaging parameters."""
    with open(header_path, 'r') as telestoheader:
        xmlstring = telestoheader.read(-1)
    xmldict = xmltodict.parse(xmlstring)
    number_of_pixels = 2048
    number_of_a_lines = int(xmldict['Ocity']['DataFiles']['DataFile'][3]['@SizeX'])
    number_of_b_frames = int(xmldict['Ocity']['Image']['SizePixel']['SizeY'])
    lateral_dimension_x = float(xmldict['Ocity']['Image']['SizeReal']['SizeX'])
    lateral_dimension_y = float(xmldict['Ocity']['Image']['SizeReal']['SizeY'])
    depth_in_mm = float(xmldict['Ocity']['Image']['SizeReal']['SizeZ'])
    return number_of_pixels, number_of_a_lines, number_of_b_frames, lateral_dimension_x, lateral_dimension_y, depth_in_mm

def _load_apodization_and_chirp(unzipped_folder_dir, number_of_a_lines, number_of_pixels):
    """Loads apodization and chirp data from extracted files."""
    apodization_array_path = os.path.join(unzipped_folder_dir, "data\Spectral0.data")
    chirp_data_path = os.path.join(unzipped_folder_dir, "data\Chirp.data")
    apodization_array = np.fromfile(apodization_array_path, dtype='u2', count=-1)
    apodization_a_lines = apodization_array.reshape(number_of_a_lines, number_of_pixels)
    chirp_data = np.fromfile(chirp_data_path, dtype='float32', count=-1)
    return apodization_a_lines, chirp_data

def _process_b_frame(
    b_frame_number, unzipped_folder_dir, number_of_a_lines, number_of_pixels,
    apodization_a_lines, chirp_data, params
):
    """Processes a single B-frame and returns processed OCT and nsOCT arrays."""
    b_frame_path = os.path.join(unzipped_folder_dir, f"data\Spectral{b_frame_number}.data")
    b_frame = np.fromfile(b_frame_path, dtype='u2', count=-1).reshape(number_of_a_lines, number_of_pixels)
    mean_apo_spectrum = np.mean(apodization_a_lines, axis=0)
    dc_removed = b_frame - mean_apo_spectrum

    chirp_data_len = chirp_data.shape[0]
    interpolation_array = np.linspace(0, chirp_data_len - 1, chirp_data_len)
    interpolation_function = ip.interp1d(chirp_data, dc_removed, kind='linear', axis=1, fill_value="extrapolate")
    linear_data = interpolation_function(interpolation_array)

    # Apodization window
    apodize_method = "Hann"
    depth_index = np.arange(number_of_pixels) - number_of_pixels // 2
    window_band = 0.3 * number_of_pixels
    apodization_window = select_window(apodize_method, number_of_pixels, number_of_a_lines, depth_index, window_band)
    apodized_data = np.multiply(linear_data, apodization_window.T)

    # IFFT and normalization
    data_after_ifft = 20 * np.log10(np.abs(spfft.ifft(apodized_data)))
    prep_normalization = data_after_ifft + np.abs(np.min(data_after_ifft))
    normalized_data = ((prep_normalization / np.amax(prep_normalization)) * 255).T

    # nsOCT calculation
    lower_wavelength = params['lower_wavelength']
    upper_wavelength = params['upper_wavelength']
    wavelength = np.linspace(lower_wavelength, upper_wavelength, number_of_pixels)
    threshold = params['threshold']
    refractive_index = params["refractive_index"]
    hz = wavelength / (2 * refractive_index) * 1000  # nm
    num_window = params['number_of_windows']
    pixels_per_window = int(number_of_pixels / num_window)

    index_max_n_win_fft_arr = []

    tukey_window_short = sg.windows.tukey(pixels_per_window, alpha=0.75)
    tukey_window_full = np.pad(
        tukey_window_short,
        pad_width=(0, number_of_pixels - pixels_per_window),
        mode='constant'
    )
    tukey_windows = np.array([
        np.roll(tukey_window_full, int(i * abs(pixels_per_window)))
        for i in range(num_window)
    ]).T

    for aline in apodized_data:
        n_window_fft = np.log10(np.abs(fft(aline[:, None] * tukey_windows, axis=0)))
        n_window_fft = np.swapaxes(n_window_fft, 0, 1)
        n_window_fft[n_window_fft < threshold] = 0
        n_window_fft = ndimage.median_filter(n_window_fft, size=(4, 4))
        index_of_max_in_n_window_fft = np.argmax(n_window_fft, axis=0)
        index_max_n_win_fft_arr.append(index_of_max_in_n_window_fft)

    win4 = np.array(index_max_n_win_fft_arr).T
    win5a = []
    for w in win4.T:
        win5a.append(hz[np.round(w * int(abs(pixels_per_window))).astype(int)])
    win5 = np.array(win5a).T

    b_frame_oct = np.asarray(normalized_data, dtype=np.float32)
    b_frame_ns_oct = np.asarray(win5, dtype=np.float32)
    return b_frame_oct, b_frame_ns_oct

def oct_ns_oct_processing(f_name):
    """
    Reads a raw .oct file, unzips all data, performs
    OCT and nsOCT processing, and saves data for plotting.

    Args:
        f_name (str): raw .oct file name and path

    Returns:
        three_d_oct: Processed 3D OCT data
        three_d_ns_oct: Processed 3D nsOCT data
        compute_bins: bins to be used for histograms
    """
    root_dir = os.path.split(f_name)[0]
    file_base_name = os.path.splitext(os.path.split(f_name)[1])[0]
    output_base_dir = os.path.join(root_dir, file_base_name)
    processed_data_dir = os.path.join(output_base_dir, "processed_data")
    re_processed_data_dir = os.path.join(output_base_dir, "re_processed_data")
    unzipped_folder_dir = os.path.join(output_base_dir, "unzipped_folder")
    b_frames_oct_dir = os.path.join(output_base_dir, "b_frames_OCT")
    b_frames_ns_oct_dir = os.path.join(output_base_dir, "b_frames_nsOCT")

    _make_dirs([processed_data_dir, unzipped_folder_dir, b_frames_oct_dir, b_frames_ns_oct_dir])

    search_path = os.path.join(b_frames_ns_oct_dir, 'nsOCT_1.npy')
    is_exist = os.path.exists(search_path)

    if not is_exist:
        output_subfolders = [
            'b_frame_images', 'enface_images', 'b_frame_histograms', 'enface_histograms', 'full_data_flattened'
        ]
        subfolder_paths = [os.path.join(processed_data_dir, osf) for osf in output_subfolders]
        _make_dirs(subfolder_paths)
        b_frame_images, enface_images, b_frame_histograms, enface_histograms, full_data_flattened = subfolder_paths

        _extract_oct_zip(f_name, unzipped_folder_dir)
        header = os.path.join(unzipped_folder_dir, "Header.xml")
        number_of_pixels, number_of_a_lines, number_of_b_frames, lateral_dimension_x, lateral_dimension_y, depth_in_mm = _parse_header(header)
        apodization_a_lines, chirp_data = _load_apodization_and_chirp(unzipped_folder_dir, number_of_a_lines, number_of_pixels)

        print(f"Number of pixels per A-line: {number_of_pixels}")
        print(f"Number of A-lines per B-frame: {number_of_a_lines}")
        print(f"Number of B-frames: {number_of_b_frames}")
        print(f"Lateral dimension X: {lateral_dimension_x} mm")
        print(f"Lateral dimension Y: {lateral_dimension_y} mm")
        print(f"The imaging depth: {depth_in_mm} mm")

        b_frames_to_process = Params["b_frames_to_process"]
        a_lines_to_process = Params["a_lines_to_process"]
        processing_b_frames = np.arange(b_frames_to_process[0], b_frames_to_process[1] + 1)

        three_d_oct_list = []
        three_d_ns_oct_list = []

        for b_frame_number in processing_b_frames:
            print(f'Processing B-frame {b_frame_number}...')
            b_frame_oct, b_frame_ns_oct = _process_b_frame(
                b_frame_number, unzipped_folder_dir, number_of_a_lines, number_of_pixels,
                apodization_a_lines, chirp_data, Params
            )
            np.save(os.path.join(b_frames_oct_dir, f'OCT_{b_frame_number}.npy'), b_frame_oct)
            np.save(os.path.join(b_frames_ns_oct_dir, f'nsOCT_{b_frame_number}.npy'), b_frame_ns_oct)
            three_d_oct_list.append(b_frame_oct)
            three_d_ns_oct_list.append(b_frame_ns_oct)

        three_d_oct = np.asarray(three_d_oct_list, dtype=np.float32)
        three_d_ns_oct = np.asarray(three_d_ns_oct_list, dtype=np.float32)
        three_d_oct = three_d_oct[:, :, a_lines_to_process[0] - 1:a_lines_to_process[1]]
        three_d_ns_oct = three_d_ns_oct[:, :, a_lines_to_process[0] - 1:a_lines_to_process[1]]

        compute_bins_prep = np.linspace(np.min(three_d_ns_oct), np.max(three_d_ns_oct), 101)
        compute_bins = np.sort(compute_bins_prep)[1:]
        threshold_spatial_period = np.min(three_d_ns_oct)

        print(f"The dimension of three_d_oct is {three_d_oct.shape}")
        print(f"The dimension of three_d_ns_oct is {three_d_ns_oct.shape}")
        print(f"The minimum spatial period is {np.min(three_d_ns_oct)}")
        print(f"The maximum spatial period is {np.max(three_d_ns_oct)}")
        print(f"The number of elements in the bins sequence: {len(compute_bins)}")

    else:
        print("Data already processed")
        output_subfolders = [
            'b_frame_images', 'enface_images', 'b_frame_histograms', 'enface_histograms',
            'full_data_flattened'
        ]
        subfolder_paths = [os.path.join(re_processed_data_dir, osf) for osf in output_subfolders]
        _make_dirs(subfolder_paths)
        b_frame_images, enface_images, b_frame_histograms, enface_histograms, full_data_flattened = subfolder_paths

        _extract_oct_zip(f_name, unzipped_folder_dir)
        header = os.path.join(unzipped_folder_dir, "Header.xml")
        number_of_pixels, number_of_a_lines, number_of_b_frames, lateral_dimension_x, lateral_dimension_y, depth_in_mm = _parse_header(header)

        Params.update({"oct_b_frames": b_frame_images})

        print(f"Number of pixels: {number_of_pixels}")
        print(f"Number of A-lines: {number_of_a_lines}")
        print(f"Number of B-frames: {number_of_b_frames}")
        print(f"Lateral dimension X: {lateral_dimension_x}")
        print(f"Lateral dimension Y: {lateral_dimension_y}")
        print(f"The depth in mm: {depth_in_mm}")

        b_frames_to_process = Params["b_frames_to_process"]
        a_lines_to_process = Params["a_lines_to_process"]
        processing_b_frames = np.arange(b_frames_to_process[0], b_frames_to_process[1] + 1)

        three_d_oct_list = []
        three_d_ns_oct_list = []

        for b_frame_number in processing_b_frames:
            oct_bf = np.load(os.path.join(b_frames_oct_dir, f'OCT_{b_frame_number}.npy'))
            three_d_oct_list.append(oct_bf)
            ns_oct_bf = np.load(os.path.join(b_frames_ns_oct_dir, f'nsOCT_{b_frame_number}.npy'))
            three_d_ns_oct_list.append(ns_oct_bf)

        three_d_oct = np.asarray(three_d_oct_list, dtype=np.float32)
        three_d_ns_oct = np.asarray(three_d_ns_oct_list, dtype=np.float32)
        three_d_oct = three_d_oct[:, :, a_lines_to_process[0] - 1:a_lines_to_process[1]]
        three_d_ns_oct = three_d_ns_oct[:, :, a_lines_to_process[0] - 1:a_lines_to_process[1]]

        print(f"The dimensions of OCT file loaded: {three_d_oct.shape}")
        print(f"The dimensions of nsOCT file loaded: {three_d_ns_oct.shape}")

        threshold_spatial_period = np.min(three_d_ns_oct)
        print(f"The minimum spatial period is {np.min(three_d_ns_oct)}")
        print(f"The maximum spatial period is {np.max(three_d_ns_oct)}")

        compute_bins_prep = np.unique(three_d_ns_oct)
        compute_bins = np.sort(compute_bins_prep)[1:]
        print(f"The number of elements in the bins sequence: {len(compute_bins)}")
        print(f"The minimum and maximum in the bins sequence are {np.min(compute_bins)}, {np.max(compute_bins)}")

    Params.update({
        'number_of_pixels': number_of_pixels,
        'number_of_a_lines': number_of_a_lines,
        'number_of_b_frames': number_of_b_frames,
        'lateral_dimension_x': lateral_dimension_x,
        'lateral_dimension_y': lateral_dimension_y,
        'depth_in_mm': depth_in_mm,
        'b_frame_images': b_frame_images,
        'enface_images': enface_images,
        'full_data_flattened': full_data_flattened,
        'b_frame_histograms': b_frame_histograms,
        'threshold_spatial_period': threshold_spatial_period,
        'enface_histograms': enface_histograms,
        'is_exist': is_exist
    })

    return three_d_oct, three_d_ns_oct, compute_bins