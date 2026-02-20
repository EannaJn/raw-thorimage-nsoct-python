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
from scipy.ndimage import median_filter



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
    """Processes a single B-frame and returns processed OCT and nsOCT arrays (MATLAB Script 1 matched)."""

    # -----------------------
    # Load + DC removal (same idea as Script 1 Apodata_mean subtraction)
    # -----------------------
    b_frame_path = os.path.join(unzipped_folder_dir, f"data\\Spectral{b_frame_number}.data")
    b_frame = np.fromfile(b_frame_path, dtype='u2', count=-1).reshape(number_of_a_lines, number_of_pixels).astype(np.float32)

    mean_apo_spectrum = np.mean(apodization_a_lines.astype(np.float32), axis=0)  # shape (N,)
    dc_removed = b_frame - mean_apo_spectrum[None, :]  # shape (A_lines, N)

    # -----------------------
    # k-linearisation (match Script 1)
    # Script 1:
    #   K = k2 - Chirp*0.000436415
    #   k = linspace(k2,k1,2048)
    #   interp1(K, fringe, k, 'spline')
    # -----------------------
    k1 = 4.4463
    k2 = 5.3396
    K = (k2 - chirp_data.astype(np.float32) * 0.000436415)  # shape (N,)
    k_uniform = np.linspace(k2, k1, number_of_pixels).astype(np.float32)  # shape (N,)

    # interp1d needs strictly increasing x; Script 1's K can be decreasing depending on chirp.
    # So sort K and apply same permutation to spectra before interpolation.
    sort_idx = np.argsort(K)
    K_sorted = K[sort_idx]

    # Reorder spectra along pixel axis to match sorted K
    dc_sorted = dc_removed[:, sort_idx]  # (A_lines, N)

    # Spline interpolation along pixel axis (axis=1) onto uniform k grid
    interp_fun = ip.interp1d(
        K_sorted, dc_sorted,
        kind='cubic', axis=1,
        bounds_error=False, fill_value="extrapolate",
        assume_sorted=True
    )
    spec_lin = interp_fun(k_uniform)  # (A_lines, N)

    # -----------------------
    # Apodization window BEFORE IFFT (match Script 1)
    # Script 1 uses: w = cos(pi*depth_index/index_length) replicated across A-lines
    # depth_index = (1:N) - round(N/2)
    # -----------------------
    depth_index = (np.arange(number_of_pixels) + 1) - int(round(number_of_pixels / 2))
    w_cos = np.cos(np.pi * depth_index / number_of_pixels).astype(np.float32)  # (N,)
    spec_lin = spec_lin * w_cos[None, :]

    # -----------------------
    # IFFT -> OCT amplitude (match Script 1: ifft along depth axis)
    # Script 1: spec_fft = ifft(spec_lin,[],1); OCT_image = abs(spec_fft(1:image_depth,:))
    # In Python your data is (A_lines, N) so ifft axis=1
    # -----------------------
    spec_fft = np.fft.ifft(spec_lin, axis=1)
    OCT_image = np.abs(spec_fft)  # (A_lines, N)

    # -----------------------
    # Noise shaping / normalization (match Script 1)
    # Script 1:
    #   aver_fract=1/10; aver_pixel=round(image_depth*aver_fract)
    #   level_noise = sum(OCT_image(end-aver_pixel:end,:))/(aver_pixel+1)
    #   OCT_complex = OCT_image .* (1./level_noise) replicated down depth
    # -----------------------
    image_depth = number_of_pixels
    aver_pixel = int(round(image_depth * 0.1))
    tail = OCT_image[:, (image_depth - aver_pixel - 1):image_depth]  # last (aver_pixel+1) samples, (A_lines, tail_len)

    # level_noise per A-line (Script 1 does per column; our columns are A-lines)
    level_noise = np.sum(tail, axis=1) / (aver_pixel + 1)  # (A_lines,)
    level_noise = np.maximum(level_noise, 1e-12)

    # Normalize each A-line by its noise level
    OCT_norm = OCT_image / level_noise[:, None]  # (A_lines, N)

    Aline_max = np.max(OCT_norm)
    Aline_max = max(Aline_max, 1e-12)

    # -----------------------
    # Log mapping to [0..255] (match Script 1)
    # Script 1: OCT_log = (20*log10(OCT_complex/Aline_max)+dB_Range)/dB_Range*255;
    # then Contrast/Brightness clip
    # -----------------------
    dB_Range = float(getattr(params, "db_range", 75)) if hasattr(params, "db_range") else 75.0
    Contrast = float(getattr(params, "contrast", 1.0)) if hasattr(params, "contrast") else 1.0
    Brightness = float(getattr(params, "brightness", 0.0)) if hasattr(params, "brightness") else 0.0

    OCT_log = (20.0 * np.log10(OCT_norm / Aline_max + 1e-12) + dB_Range) / dB_Range * 255.0
    OCT_display = Contrast * OCT_log + Brightness
    OCT_display = np.clip(OCT_display, 0, 255).astype(np.uint8)  # (A_lines, N)

    # Script 1: subtract column mean (mean across depth per A-line) and median filter [3 3]
    # MATLAB line: OCT_display = OCT_display - ones(depth,1)*mean(OCT_display,1)
    col_mean = np.mean(OCT_display, axis=1, keepdims=True).astype(np.float32)  # (A_lines,1)
    OCT_display = (OCT_display.astype(np.float32) - col_mean)
    OCT_display = np.clip(OCT_display, 0, 255).astype(np.uint8)

    # 2D median filter
    OCT_display = median_filter(OCT_display, size=(3, 3))

    # Return OCT in the same orientation you used before (depth x A_lines)
    b_frame_oct = OCT_display.T.astype(np.float32)  # (N, A_lines)

    # -----------------------
    # === nsOCT (keep your Script 1-style implementation) ===
    # IMPORTANT: use spec_lin (cos-apodized, k-linearised) as nsOCT input to match MATLAB
    # Your existing nsOCT uses linear_data; switch to spec_lin to match Script 1.
    # -----------------------
    threshold = params['threshold']
    refractive_index = params["refractive_index"]
    num_window = params['number_of_windows']
    pixels_per_window = number_of_pixels // num_window
    half = number_of_pixels // 2

    Fringe_nsOCT = spec_lin.T  # (N, A_lines) like MATLAB Fringe_nsOCT = spec_lin

    # HZ axis from chirp like Script 1 (your existing form)
    k1_ns = 4.4463
    K_ns = k1_ns + chirp_data.astype(np.float32) * 0.000436415
    wavelength = (2 * np.pi / K_ns)[::-1]
    hz = (wavelength / (2 * refractive_index)) * 1e3  # nm

    twin = sg.windows.tukey(pixels_per_window, alpha=0.75).astype(np.float32)
    SpecC_C = np.zeros((half, number_of_a_lines, num_window), dtype=np.float32)

    for shr in range(num_window):
        win = np.zeros(number_of_pixels, dtype=np.float32)
        s = shr * pixels_per_window
        e = s + pixels_per_window
        win[s:e] = twin

        spec = Fringe_nsOCT * win[:, None]  # (N, A_lines)
        depth01 = np.fft.fftshift(np.fft.fft(spec, n=number_of_pixels, axis=0), axes=0)
        amp = depth01[:half, :]

        ii = np.log10(np.abs(amp) + 1e-12)
        ii = ndimage.median_filter(ii, size=(4, 4))
        SpecC_C[:, :, shr] = ii

    SpecC_C[SpecC_C < threshold] = 0
    zone_idx = np.argmax(SpecC_C, axis=2)

    center_index = zone_idx * pixels_per_window + (pixels_per_window // 2)
    center_index = np.clip(center_index, 0, len(hz) - 1).astype(int)

    b_frame_ns_oct = hz[center_index]
    b_frame_ns_oct = np.flipud(b_frame_ns_oct).astype(np.float32)

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