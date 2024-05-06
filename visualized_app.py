import os
import streamlit as st
import cv2
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import glob
import re
import matplotlib.dates as mdates
import math
import sys
# sys.path.append("../PhyQual/wifi/")
# import parser
import struct
import time
import queue
import time
import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from motion_detect import detect_motion, convert_to_real_timestamps
import json
import subprocess
from PIL import Image
import io

# from metric.motion_statistics import cal_ms
# data_path = "/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/wifi/exp-20240229161513_node_1_modality_wifi_subject_2_activity_combo_trial_1/exp-20240229161513_node_1_modality_wifi_subject_2_activity_combo_trial_1_20240130103631.csi"

# %%
def bytes_arr_to_complex_(v):
    v_complex = []
    for i in range(0, len(v), 4):
        try:
            R = v[i] if v[i+1]==0 else -(256-v[i])
            I = v[i+2] if v[i+3]==0 else -(256-v[i+2])
            comp = complex(R, I)
            v_complex.append(comp)
        except Exception as e:
            print("parse error, break")
            break
    return v_complex

def parse_header(header):
    def parse_from_range(header, start, end, dtype="int"):
        # print("start: ", start, "end: ", end)
        # print(header[start:end])
        if dtype == "int":
            return int.from_bytes(header[start:end], byteorder='little')
        elif dtype == "mac":
            b = header[start:end]
            mac_str = ':'.join(['{:02X}'.format(byte) for byte in b])
            return mac_str
        elif dtype == "hex2dec":
            return int.from_bytes(header[start:end], byteorder='little', signed=False)
    result = {}
    
    result["n_link"] = parse_from_range(header, 50, 51, dtype="hex2dec")
    result["n_subc"] = parse_from_range(header, 52, 56, dtype="hex2dec")
    result["rssi_a"] = parse_from_range(header, 60, 64, dtype="hex2dec")
    result["rssi_b"] = parse_from_range(header, 64, 68, dtype="hex2dec")
    result["mac_src"] = parse_from_range(header, 68, 74, dtype="mac")
    result["seq"] = parse_from_range(header, 76, 77, dtype="hex2dec")
    result['timestamp'] = parse_from_range(header, 88, 92, dtype="int")
    return result

# %%
def read_csi_all(file_path):
    csi_all = []
    ts_all = []
    header_all = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                # Read time
                ts_b = f.read(19).decode('utf-8')
                time = int(ts_b)
                ts_all.append(time)
                
                # Read header length, data_num, and data_len
                hdr_len, data_num, data_len = struct.unpack('>III', f.read(12))

                # Read header
                header = f.read(hdr_len)
                header_all.append(header)
                
                # Read CSI data
                csi_data = []
                for _ in range(data_num):
                    data = f.read(int(data_len / data_num))
                    csi_data.append(data)

                csi_all.append(csi_data)
            except Exception as e:
                # print(e)
                break
    return csi_all, ts_all, header_all

def read_csi_data_by_shape(file_path, csi_shape=None, mac_filter=None):
    csi_all, ts_all, header_all = read_csi_all(file_path)
    
    header_all_parsed = [parse_header(header) for header in header_all]

    # Convert CSI data to complex numbers for each record
    for i, csi_data in enumerate(csi_all):
        csi_data_complex = [bytes_arr_to_complex_(data) for data in csi_data]
        csi_all[i] = csi_data_complex

    
    csi_all_cleaned = []
    ts_all_cleaned = []
    header_all_cleaned = []
    
    for i, csi in enumerate(csi_all):
        csi = np.array(csi)
        # print(header_all_parsed[i])
        if (csi_shape and csi.shape != csi_shape):
            continue
        if (mac_filter and header_all_parsed[i]["mac_src"].upper() != mac_filter.upper()):
            continue
        csi_all_cleaned.append(csi)
        ts_all_cleaned.append(ts_all[i])
        header_all_cleaned.append(header_all_parsed[i])

    # print("shape of csi_all_cleaned:", csi_all_cleaned.shape)
    # print("Number of records:", len(csi_all))
    return csi_all_cleaned, ts_all_cleaned, header_all_cleaned

# %%
def cal_ms(x, time_dim=0, subc_dim=2, raw=True):
    """
    Calculate the Motion Statistics of a signal.
    
    INPUTS:
        x: signal, [time, channel]
        time_dim: time dimension
    OUTPUTS:
        ms: ms of the signal
    """

    if raw:
        x = np.abs(x)**2
        # L1 normalization on each subcarrier
        x = x / np.sum(x, axis=subc_dim, keepdims=True)
        
    S = x.shape

    if time_dim > len(S):
        raise ValueError("time_dim is larger than the number of dimensions of x")

    S_output = S[:time_dim] + S[time_dim+1:]

    if len(S_output) == 1:
        if time_dim == 1:
            S_output = (1,) + S_output
        else:
            S_output = S_output + (1,)
    # calculate N of element before and after time_dim
    left_dims = int(np.prod(S[:time_dim]))
    right_dims = int(np.prod(S[time_dim+1:]))

    x = np.reshape(x, (left_dims, S[time_dim], right_dims))
    x = np.transpose(x, (1, 0, 2))
    x = np.reshape(x, (S[time_dim], left_dims * right_dims))

    ms = np.zeros((left_dims * right_dims))
    for i in range(left_dims * right_dims):
        ms[i] = cal_ms_1d(x[:, i])

    ms = np.reshape(ms, (1, left_dims, right_dims))
    ms = np.transpose(ms, (1, 0, 2))
    ms = np.reshape(ms, S_output)

    return np.squeeze(ms)

def cal_ms_1d(x):
    """
    Calculate the Motion Statistics of a 1-dim signal.
    
    INPUTS:
        x: signal
    OUTPUTS:
        ms: ms of the signal
    """
    y = x - np.nanmean(x)
    acf = np.zeros(2)
    for j in range(2):
        cross = y[:-j or None] * y[j:]
        iNonNaN = ~np.isnan(cross)

        if np.any(iNonNaN):
            T = np.sum(iNonNaN) + np.sum(~np.isnan(y[:j]))
            acf[j] = np.nansum(cross) / T

    ms = acf[1] / acf[0]
    return ms

# %%
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

# from metric.motion_statistics import cal_ms

def nextpow2(i):
    return np.ceil(np.log2(i))
def autocorr(x):
    x = x - np.mean(x)
    nFFT = int(2**(nextpow2(len(x))+1))
    F = np.fft.fft(x,nFFT)
    F = F*np.conj(F)
    acf = np.fft.ifft(F)
    acf = acf[0:len(x)] # Retain nonnegative lags
    acf = np.real(acf)
    acf = acf/acf[0] # Normalize
    return acf

def peak_detect(y, packet_rate, min_MS, motion_inference_ratio = 0.5):
    import matplotlib.pyplot as plt

    if y.shape[0] != 1:
        y = y.T
    win_size = len(y)

    upsample_rate = 1
    max_MS = 1
    
    locs_sel = None
    pks_sel = None
    # [win_size-1 ... 1 1... win_size-1]
    to_interp_y = np.concatenate((np.flip(y[1:win_size]), y[1:win_size]))    
    to_interp_x = np.array(list(np.flip(-1 * np.arange(1, win_size))) + list(np.arange(1, win_size)))
    # from 1 to win_size, step 1/upsample_rate
    interp_x_half = np.linspace(1, win_size-1, num=(win_size-2)*upsample_rate+1)
    
    interp_x = np.concatenate((np.flip(-1*interp_x_half), interp_x_half))
    interp_y = np.interp(interp_x, to_interp_x, to_interp_y)

    half_rate = packet_rate * upsample_rate / 2
    low_order = 2
    low_pass = 10
    lb, la = butter(low_order, low_pass / half_rate, btype="low")
    
    lowpass_acf = filtfilt(lb, la, interp_y)
    
    # bt_bpm = [0, 60 * 10]
    # bt_cycle = np.divide(60, bt_bpm)
    # bt_ts = bt_cycle

    to_detect_y = lowpass_acf[((win_size - 2) * upsample_rate)+1:]
    to_detect_x = interp_x[((win_size - 2) * upsample_rate)+1:] / packet_rate
    MS = to_detect_y[0]
    minPeakHeight = MS * motion_inference_ratio

    if min_MS <= MS and MS <= max_MS:
        # plt.plot(interp_y)
        # plt.plot(lowpass_acf)
        # plt.show()
        peaks, _ = find_peaks(to_detect_y, height=minPeakHeight)
        locs = to_detect_x[peaks]
        pks = to_detect_y[peaks]

        # constraint = (locs >= bt_ts[1]) & (locs <= bt_ts[0])
        # locs = locs[constraint]
        # pks = pks[constraint]

        if len(locs) > 0:
            locs_sel = locs[0]
            pks_sel = pks[0]

    return locs_sel

def breath_detection(csi_nWin_nLink_nSubc, fs, win_size, win_step):
    """Breath detection algorithm

    Args:
        csi_nWin_nLink_nSubc (numpy array): (nWin, nLink, nSubc)
        fs (int): frame rate
        win_size (int): window size
        win_step (int): window step

    Returns:
        result: dict
    """
    csi_raw = csi_nWin_nLink_nSubc
    csi_len = csi_raw.shape[0]

    k_top_ratio = 0.1
    n_subc = csi_raw.shape[1]

    win_start_list = np.arange(0, csi_len - win_size + 1, win_step)
    win_num = len(win_start_list)
    peak_locs_all = np.zeros(win_num)
    ms_all = np.zeros((win_num, n_subc))

    agg_acf_all = np.zeros((win_num, win_size))
    acf_all = np.zeros((win_num, win_size, n_subc))
    for win_i in range(win_num):
        win_start = win_start_list[win_i]

        csi_win = csi_raw[win_start:win_start+win_size, :]
        csi_mag_power = np.abs(csi_win)**2
        csi_mag_power_norm = csi_mag_power / np.sum(csi_mag_power, axis=1, keepdims=True)
        
        acf = np.zeros(csi_mag_power_norm.shape)
        ms = cal_ms(csi_mag_power_norm, raw=False)
        ms[np.isnan(ms)] = 0
        k_top = ms
        if np.max(k_top) > 0:
            k_top = k_top * (k_top > 0)
        else:
            k_top[k_top < 0] = 1e-6
        # select top k-ratio*subc subcarriers
        k_top_idx = np.argpartition(k_top, -int(n_subc * k_top_ratio))[-int(n_subc * k_top_ratio):]
        # if idx not in k_top_idx, set to 0
        mask = np.ones(k_top.shape, dtype=bool)
        mask[k_top_idx] = False
        k_top[mask] = 0
        
        
        # for i_subc in k_top_idx:
        #     num_lags = win_size - 1
        #     acf[:num_lags+1, i_subc] = autocorr(csi_mag_power_norm[:, i_subc])
        
        for i_subc in range(n_subc):
            num_lags = win_size - 1
            acf[:num_lags+1, i_subc] = autocorr(csi_mag_power_norm[:, i_subc])
        
        acf_all[win_i, :, :] = acf[:, :]
        # remove nan in acf, replaced with 0
        acf[np.isnan(acf)] = 0
        
        k_top = k_top / np.sum(k_top)
        ms_all[win_i] = ms

        agg_acf = np.dot(acf, k_top.T)
        agg_acf_all[win_i, :] = agg_acf

        peak_loc = peak_detect(agg_acf, fs, 0.2)

        if peak_loc == None:
            peak_loc = np.nan
        peak_locs_all[win_i] = peak_loc

    detected_num = np.sum(peak_locs_all > 0, axis=0, where=~np.isnan(peak_locs_all))
    output = {
        'dr': detected_num / win_num,
        'bpm_est_raw': 60. / peak_locs_all,
        'bpm_est': np.nanmean(60. / peak_locs_all),
        'ms': ms_all,
        'agg_acf_all': agg_acf_all,
        'all_acf': np.array(acf_all),
        'k_top': k_top
    }
    return output



def process_csi_and_detect_breath(csi_file_path):
    # Assuming read_csi_data_by_shape is defined elsewhere and accessible
    csi_all, ts_all, header_all = read_csi_data_by_shape(csi_file_path, csi_shape=(2, 114), mac_filter="A4:A9:30:B1:AF:7D")
    # csi_all, ts_all, header_all = read_csi_data_by_shape(csi_file_path, csi_shape=(2, 484), mac_filter="A4:A9:30:B1:AF:7D")
    # csi_all, ts_all, header_all = read_csi_data_by_shape(csi_file_path, csi_shape=(1, 56), mac_filter="C8:BF:4C:74:39:41")
    # Clean data based on RSSI
    csi_all_cleaned = []
    ts_all_cleaned = []
    header_all_parsed_cleaned = []
    for i, header in enumerate(header_all):
        if header["rssi_a"] < 180:
            csi_all_cleaned.append(csi_all[i])
            ts_all_cleaned.append(ts_all[i])
            header_all_parsed_cleaned.append(header)
    # BW = "160MHz"
    BW = "40MHz"
    # Define subcarrier deletion based on bandwidth
    if BW == "160MHz":
        del_subc = [19, 47, 83, 111, 130, 158, 194, 222, 275, 303, 339, 367, 386, 414, 450, 478, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
    elif BW == "40MHz":
        del_subc = [5, 33, 47, 66, 80, 108]
    elif BW == "20MHz":
        del_subc = [7, 21, 34, 48]
    else:
        del_subc = []
    
    ts_trim = np.array(ts_all_cleaned[200:])
    csi_trim = np.array(csi_all_cleaned[200:])
    csi_trim = np.delete(csi_trim, del_subc, axis=2)

    # Compute sampling rate
    ts_diff_mean = np.mean(ts_trim[-1] - ts_trim[0]) / len(ts_trim)
    fs = 1e9 / ts_diff_mean
    fs_round = int(np.round(fs))
    win_size = fs_round * 8
    win_step = fs_round

    # Flatten CSI for further analysis
    csi_3000_flatten = np.reshape(csi_trim[:, 0, :], (np.shape(csi_trim)[0], -1))

    # Save the flattened CSI data to a .mat file
    sio.savemat('csi_3000_flatten.mat', {'csi_3000_flatten': csi_3000_flatten})

    # Apply breath detection
    result_all = breath_detection(csi_3000_flatten, fs_round, win_size, win_step)

    # Extract and report top k indices
    # k_top = result_all["k_top"]
    # k_top_idx = np.argwhere(k_top > 0)
    
    return result_all, csi_trim

# Example usage:
# result_all, csi_trim = process_csi_and_detect_breath(data_path, (1, 114), "E4:5F:01:E5:7A:E9", "40MHz")


# Function to extract the numerical part from the folder name
def extract_number(folder_name):
    match = re.search(r'\d+$', folder_name)
    return int(match.group()) if match else 0

# Function to load and parse timestamps
def load_timestamps(file_path):
    with open(file_path, 'r') as file:
        timestamps = [datetime.strptime(line.strip(), "%Y-%m-%d %H:%M:%S.%f") for line in file if line.strip()]
    return timestamps

# Assuming calculate_start_time function is defined as before, adjusted to return seconds as a float for higher precision
def calculate_start_time(timestamps, selected_timestamp):
    start_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S.%f")
    selected_time = datetime.strptime(selected_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    return (selected_time - start_time).total_seconds()

def extract_frame(video_path, timestamp):
    """Extracts a frame from the video at the specified timestamp and returns it as a PIL image."""
    cmd = [
        'ffmpeg',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        '-f', 'image2pipe',  # Output format to pipe
        '-vcodec', 'mjpeg',  # Output codec
        'pipe:1'  # Use standard output as output location
    ]
    # Run the command and capture the output image data
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    image_data = io.BytesIO(result.stdout)
    image = Image.open(image_data)
    return image
# Calculate start_time in seconds from the beginning of the video
# def calculate_start_time_video(timestamps, selected_timestamp):
#     # Ensure start_time is converted to datetime
#     start_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S.%f")
#     # Ensure selected_time is converted to datetime
#     selected_time = datetime.strptime(selected_timestamp, "%Y-%m-%d %H:%M:%S.%f")
#     # Calculate and return the difference in seconds as an integer
#     return int((selected_time - start_time).total_seconds())
# Function to convert bytes to a more readable format
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

# Define the SubpageInterpolating function
def SubpageInterpolating(subpage):
    shape = subpage.shape
    mat = subpage.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mat[i,j] > 0.0:
                continue
            num = 0
            try:
                top = mat[i-1,j]
                num += 1
            except IndexError:
                top = 0.0
            
            try:
                down = mat[i+1,j]
                num += 1
            except IndexError:
                down = 0.0
            
            try:
                left = mat[i,j-1]
                num += 1
            except IndexError:
                left = 0.0
            
            try:
                right = mat[i,j+1]
                num += 1
            except IndexError:
                right = 0.0
            
            mat[i,j] = (top + down + left + right)/num if num > 0 else 0.0
    return mat

# Define the function to load data from a pickle file
def load_data_from_pickle(pickle_file):
    data = []
    with open(pickle_file, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass
    return data

# Define the function to display a frame
def display_frame_ira(data, index):
    if index >= len(data):
        st.write(f"Invalid index. Please choose a value between 0 and {len(data) - 1}.")
        return
    
    frame_data = data[index]
    st.write(frame_data)
    st.write("frame_data['Detected_Temperature'].shape", frame_data['Detected_Temperature'].shape)
    timestamp = frame_data['timestamp']
    Detected_temperature = np.array(frame_data['Detected_Temperature']).reshape((24,32))
    ira_interpolated = SubpageInterpolating(Detected_temperature)
    ira_norm = ((ira_interpolated - np.min(ira_interpolated)) / (37 - np.min(ira_interpolated))) * 255
    ira_expand = np.repeat(ira_norm, 20, 0)
    ira_expand = np.repeat(ira_expand, 20, 1)
    ira_img_colored = cv2.applyColorMap(ira_expand.astype(np.uint8), cv2.COLORMAP_JET)

    # Convert to RGB and display using Streamlit
    st.image(cv2.cvtColor(ira_img_colored, cv2.COLOR_BGR2RGB), caption=f"Timestamp: {timestamp}")

# ToF Visualization Function
def display_frame_tof(data, index):
    if index >= len(data):
        st.write(f"Invalid index. Please choose a value between 0 and {len(data) - 1}.")
        return

    frame_data = data[index]
    timestamp = frame_data['timestamp']
    tof_depth = np.array(frame_data['tof_depth'])
    tof_bins = np.array(frame_data['tof_bins'])
    st.write(frame_data, "tof_depth.shape:", tof_depth.shape, "tof_bins.shape:", tof_bins.shape)
    if tof_depth is not None:
        # Process tof_depth
        vis_data = cv2.resize(tof_depth, (400, 400), interpolation=cv2.INTER_NEAREST)
        vis_data = (vis_data / 3500) * 255
        vis_data[vis_data > 255] = 255
        vis_data = cv2.applyColorMap(vis_data.astype(np.uint8), cv2.COLORMAP_JET)
        vis_data = np.flip(vis_data, 0)

        # Process tof_bins
        bin_data = (tof_bins / tof_bins.max()) * 255
        bin_data[bin_data > 255] = 255
        bin_data = cv2.applyColorMap(bin_data.astype(np.uint8), cv2.COLORMAP_JET)
        bin_data = cv2.resize(bin_data, (400, 400), interpolation=cv2.INTER_NEAREST)

        # Combine and show data
        show_data = cv2.hconcat([vis_data, bin_data])

        # Convert to RGB and display using Streamlit
        st.image(cv2.cvtColor(show_data, cv2.COLOR_BGR2RGB), caption=f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')}")

# Define a function for 3D scatter plot
def plot_3d_scatter_mmwave(data, index):
    if index >= len(data):
        st.write(f"Invalid index. Please choose a value between 0 and {len(data) - 1}.")
        return
    
    data_entry = data[index]
    timestamp = data_entry['timestamp']
    frame_data = data_entry['data']
    st.write(data_entry)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = -frame_data['x']
    y = frame_data['y']
    z = frame_data['z']

    scatter = ax.scatter(x, y, z)

    # Set axis limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 5)
    ax.set_zlim(-5, 5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f'3D Scatter Plot at Timestamp: {timestamp}')

    st.pyplot(fig)



# # UWB: Function to load data from a selected pickle file
def load_data_from_pickle_uwb(file_path):
    data = {}
    with open(file_path, 'rb') as file:
        while True:
            try:
                frame_data = pickle.load(file)
                timestamp = frame_data["timestamp"]
                data[timestamp] = frame_data  # adjust based on your data structure
            except EOFError:
                break
    return data


def list_sorted_wav_files(directory):
    # List all subdirectories in the main directory
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    # Sort these directories by modification time in descending order
    subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    
    # For each subdir, find the .wav file inside it
    wav_files = []
    for subdir in subdirs:
        # Get all .wav files in the current subdir
        wav_file_paths = glob.glob(os.path.join(directory, subdir, '*.wav'))
        # If there are .wav files, extend the wav_files list with them
        if wav_file_paths:
            wav_files.extend(wav_file_paths)

    return wav_files

# Function to render the acoustic data visualization
def render_acoustic_visualization(node_dir):
    st.subheader("Acoustic Data Visualization")
    acoustic_dir = f'{node_dir}/acoustic/data'
    wav_files = list_sorted_wav_files(acoustic_dir)
    if wav_files:
        selected_wav_file = st.selectbox('Select an acoustic file:', wav_files, index=0)
        # Display file size for Acoustic file
        acoustic_file_size = convert_size(os.path.getsize(selected_wav_file))
        st.write(f"Selected Acoustic file size: {acoustic_file_size}")
        audio_file = open(selected_wav_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
    else:
        st.write("No acoustic files found.")


# [All your defined functions like load_data_from_pickle, display_frame_ira, etc. go here]

# Function to render visualizations for a specific node
def render_node_visualizations(node_dir):
    st.header(f"Visualizations for Node: {node_dir}")
    # Wifi visualization
    # Step 1: Select a Directory
    exp_dirs = glob.glob(os.path.join(node_dir, "wifi/exp-*"))
    exp_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # Sort by modification time, newest first
    selected_dir = st.selectbox('Select an experiment directory:', exp_dirs)
    if selected_dir:
    # Step 2: Automatically locate the .csi file within the selected directory
        csi_files = glob.glob(os.path.join(selected_dir, "*.csi"))
        if csi_files and len(csi_files) == 1:
            csi_file_path = csi_files[0]
            file_size = convert_size(os.path.getsize(csi_file_path))
            st.write(f"Selected CSI file: {csi_file_path}")
            st.write(f"File size: {file_size}")
        
        # Proceed with processing the selected WiFi file
        st.subheader("Wifi Visualization")
        # st.subheader("MS")
        result_all, csi_trim = process_csi_and_detect_breath(csi_file_path)
        # # Transposing the matrix to swap the axes
        # fig, ax = plt.subplots()
        # # Use the .T attribute to transpose the matrix for the heatmap
        # cax = ax.imshow(result_all['ms'].T, aspect="auto", cmap="jet")  # Transpose with .T
        # fig.colorbar(cax)

        # # After transposing, the axes labels are swapped
        # ax.set_xlabel("Time")  # Now represents the horizontal axis
        # ax.set_ylabel("Subcarrier")  # Now represents the vertical axis

        # st.pyplot(fig)

        # Visualization 2: Plot of the mean MS value across subcarriers over time
        st.subheader("Mean MS Value Over Time")
        fig, ax = plt.subplots()
        ax.plot(np.mean(result_all['ms'], axis=1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean MS across subcarriers")
        st.pyplot(fig)

        # Visualization 3: Plot of the CSI amplitude for the first 100 samples across all subcarriers
        st.subheader("CSI Amplitude")
        fig, ax = plt.subplots(figsize=(20, 3))  # Adjusted for better display in Streamlit
        ax.plot(np.abs(csi_trim[::20, 0, :]).T)
        ax.set_xlabel("Subcarrier index")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # st.subheader("ACF Heatmap")
        # fig, ax = plt.subplots()
        # cax = ax.imshow(result_all["all_acf"][12, :, :], aspect='auto', cmap='jet')
        # fig.colorbar(cax)
        # ax.set_xlabel("Subcarrier Index")
        # ax.set_ylabel("Lag")
        # st.pyplot(fig)

        st.subheader("agg_acf_all shape")
        fig, ax = plt.subplots()
        cax = ax.imshow(result_all["agg_acf_all"], aspect='auto', origin='lower', vmin=-0.05, vmax=0.05, cmap='jet')
        fig.colorbar(cax)
        st.pyplot(fig)
    else:
        st.write("No WiFi files found.")

    # IRA Visualization
    st.subheader("IRA Visualization")
    ira_output_dir = f'{node_dir}/IRA/data'  # Update this to your IRA directory path
    ira_files = glob.glob(os.path.join(ira_output_dir, '*.pickle'))
    if ira_files:
        ira_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        ira_file_path = st.selectbox('Select an IRA file:', ira_files, key='ira_select')
        file_size = convert_size(os.path.getsize(ira_file_path))
        st.write(f"Selected file size: {file_size}")
        ira_data = load_data_from_pickle(ira_file_path)
        st.write(f"Loaded {len(ira_data)} frames for IRA.")
        ira_index = st.slider("Select Frame Index for IRA", 0, len(ira_data) - 1, 0, key='ira_slider')
        display_frame_ira(ira_data, ira_index)
    else:
        st.write("No IRA files found.")
    
    # mmWave Visualization
    st.subheader("mmWave Visualization")
    st.header("mmWave Visualization")
    mmwave_output_dir = f'{node_dir}/mmWave/data'  # Update this to your mmWave directory path
    mmwave_files = glob.glob(os.path.join(mmwave_output_dir, '*.pickle'))
    if mmwave_files:
        mmwave_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        mmwave_file_path = st.selectbox('Select an mmWave file:', mmwave_files, key='mmwave_select')
        mmwave_data = load_data_from_pickle(mmwave_file_path)
        st.write(f"Loaded {len(mmwave_data)} frames for mmWave.")
        mmwave_index = st.slider("Select Frame Index for 3D Scatter", 0, len(mmwave_data) - 1, 0, key='mmwave_slider')
        plot_3d_scatter_mmwave(mmwave_data, mmwave_index)
    else:
        st.write("No mmWave files found.")

    # Depth Cam Visualization
    st.subheader("Depth Cam Visualization")
    base_depthcam_dir = f'{node_dir}/depthCamera/data'  # Adjusted to new base directory for Depth Cam

    def list_folders(base_directory):
        """List all subdirectories in the base directory."""
        folders = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
        folders.sort(reverse=True)  # Sort in descending order
        return folders

    def png_to_temperature(filepath, min_temp, max_temp):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
        if img is None:
            raise ValueError("Image not found or path is incorrect")

        img = img.astype(np.float32)  # Convert to float for precise calculations
        temperature = (img / 255.0) * (max_temp - min_temp) + min_temp
        return temperature
    def display_image(file_path):
        """Display an image using Streamlit if the file exists."""
        if os.path.exists(file_path):
            image = Image.open(file_path)
            st.image(image, use_column_width=True)
        else:
            st.error("Image file not found.")

    def list_png_files(directory):
        """List .png files in the directory."""
        return [f for f in os.listdir(directory) if f.endswith('.png') and 'depth' in f]


    # List folders that potentially contain depth data
    folders = list_folders(base_depthcam_dir)
    folders.sort(reverse=True)

    if folders:
        selected_folder = st.selectbox('Select a folder with depth data:', folders, key='folder_select')
        
        # List all PNG files in the selected folder
        depth_files = list_png_files(selected_folder)
        depth_files.sort(reverse=False)
        
        if depth_files:
            selected_file = st.selectbox('Select a depth data file:', depth_files, key='file_select')
            selected_file_path = os.path.join(selected_folder, selected_file)
            
            # Display the selected image
            st.write(f"Selected Depth Image: {selected_file}")
            display_image(selected_file_path)
            recovered_depth = png_to_temperature(selected_file_path, 0, 10)
            st.write(recovered_depth)
        else:
            st.write("No depth images found in the selected folder.")
    else:
        st.write("No folders found.")

    # 
    #     # Extract timestamps from the filenames
    #     timestamps = sorted(set([os.path.basename(f).split('_')[1].split('.')[0] for f in depth_files]))
    #     timestamp_file_path = selected_depth_video_path.replace('.mp4', '.txt')
    
    #     if os.path.exists(timestamp_file_path):
    #         with open(timestamp_file_path, 'r') as f:
    #             timestamps_depth = [ts.strip() for ts in f.readlines()]
            
    #         st.write("Depth Image:")
    #         display_image(selected_depth_video_name)
        
        #     if timestamps_depth:
        #         if os.path.exists(selected_depth_video_path):
        #             depth_video_file_size = convert_size(os.path.getsize(selected_depth_video_path))
        #             st.write(f"Selected Depth video file size: {depth_video_file_size}")
        #         selected_timestamp = st.select_slider('Select a timestamp to start from:', options=timestamps_depth, key='timestamp_select')
        #         start_time_seconds = calculate_start_time(timestamps_depth, selected_timestamp)
                # Use FFmpeg to extract the frame
                # frame_path = extract_frame(selected_depth_video_path, start_time_seconds)
                # # Load the image directly from the path
                # gray_image = Image.open(frame_path).convert('L')
                # depth_array = np.array(gray_image, dtype=np.float32)  # Ensure float32 for accurate calculations
                # min_temp = 0
                # max_temp = 10
                # temp_range = max_temp - min_temp
                # temp_approx = (depth_array / 255) * temp_range + min_temp
                # st.image(gray_image)  # Display grayscale image
                # st.write(temp_approx)  # Display temperature data
                # st.write("Image shape:", temp_approx.shape)
        #     else:
        #         st.write("Depth video file not found.")
        # else:
        #     st.write("Timestamp file not found.")
    # else:
    #     st.write("No depth video files found.")

    # RGB Video Visualization
    st.subheader("RGB Video Visualization")
    base_rgb_dir = f'{node_dir}/depthCamera/data'  # Adjusted to new base directory for RGB videos

    # Select an RGB video file
    rgb_videos = [video for video in glob.glob(os.path.join(base_rgb_dir, '*', '*rgbcam*.mp4'))]  # Adjusted to include identifier for RGB videos
    rgb_video_names = [os.path.basename(video) for video in rgb_videos]
    rgb_video_names.sort(reverse=True)  # Sort in descending order

    if rgb_video_names:
        selected_rgb_video_name = st.selectbox('Select an RGB video file:', rgb_video_names, key='rgb_video_select')
        # Find the full path of the selected video
        selected_rgb_video_path = [video for video in rgb_videos if os.path.basename(video) == selected_rgb_video_name][0]
        
        timestamp_file_path = selected_rgb_video_path.replace('.mp4', '.txt')
    
        if os.path.exists(timestamp_file_path):
            with open(timestamp_file_path, 'r') as f:
                timestamps_rgb = [ts.strip() for ts in f.readlines()]
        
            if timestamps_rgb:
                selected_timestamp = st.select_slider('Select a timestamp to start from:', options=timestamps_rgb, key='timestamp_select_rgb')
                start_time_seconds = calculate_start_time(timestamps_rgb, selected_timestamp)
                # Extract the frame as a PIL image
                pil_image = extract_frame(selected_rgb_video_path, start_time_seconds)
                # Display the extracted frame
                st.image(pil_image)
                # Display selected RGB video
                if os.path.exists(selected_rgb_video_path):
                    rgb_video_file_size = convert_size(os.path.getsize(selected_rgb_video_path))
                    st.write(f"Selected RGB video file size: {rgb_video_file_size}")
                    video_file_rgb = open(selected_rgb_video_path, 'rb')
                    video_bytes_rgb = video_file_rgb.read()
                    # Find the directory of the selected RGB video
                    video_dir = os.path.dirname(selected_rgb_video_path)
                    # Search for files ending with *datetime.json in that directory
                    real_timestamps_files = selected_rgb_video_path.replace('.mp4', '_datetime.json')
                    
                    
            # # Proceed if there is at least one matching file
            # if real_timestamps_files:
                
            #     # Read and display the real timestamps
            #     with open(real_timestamps_files, 'r') as f:
            #         real_timestamps = json.load(f)

            #     col1, col2 = st.columns(2)

            #     with col1:
            #         st.write("Real Timestamps")
            #         for timestamp in real_timestamps:
            #             st.json(timestamp)

            #     with col2:
            #         # Load default activities from a text file
            #         activities_file_path = '/home/aiot/Desktop/Dongsheng/Octonet_Server/node_1/depthCamera/data/activity.txt'
            #         try:
            #             with open(activities_file_path, 'r') as file:
            #                 default_activities = [line.strip() for line in file.readlines()]
            #         except FileNotFoundError:
            #             st.error(f"File not found: {activities_file_path}")
            #             default_activities = []

            #         # Iterate through real timestamps and display them along with input fields for activities
            #         for index, timestamp in enumerate(real_timestamps):
            #             # Use the default activity from the file if available, else fallback to a generic name
            #             default_activity = default_activities[index] if index < len(default_activities) else f"Activity {index+1}"
            #             # Allow the user to modify the default activity name
            #             activity = st.text_input(f"Activity for {timestamp['start']} to {timestamp['end']}", value=default_activity, key=f"activity_{index}")
            #             timestamp['activity'] = activity  # Append activity to the timestamp

            #         # Optionally: Provide a button to save the updated real timestamps with activities
            #         if st.button("Save Activities"):
            #             # Assuming real_timestamps_path is defined and accessible in this scope
            #             with open(real_timestamps_files, 'w') as f:
            #                 json.dump(real_timestamps, f, indent=4)
            #             st.success("Activities saved successfully.")
            # else:
            #     st.write("No matching *datetime.json files found in the video directory.")
                else:
                    st.write("RGB video file not found.")
        else:
            st.write("Timestamp file not found.")

    else:
        st.write("No RGB video files found.")
    # RGB Cam Visualization
    # st.subheader("RGB Visualization")
    # base_rgb_dir = f'{node_dir}/depthCamera/data'  # Adjusted to new base directory for Depth Cam

    # def list_rgbpng_files(directory):
    #     """List .png files in the directory."""
    #     return [f for f in os.listdir(directory) if f.endswith('.png') and 'rgb' in f]


    # # List folders that potentially contain depth data
    # folders = list_folders(base_rgb_dir)
    # folders.sort(reverse=True)

    # if folders:
    #     selected_folder = st.selectbox('Select a folder with RGB data:', folders, key='rgb_folder_select')
        
    #     # List all PNG files in the selected folder
    #     rgb_files = list_rgbpng_files(selected_folder)
    #     rgb_files.sort(reverse=False)
        
    #     if rgb_files:
    #         selected_file = st.selectbox('Select a RGB data file:', rgb_files, key='rgb_file_select')
    #         selected_file_path = os.path.join(selected_folder, selected_file)
            
    #         # Display the selected image
    #         st.write(f"Selected RGB Image: {selected_file}")
    #         display_image(selected_file_path)
    #     else:
    #         st.write("No RGB images found in the selected folder.")
    # else:
    #     st.write("No folders found.")
    
    # seekthermal Cam Visualization
    st.subheader("SeekThermal Visualization")
    base_thermal_dir = f'{node_dir}/seekThermal/data'  # Adjusted to new base directory for Depth Cam

    def list_thermalpng_files(directory):
        """List .png files in the directory."""
        return [f for f in os.listdir(directory) if f.endswith('.png') and 'thermal' in f]


    # List folders that potentially contain depth data
    folders = list_folders(base_thermal_dir)
    folders.sort(reverse=True)

    if folders:
        selected_folder = st.selectbox('Select a folder with Thermal data:', folders, key='thermal_folder_select')
        
        # List all PNG files in the selected folder
        thermal_files = list_thermalpng_files(selected_folder)
        thermal_files.sort(reverse=False)
        
        if thermal_files:
            selected_file = st.selectbox('Select a Thermal data file:', thermal_files, key='thermal_file_select')
            selected_file_path = os.path.join(selected_folder, selected_file)
            
            # Display the selected image
            st.write(f"Selected Thermal Image: {selected_file}")
            display_image(selected_file_path)
            recovered_temperature = png_to_temperature(selected_file_path, 15, 50)
            st.write(recovered_temperature)
        else:
            st.write("No Thermal images found in the selected folder.")
    else:
        st.write("No folders found.")
    # # SeekThermal Visualization
    # st.subheader("SeekThermal Visualization")
    # base_seekthermal_dir = f'{node_dir}/seekThermal/data'  # Base directory for SeekThermal videos

    # # List the available thermal video files (e.g., thermal_0.mp4, thermal_1.mp4, etc.)
    # thermal_files = glob.glob(os.path.join(base_seekthermal_dir, '*.mp4'))
    # thermal_file_names = [os.path.basename(file) for file in thermal_files]
    # thermal_file_names.sort(key=lambda x: os.path.getmtime(os.path.join(base_seekthermal_dir, x)), reverse=True)  # Sort the files in descending order

    # if thermal_file_names:
    #     selected_thermal_file = st.selectbox('Select a thermal video file:', thermal_file_names, key='thermal_file_select')

    #     # Construct the path to the selected thermal video file
    #     thermal_video_path = os.path.join(base_seekthermal_dir, selected_thermal_file)
    #     timestamp_file_path = thermal_video_path.replace('.mp4', '.txt')
    
    #     if os.path.exists(timestamp_file_path):
    #         with open(timestamp_file_path, 'r') as f:
    #             timestamps_thermal = [ts.strip() for ts in f.readlines()]
        
    #         if timestamps_thermal:
    #             selected_timestamp = st.select_slider('Select a timestamp to start from:', options=timestamps_thermal, key='timestamp_select_thermal')
    #             start_time_seconds = calculate_start_time(timestamps_thermal, selected_timestamp)
    #             # Use FFmpeg to extract the frame
    #             frame_path = extract_frame(thermal_video_path, start_time_seconds)
    #             # Load the image directly from the path
    #             gray_image = Image.open(frame_path).convert('L')
    #             depth_array = np.array(gray_image, dtype=np.float32)  # Ensure float32 for accurate calculations
    #             min_temp = 10
    #             max_temp = 39
    #             temp_range = max_temp - min_temp
    #             temp_approx = (depth_array / 255) * temp_range + min_temp
    #             st.image(gray_image)  # Display grayscale image
    #             st.write(temp_approx)  # Display temperature data
    #             st.write("Image shape:", temp_approx.shape)
    #             # print("Normalized temperature data:", temp_normalized)
    #             # print("Saved grayscale image statistics:", np.min(temp_normalized), np.max(temp_normalized))
    #             st.write("Recovered temperature range:", np.min(temp_approx), np.max(temp_approx))
    #             # Check if the video file exists and display it
    #             if os.path.exists(thermal_video_path):
    #                 # Display file size for SeekThermal video file
    #                 seekthermal_file_size = convert_size(os.path.getsize(thermal_video_path))
    #                 st.write(f"Selected SeekThermal video file size: {seekthermal_file_size}")
                    
                    
    #             else:
    #                 st.write("Thermal video file not found.")
    #     else:
    #         st.write("Timestamp file not found.")
                    
    # else:
    #     st.write("No thermal video files found in the base directory.")

    # UWB Visualization# 
    # Function to plot UWB data
    def plot_uwb_data(frame_data, timestamp):
        frame = np.array(frame_data["frame"])  # adjust this line based on your data structure
        plt.figure()
        plt.ylim(-0.03, 0.03)
        plt.plot(frame)
        plt.title(f"Data for Timestamp: {timestamp}")
        st.pyplot(plt)
        st.write(frame_data, frame.shape)
    st.subheader("UWB Visualization")
    uwb_output_dir = f'{node_dir}/uwb/data'  # Update this to your UWB directory path
    uwb_files = glob.glob(os.path.join(uwb_output_dir, '*.pickle'))
    if uwb_files:
        uwb_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        uwb_file_path = st.selectbox('Select a UWB file:', uwb_files, key='uwb_select')
        uwb_data = load_data_from_pickle_uwb(uwb_file_path)
        st.write(f"Loaded {len(uwb_data)} frames for UWB.")
        uwb_index = st.slider("Select Frame Index for UWB", 0, len(uwb_data) - 1, 0, key='uwb_slider')
        selected_timestamp = list(uwb_data.keys())[uwb_index]
        plot_uwb_data(uwb_data[selected_timestamp], selected_timestamp)
    else:
        st.write("No UWB files found.")

    # Heart Trace Visualization
    def load_heart_trace_data(file_path):
        data = {}
        with open(file_path, 'rb') as file:
            while True:
                try:
                    frame_data = pickle.load(file)
                    timestamp = frame_data["timestamp"]
                    data[timestamp] = frame_data  # adjust based on your data structure
                except EOFError:
                    break
        return data


    # Function to plot heart trace data
    def plot_heart_trace(heart_trace_data):
        timestamps = [data["timestamp"] for data in heart_trace_data]
        values = [data["data"] for data in heart_trace_data]
        st.write(heart_trace_data)
        plt.figure()
        plt.plot(timestamps, values, marker='o')
        plt.xlabel('Time')
        plt.ylabel('Heart Rate')
        plt.title('Heart Trace Data')
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)


    # ToF Data Visualization
    st.header("ToF Visualization")
    tof_output_dir = f'{node_dir}/ToF/data'  # Update this with your ToF data directory path
    tof_files = glob.glob(os.path.join(tof_output_dir, '*.pickle'))
    if tof_files:
        tof_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        tof_file_path = st.selectbox('Select a ToF file:', tof_files, key='tof_select')
        tof_data = load_data_from_pickle(tof_file_path)
        st.write(f"Loaded {len(tof_data)} frames for ToF.")
        tof_index = st.slider("Select Frame Index for ToF", 0, len(tof_data) - 1, 0, key='tof_slider')
        display_frame_tof(tof_data, tof_index)
    else:
        st.write("No ToF files found.")


    # Streamlit UI for Heart Trace Visualization
    st.header("Heart Trace Visualization")

    # Select a heart trace data file
    heart_trace_dir = f'{node_dir}/polar/data'  # Update this to your directory path
    heart_trace_files = glob.glob(os.path.join(heart_trace_dir, '*.pickle'))
    if heart_trace_files:
        heart_trace_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        heart_trace_file = st.selectbox('Select a heart trace data file:', heart_trace_files)
        # heart_trace_data = load_heart_trace_data(heart_trace_file) 
        heart_trace_data = load_data_from_pickle(heart_trace_file) 
        st.write(f"Loaded data for Heart Trace.")
        plot_heart_trace(heart_trace_data)
    else:
        st.write("No Heart Trace files found.")

    # Render Acoustic Data Visualization
    render_acoustic_visualization(node_dir)




# Streamlit UI setup
st.title('Multi-Node Data Visualization Dashboard')

# Sidebar dashboard for node selection
node_dirs = ["node_1", "node_2", "node_3", "node_4", "node_5"]  # Update with your actual node directory names
st.sidebar.title("Node Selection")
selected_node = st.sidebar.radio("Choose a Node:", node_dirs)

# Render visualizations for the selected node
render_node_visualizations(selected_node)