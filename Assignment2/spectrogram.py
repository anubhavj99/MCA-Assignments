# Name: Anubhav Jaiswal
# Roll Number: 2016014

from os import listdir, system
from os.path import join

import numpy as np
from scipy.io import wavfile


def get_all_files_in_dir(dir_path):
    return [join(dir_path, f) for f in listdir(dir_path)]


def get_fft_part(val, n):
    size = len(val)
    ks = np.arange(0, size, 1)
    xn = np.sum(val * np.exp((1j * 2 * np.pi * ks * n) / size)) / size
    return xn


def get_fft(signal_seq):
    mag = []
    length = len(signal_seq)
    for n in range(int(length / 2)):
        mag.append(np.abs(get_fft_part(signal_seq, n)) * 2)
    return mag


def create_spectrogram(ts, window_size, n_overlap, log_normalize=True, debug=False):
    step = window_size - n_overlap
    block_indices = np.arange(0, len(ts), step, dtype=int)
    # remove any window with less than NFFT sample size
    block_indices = block_indices[block_indices + window_size <= len(ts)]
    hanning_weights = np.hanning(len(block_indices))
    xns = []
    for i, block_index in enumerate(block_indices):
        # short term discrete fourier transform
        ts_block_for_fft = ts[block_index:block_index + window_size] * hanning_weights[i]
        ts_window = get_fft(ts_block_for_fft)
        # print(ts_window.shape)
        xns.append(ts_window)

    spectrogram = np.array(xns).T
    avg_spec = np.average(spectrogram)
    spectrogram = np.where(spectrogram <= 0, avg_spec, spectrogram)

    if log_normalize:
        spectrogram = 20 * np.log10(spectrogram)

    return block_indices, spectrogram


def clip_sound_data(sample_rate, data):
    final_data = data.astype(dtype=np.float64)
    if len(data) < 16000:
        final_data = np.append(final_data, np.zeros((16000 - len(data),)))

    final_data = final_data + np.finfo(dtype=final_data.dtype).eps
    # np.finfo(dtype=x.dtype).eps
    return sample_rate, final_data


def start(file_name, window_size, n_overlap, log_normalize=True, debug=False):
    sample_rate, ts = wavfile.read(file_name)
    sample_rate, ts = clip_sound_data(sample_rate, ts)
    starts, spec = create_spectrogram(ts, window_size, n_overlap=n_overlap, log_normalize=log_normalize, debug=True)

    return spec


if __name__ == '__main__':
    import pickle
    from datetime import datetime
    window = 256
    n_overlap = 84
#     experiments = [
#         (256, 84),
#         (160, 80),
#         # (16000 // 50, 16000 // 150),
#         # (16000 // 100, 16000 // 200),
#         # (16000 // 100, 16000 // 300),
#         # (16000 // 80, 16000 // 160),
#         # (16000 // 80, 16000 // 240),
#     ]
#     dir_value = {
#         "eight": 8,
#         "five": 5,
#         "four": 4,
#         "nine": 9,
#         "one": 1,
#         "seven": 7,
#         "six": 6,
#         "three": 3,
#         "two": 2,
#         "zero": 0,
#     }
#     dataset_read_dir = "Dataset_noise"
#     save_directory = "spectrogram_noise_dataset/spectrogram_aug0.7_noise50"
#     system("mkdir -p " + save_directory)
#     start_time = datetime.now()
#     for dataset_type in ["training", "validation"]:
# #         print("dataset_type", dataset_type)
#         for window, n_overlap in experiments:
#             data = {}
#             print("type:{} window:{} overlap:{}".format(dataset_type, window, n_overlap))
#             for dir_count, dir_name in enumerate(get_all_files_in_dir( dataset_read_dir + "/" + dataset_type)):
#                 start_time_this = datetime.now()
#                 for i, file in enumerate(get_all_files_in_dir(dir_name)):
#                     value = start(file, window, n_overlap, log_normalize=True)
#                     data[file] = dir_value[dir_name.split("/")[2]], value
#                 print("\rtime_elapsed:{} dir:{}, time_taken:{} ".format(datetime.now() - start_time,
#                                                                         dir_count + 1,
#                                                                         datetime.now() - start_time_this),
#                       end=" ")
#
#             save_file_name = "{}/{}_L{}_no{}.pkl".format(save_directory, dataset_type, window, n_overlap)
#             pickle.dump(data, open(save_file_name, "wb"))
#             print("\nsaved at {}\n".format(save_file_name))

    spec_x = start("Dataset/training/zero/0b56bcfe_nohash_0.wav", window, n_overlap, debug=True)
    print("shape", spec_x.shape)

# 0b56bcfe_nohash_0.wav
# 0b56bcfe_nohash_1.wav
# 0b56bcfe_nohash_2.wav
# 0b09edd3_nohash_0.wav
# 0bd689d7_nohash_0.wav
# 1a892463_nohash_1.wav