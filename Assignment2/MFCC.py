# Name: Anubhav Jaiswal
# Roll Number: 2016014

from os import listdir, system
from os.path import join
import numpy as np
from scipy import fftpack
from scipy.io import wavfile

from spectrogram import create_spectrogram, clip_sound_data

dir_value = {
    "eight": 8,
    "five": 5,
    "four": 4,
    "nine": 9,
    "one": 1,
    "seven": 7,
    "six": 6,
    "three": 3,
    "two": 2,
    "zero": 0,
}


def get_all_files_in_dir(dir_path):
    return [join(dir_path, f) for f in listdir(dir_path)]


def ftm(f):
    return 2595.0 * np.log(1.0 + f / 700.0)


def mtf(m):
    return 700.0 * (np.exp(m / 2595.0) - 1.0)


def iter_bin(out, curr_bin, next_bins, sign):
    next_bin = next_bins[np.where(next_bins > curr_bin)][0]
    if sign == -1:
        bias = next_bin
    else:
        bias = curr_bin

    for f in range(int(curr_bin), int(next_bin)):
        out[f] = sign * (f - bias) / (next_bin - curr_bin)


def create_mel_filterbanks(num_bank, num_freq, sample_freq):
    num_fft = (num_freq - 1) * 2
    low_mel = ftm(0)
    high_mel = ftm(sample_freq // 2)
    bank_set = np.linspace(low_mel, high_mel, num_bank + 2)
    bin_set = np.floor((num_fft + 1) * mtf(bank_set) / sample_freq)
    mel_filter_bank_set = np.zeros((num_bank, num_fft // 2 + 1))
    for bank_index in range(num_bank):
        iter_bin(mel_filter_bank_set[bank_index], bin_set[bank_index], bin_set[bank_index + 1:], -1)
        iter_bin(mel_filter_bank_set[bank_index], bin_set[bank_index + 1], bin_set[bank_index + 2:], 1)
    return mel_filter_bank_set


def create_mfcc(sample_freq, signal_seq, window_size, n_overlap, mel_bank_bins, dct_ceps_threshold,
                dct_filter_size, debug=False):
    pre_coeff = 0.95
    signal_seq = np.append(signal_seq[0], signal_seq[1:] - pre_coeff * signal_seq[:-1])
    indices, spectrogram = create_spectrogram(signal_seq, window_size, n_overlap, log_normalize=False)
    if debug:
        print("spectrogram.shape", spectrogram.shape)
    filter_banks = create_mel_filterbanks(mel_bank_bins, spectrogram.shape[0], sample_freq)
    filter_bank_spectrogram = np.dot(filter_banks, spectrogram)
    filter_bank_spectrogram[np.where(filter_bank_spectrogram == 0)] = np.finfo(dtype=filter_bank_spectrogram.dtype).eps
    filter_bank_spectrogram_normalized = 10 * np.log(filter_bank_spectrogram)

    DCT_features = fftpack.dct(filter_bank_spectrogram_normalized, axis=0)
    DCT_features = DCT_features[: dct_ceps_threshold]
    mel_filter = 1 + dct_filter_size / 2.0 * np.sin(np.pi * np.arange(dct_ceps_threshold) / dct_filter_size)
    mfcc_features = mel_filter[:, np.newaxis] * DCT_features

    return mfcc_features


def start(file_name, window_size, n_overlap, mel_bank_bins=23, dct_ceps_threshold=13, dct_filter_size=22, debug=False):
    fs, data = wavfile.read(file_name)
    fs, data = clip_sound_data(fs, data)
    data_signal = data.astype(np.float64)
    mfcc_features = create_mfcc(fs, data_signal, window_size, n_overlap, mel_bank_bins=mel_bank_bins,
                                dct_ceps_threshold=dct_ceps_threshold, dct_filter_size=dct_filter_size, debug=debug)
    return mfcc_features


if __name__ == '__main__':
    import pickle
    from datetime import datetime

    window = 16000 // 100
    n_overlap = window // 2

    max_freq_reduction = 0
    min_freq_threshold = 0
    mel_bank_bins = 80
    dct_ceps_threshold = 13
    dct_filter_size = 22
    # experiments = [
    #     (256, 84),
    #     # (16000 // 50, 16000 // 150),
    #     (160, 80),
    #     # (16000 // 100, 16000 // 300),
    #     # (16000 // 80, 16000 // 160),
    #     # (16000 // 80, 16000 // 240),
    # ]
    # dataset_read_dir = "Dataset_noise"
    # save_directory = "mfcc_noise_dataset/mfcc_aug0.7_noise50_bins80_ceps13"
    # system("mkdir -p " + save_directory)
    # start_time = datetime.now()
    # for dataset_type in ["training", "validation"]:
    #     for window, n_overlap in experiments:
    #         data = {}
    #         print("type:{} window:{} overlap:{}".format(dataset_type, window, n_overlap))
    #         for dir_count, dir_name in enumerate(get_all_files_in_dir(dataset_read_dir + "/" + dataset_type)):
    #             start_time_this = datetime.now()
    #             for i, file in enumerate(get_all_files_in_dir(dir_name)):
    #                 value = start(file, window, n_overlap, num_mel_bins=mel_bank_bins, num_ceps=dct_ceps_threshold,
    #                               num_filter=dct_filter_size)
    #                 data[file] = dir_value[dir_name.split("/")[2]], value
    #             print("\rtime_elapsed:{} dir:{}, time_taken:{} ".format(datetime.now() - start_time,
    #                                                                     dir_count + 1,
    #                                                                     datetime.now() - start_time_this),
    #                   end=" ")
    #
    #         save_file_name = "{}/{}_L{}_no{}.pkl".format(save_directory, dataset_type, window, n_overlap)
    #         pickle.dump(data, open(save_file_name, "wb"))
    #         print("\nsaved at {}\n".format(save_file_name))

    mel = start("Dataset/training/zero/0b09edd3_nohash_0.wav", window, n_overlap, mel_bank_bins=mel_bank_bins,
                dct_ceps_threshold=dct_ceps_threshold, dct_filter_size=dct_filter_size, debug=True)
    print("mel.shape", mel.shape)

# 0b56bcfe_nohash_0.wav
# 0b56bcfe_nohash_1.wav
# 0b56bcfe_nohash_2.wav
# 0b09edd3_nohash_0.wav
# 0bd689d7_nohash_0.wav
# 1a892463_nohash_1.wav
