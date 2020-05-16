
from os import listdir, system
from os.path import join
from random import choice

from scipy.io import wavfile
import numpy as np

from spectrogram import clip_sound_data


def get_all_files_in_dir(dir_path):
    return [join(dir_path, f) for f in listdir(dir_path)]


def repeat_sound_data(noise_signal):
    if len(noise_signal) >= 16000:
        return noise_signal[:16000]
    final_data = noise_signal.copy()
    repeat_count = 16000 // len(noise_signal)
    print("len(noise_signal)", len(noise_signal))
    for i in range(repeat_count):
        final_data = np.append(final_data, noise_signal.copy())
    residual_length = 16000 % len(noise_signal)
    final_data = np.append(final_data, noise_signal.copy()[:residual_length])
    return final_data


def get_noise_file(file_name):
    sample_rate, ts = wavfile.read(file_name)
    return repeat_sound_data(ts)


def augment(data_signal, sample_rate, augment_ratio):
    noise_name = choice(["doing_the_dishes.wav", "dude_miaowing.wav", "exercise_bike.wav", "pink_noise.wav", "running_tap.wav", "white_noise.wav"])
    noise_signal = get_noise_file("Dataset/_background_noise_/" + noise_name)
#     noise_signal = noise_signal / np.max(noise_signal)
    final_audio = augment_ratio*data_signal + (1.0-augment_ratio)*noise_signal
    return final_audio
#     Audio(data_signal, rate=16000)
#     Audio(final_audio, rate=16000)
#     wavfile.write("noise_signal.wav", sample_rate, noise_signal)
#     wavfile.write("data_signal.wav", sample_rate, data_signal)
#     wavfile.write("final_audio.wav", sample_rate, final_audio)

def start(file_name, augment_ratio):
    sample_rate, ts = wavfile.read(file_name)
    sample_rate, ts = clip_sound_data(sample_rate, ts)
    final_audio = augment(ts, sample_rate, augment_ratio)
    return sample_rate, final_audio

if __name__ == '__main__':
    from datetime import datetime
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
    augment_ratio = 0.7
    save_directory = "Dataset/data_with_noise_aug{}/".format(augment_ratio)
    system("mkdir -p " + save_directory)
    start_time = datetime.now()
    for dataset_type in ["training", "validation"]:
        for dir_count, dir_name in enumerate(get_all_files_in_dir("Dataset/" + dataset_type)):
            start_time_this = datetime.now()
            new_dir_name = save_directory + "/".join(dir_name.split("/")[1:])
            system("mkdir -p " + new_dir_name)
            for i, file in enumerate(get_all_files_in_dir(dir_name)):
                rate, audio = start(file, augment_ratio)
                new_file_name = save_directory + "/".join(file.split("/")[1:])
                wavfile.write(new_file_name, rate=rate, data=audio)
            print("\rtime_elapsed:{} dataset_type:{} dir:{} time_taken:{}".format(datetime.now() - start_time,
                                                                dataset_type, dir_count + 1,
                                                                datetime.now() - start_time_this),
              end=" ")
