from os import listdir, system
from os.path import join

from scipy.io import wavfile


def get_all_files_in_dir(dir_path):
    return [join(dir_path, f) for f in listdir(dir_path)]


if __name__ == '__main__':
    normal_dir = "Dataset"
    noise_dir = "Dataset/data_with_noise_aug0.7"
    output_dir = "Dataset_noise"
    for dataset_type in ["training", "validation"]:
        for dir_count, dir_name in enumerate(get_all_files_in_dir("Dataset/" + dataset_type)):
            files = listdir(dir_name)
            this_output_dir = output_dir + "/" + dataset_type + "/" + str(dir_name.split("/")[-1])
            system("rm -r " + this_output_dir)
            system("mkdir -p " + this_output_dir)
            for file in files:
                read_file_name_normal = normal_dir + "/" + dataset_type + "/" + str(
                    dir_name.split("/")[-1]) + "/" + file

                output_file_name_normal = this_output_dir + "/" + str(file.split(".")[0]) + "_" + "normal.wav"
                fs, normal_audio = wavfile.read(read_file_name_normal)
                wavfile.write(output_file_name_normal, rate=fs, data=normal_audio)

                if dataset_type == "training":
                    read_file_name_noise = noise_dir + "/" + dataset_type + "/" + str(
                        dir_name.split("/")[-1]) + "/" + file
                    output_file_name_noise = this_output_dir + "/" + str(file.split(".")[0]) + "_" + "noise.wav"
                    fs, noise_audio = wavfile.read(read_file_name_noise)
                    wavfile.write(output_file_name_noise, rate=fs, data=noise_audio)

            print("\r dataset_type:{} dir:{}".format(dataset_type, dir_count + 1), end=" ")
        print()
