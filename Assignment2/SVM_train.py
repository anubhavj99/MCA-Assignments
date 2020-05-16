
import pickle
from datetime import datetime

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report

output_file = "svm_train_output.txt"


def get_pickle_file(load_directory, dataset_type, window, n_overlap):
    file_name = "{}/{}_L{}_no{}.pkl".format(load_directory, dataset_type, window, n_overlap)
    print("loading {}...".format(file_name))
    return pickle.load(open(file_name, "rb"))


def print_and_to_file(value):
    print(value)
    f = open(output_file, "a+")
    f.write(str(value) + "\n")
    f.close()


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def process_dataset(data):
    X = []
    y = []
    for file_name in data:
        row = data[file_name]
        X.append(row[1].flatten())
        y.append(row[0])

    X = np.array(X)
    y = np.array(y)

    # y = get_one_hot(y, np.max(y)+1)
    z = []
    for a in X:
        z.append(a.shape)
    print(list(set(z)))
    print(y.shape, X.shape)

    return X, y


def normalize_dataset(X_train, X_val):
    max_val = max(abs(np.max(X_train)), abs(np.min(X_train)), abs(np.max(X_val)), abs(np.min(X_val)))
    print("max_val", max_val)
    X_train, X_val = X_train / max_val, X_val / max_val
    print("checking", np.max(X_train), np.min(X_train), np.max(X_val), np.min(X_val))
    return X_train, X_val


def train(feature_type, directory_spec, window, overlap, normalize="", load_clf=False):
    load_directory = "{}/{}".format(feature_type, directory_spec)
    X_train, y_train = process_dataset(get_pickle_file(load_directory, "training", window, overlap))
    X_val, y_val = process_dataset(get_pickle_file(load_directory, "validation", window, overlap))
    if len(normalize) > 0:
        X_train, X_val = normalize_dataset(X_train, X_val)
        
    if not load_clf:
        clf = svm.SVC(kernel="linear", verbose=True)
        print("started training {}".format(normalize))
        start_time = datetime.now()
        clf.fit(X_train, y_train)
        print("fit in", datetime.now() - start_time)
        pickle.dump(clf, open("{}/clf{}_L{}_no{}.pkl".format(load_directory, normalize, window, overlap), "wb"))
    else:
        clf = pickle.load(open("{}/clf{}_L{}_no{}.pkl".format(load_directory, normalize, window, overlap), "rb"))

    print_and_to_file(
        "\n\n\nshowing results norm:{} feature:{} specs:{} window:{} overlap:{}".format(normalize, feature_type, load_directory, window,
                                                                                overlap))
    y_pred = clf.predict(X_val)
    print_and_to_file(classification_report(y_val, y_pred))

#     y_pred = clf.predict(X_train)
#     print_and_to_file(classification_report(y_train, y_pred))


if __name__ == '__main__':
    experiment_set = [
        {
            "feature_type": "mfcc_noise_dataset",
            "directory_spec": "mfcc_aug0.7_noise50_bins80_ceps13",
            "window": 160,
            "overlap": 80,
        },
        {
            "feature_type": "mfcc_noise_dataset",
            "directory_spec": "mfcc_aug0.7_noise50_bins80_ceps13",
            "window": 256,
            "overlap": 84,
        },
        {
            "feature_type": "spectrogram_noise_dataset",
            "directory_spec": "spectrogram_aug0.7_noise50",
            "window": 256,
            "overlap": 84,
        },
        {
            "feature_type": "spectrogram_noise_dataset",
            "directory_spec": "spectrogram_aug0.7_noise50",
            "window": 160,
            "overlap": 80,
        },
    ]

    for experiment in experiment_set:
        feature_type = experiment["feature_type"]
        directory_spec = experiment["directory_spec"]
        window = experiment["window"]
        overlap = experiment["overlap"]
        train(feature_type, directory_spec, window, overlap, normalize="norm_", load_clf=False)
