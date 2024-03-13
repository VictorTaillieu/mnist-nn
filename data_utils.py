import gzip

import idx2numpy


def load_data(dataset_name):
    """Load MNIST data"""
    images_path = f"data/{dataset_name}-images-idx3-ubyte.gz"
    labels_path = f"data/{dataset_name}-labels-idx1-ubyte.gz"

    with gzip.open(images_path, "rb") as images_file:
        images = idx2numpy.convert_from_file(images_file)

    with gzip.open(labels_path, "rb") as labels_file:
        labels = idx2numpy.convert_from_file(labels_file)

    return images, labels


def load_train_data():
    """Load MNIST train data"""
    return load_data("train")


def load_test_data():
    """Load MNIST test data"""
    return load_data("t10k")
