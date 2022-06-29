from os import path, makedirs
import random
import numpy as np
import tensorflow as tf


def fix_random(seed: int):
    """Fix all the possible sources of randomness.
    Args:
        seed: the seed to use.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_path(filepath: str) -> None:
    """
    Creates a path to a file, if it does not exist.

    :param filepath: the filepath.
    """
    # Get the file's directory.
    directory = path.dirname(filepath)

    # Create directory if it does not exist
    if not path.exists(directory) and not directory == '':
        makedirs(directory)


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Converts an rgb image array to a grey image array.

    :param rgb: the rgb image array.
    :return: the converted array.
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)