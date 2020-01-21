import librosa
import numpy as np
from os import listdir


def duplicate_data():
    for dir_path in ['../data/Train/dogs/', '../data/Train/cats/', '../data/Test/dogs/', '../data/Train/cats/']:
        files = listdir(dir_path)
        for filename in files:
            data, sampling_rate = librosa.load('{}{}'.format(dir_path, filename))
            librosa.output.write_wav('../data/Duplicated/{}a.wav'.format(filename), np.asfortranarray(data[::2]),
                                     sampling_rate)
            librosa.output.write_wav('../data/Duplicated/{}b.wav'.format(filename), np.asfortranarray(2 * data),
                                     sampling_rate)
            librosa.output.write_wav('../data/Duplicated/{}c.wav'.format(filename),
                                     data + np.random.uniform(-1e-2, 1e-2, data.size), sampling_rate)
            librosa.output.write_wav('../data/Duplicated/{}d.wav'.format(filename), np.repeat(data, 2), sampling_rate)
