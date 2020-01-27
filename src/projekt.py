import librosa
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
import sys
from playsound import playsound


DATA_FOLDER  = "../input/audio-cats-and-dogs/cats_dogs/train/cat/"


def audio2array(file_name):

    # here kaiser_fast is a technique used for faster extraction
    [data, sample_rate] = librosa.load(file_name, res_type='kaiser_fast') 

    # we extract mfcc feature from data
    feature = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40), axis=1).T #40 values
    if ('cat' in file_name):
        label = 0
    elif ('dog' in file_name):
        label = 1

    return [feature, label]


def train():

    # function to load files and extract features
    X = np.empty((0,40)); Y = np.empty(0)
    file_names = [join(DATA_FOLDER, file) for file in listdir(DATA_FOLDER) if isfile(join(DATA_FOLDER, file))]
    for file_name in file_names:
        # handle exception to check if there isn't a file which is corrupted
        try:
            [x, y] = audio2array(file_name)
        except Exception as exc:
            print(f"Error encountered while parsing file\n{exc}")
            continue
        X = np.vstack([X,x]); Y = np.vstack([Y,y])

    num_labels  = Y.shape[1]

    # build model
    model = Sequential()

    model.add(Dense(256, input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X, Y, batch_size=32, epochs=5, validation_data=(val_x, val_y))

    # save model and architecture to single file
    model.save("neural_net.h5")


if __name__ == '__main__':

    if (len(sys.argv) != 2):
        print("Provide test file name and only that!")
        sys.exit()
    else:

        # load model
        model = load_model('model.h5')

        # load test file
        file_name = sys.argv[1]
        [x, y] = audio2array(file_name)

        # evaluate loaded model on test data
        probabilities = model.predict(x, batch_size=32)
        label = np.argmax(probabilities)
        
        playsound(file_name)

        print(f"I think it is a {label}.\nIt is {y} in fact.")
