import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
import sys


def audio2array(file_name):

    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T,axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    feature = mfccs
    label = row.Class

    return [feature, label]

    temp = train.apply(audio2array, axis=1)
    temp.columns = ['feature', 'label']

    X = np.array(temp.feature.tolist())
    Y = np.array(temp.label.tolist())

    lb = LabelEncoder()

    Y = np_utils.to_categorical(lb.fit_transform(y))

    num_labels = y.shape[1]
    filter_size = 2


def train():

    # function to load files and extract features
    file_name = os.path.join(os.path.abspath(data_dir), 'Train', str(row.ID) + '.wav')

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

    # load model
    model = load_model('model.h5')

    # load test file
    test_file = sys.argv[1]
    [X,Y] = audio2array(test_file)

    # evaluate loaded model on test data
    score = model.evaluate(X, Y, verbose=0)
    print(f"{model.metrics_names[1]}: {score[1]:.2}")
