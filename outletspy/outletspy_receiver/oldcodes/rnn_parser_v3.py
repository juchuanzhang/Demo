import os
import sys
import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile
from scipy.fftpack import dct
from statsmodels.tools import categorical

nb_classes = 16

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(path):

    features = np.array([], dtype='float32')

    files = [file for file in os.listdir(path)]
    label = ["" for x in range(len(files))]
    np.random.shuffle(files)

    i=0
    count=0
    print(os.path.basename(path))
    for file_name in files:

        unit = os.path.join(path,(str)(file_name))
        df = pd.read_csv(unit, header=None, index_col=False)
        #df.loc[:,0] = df.loc[:,0].astype('float32')
        sample = df.iloc[:,0].values
        #sample = df.iloc[0].values
        name = file_name[:file_name.index('_')]
        label[i] = name
        i=i+1
        print((str)(i)+" files done.")
        sample = sample / max(abs(sample))
        mfccs = np.zeros([32, 4])
        for j in range(0, 32):
            temp = sample[j * 24 : (j + 1) * 24]
            temp_dct = dct(temp)
            mfccs[j, :] = temp_dct[0:4]

        mfccs = mfccs[np.newaxis, :, :]
        if count==0:
            features = mfccs
            count = 1
        else:
            features = np.vstack((features, mfccs)) # ( number of files, frames=32, 26 )

    return np.array(features), np.asarray(label)

def one_hot_encode(labels):
    labels = categorical(labels, drop=True)
    return labels


#tr_features,tr_labels = extract_features(path="pfc_train_1/")
#tr_labels = one_hot_encode(tr_labels)

#np.save('train_features_rnn', tr_features)
#print('rnn features saved: ',tr_features.shape)
#np.save('train_labels_rnn', tr_labels)
#print('labels saved: ',tr_labels.shape)



dirname = "split_data_40_new"
if len(sys.argv) > 1:
    dirname = sys.argv[1]

tr_features, tr_labels = extract_features(path=""+ dirname + "/")
tr_labels = one_hot_encode(tr_labels)

np.save("features/" + dirname + "_features", tr_features)
print('rnn features saved: ',tr_features.shape)
np.save("features/" + dirname + "_labels", tr_labels)
print('labels saved: ',tr_labels.shape)
