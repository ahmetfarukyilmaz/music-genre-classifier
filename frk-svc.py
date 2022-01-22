import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

data = pd.read_csv("features_3_sec.csv")

mapping = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9,
}

# get columns of data from 20 to 42
X = data.iloc[:, 19:45]

y = np.array([mapping[i] for i in data.iloc[:, -1]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


svm = SVC(decision_function_shape="ovo")
svm.fit(X_train, y_train)


mapping = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock",
}

y, sr = librosa.load("test/dre.wav")

chroma_stft = librosa.feature.chroma_stft(y)
chroma_stft_mean = np.mean(chroma_stft)
chroma_stft_var = np.var(chroma_stft)

rms = librosa.feature.rms(y)
rms_mean = np.mean(rms)
rms_var = np.var(rms)

spectral_centroid = librosa.feature.spectral_centroid(y)
spectral_centroid_mean = np.mean(spectral_centroid)
spectral_centroid_var = np.var(spectral_centroid)

spectral_bandwidth = librosa.feature.spectral_bandwidth(y)
spectral_bandwidth_mean = np.mean(spectral_bandwidth)
spectral_bandwidth_var = np.var(spectral_bandwidth)

spectral_rolloff = librosa.feature.spectral_rolloff(y)
spectral_rolloff_mean = np.mean(spectral_rolloff)
spectral_rolloff_var = np.var(spectral_rolloff)

zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
zero_crossing_rate_mean = np.mean(zero_crossing_rate)
zero_crossing_rate_var = np.var(zero_crossing_rate)

harmonic, perceptr = librosa.effects.hpss(y)
perceptr_mean = np.mean(perceptr)
perceptr_var = np.var(perceptr)
harmonic_mean = np.mean(harmonic)
harmonic_var = np.var(harmonic)

tempo = librosa.beat.tempo(y)

mfcc = librosa.feature.mfcc(y)
mfcc_means = np.array([np.mean(mfcc[i]) for i in range(len(mfcc))])
mfcc_vars = np.array([np.var(mfcc[i]) for i in range(len(mfcc))])

feature_vector = []

for i in range(13):
    feature_vector.append(mfcc_means[i])
    feature_vector.append(mfcc_vars[i])

feature_vector = np.array(feature_vector)
feature_vector = feature_vector.reshape((1, 26))

y_pred = svm.predict(X_test)
y_pred_train = svm.predict(X_train)

print(f"Training accuracy: {accuracy_score(y_train, y_pred_train)}")
print(f"Validation accuracy: {accuracy_score(y_test, y_pred)}")



y_pred = svm.predict(feature_vector)
print(y_pred)
