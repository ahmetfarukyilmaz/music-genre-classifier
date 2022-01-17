import librosa
import numpy as np
import tensorflow as tf

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

file = "test/o-sen-olsan.wav"
y, sr = librosa.load(file)

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

tempo = librosa.beat.tempo(y)[0]

mfcc = librosa.feature.mfcc(y)
mfcc_means = np.array([np.mean(mfcc[i]) for i in range(len(mfcc))])
mfcc_vars = np.array([np.var(mfcc[i]) for i in range(len(mfcc))])

feature_vector = [chroma_stft_mean, chroma_stft_var, rms_var, spectral_bandwidth_mean,
                  harmonic_mean, harmonic_var, perceptr_mean, perceptr_var, tempo]

for i in range(0, 5):
    feature_vector.append(mfcc_means[i])
    feature_vector.append(mfcc_vars[i])

for i in range(5, 9):
    feature_vector.append(mfcc_means[i])

for i in [10, 17]:
    feature_vector.append(mfcc_means[i])

feature_vector = np.array(feature_vector)
feature_vector = feature_vector.reshape((1, 25))
frk_classifier = tf.keras.models.load_model('./frk-classifier')

# Check its architecture
frk_classifier.summary()

prediction = frk_classifier.predict(feature_vector)

# return position of max
MaxPosition = int(np.argmax(prediction))
prediction_label = mapping[MaxPosition]
print(f"File: {file}")
print(f"Label: {prediction_label}\n")

for i in range(10):
    print(f"{mapping[i]}: {prediction[0][i]}")

print()
