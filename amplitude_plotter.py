import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# blues     00040 Stevie Ray Vaughan - Love Struck Baby
# classical 00045 Richard Strauss - Fuge Für Klavier
# country   00055 Vince Gill - Take Your Memory With You
# disco     00033 The Tymes - You Little Trustmaker
# hiphop    00008 Beastie Boys - Slow and Low
# jazz      00081 Tony Williams - Geo Rose
# metal     00089 Danzig - Apokalips
# pop       00039 Britney Spears - Deep In My Heart
# reggae    00072 Freddie McGregor - Prophecy
# rock      00033 The Rolling Stones - Gimme Shelter


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def plot_wave_form(path, title):
    y, sr = librosa.load(path)

    plt.figure(figsize=(10, 6))

    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig(f"figures/wave/{title}")
    plt.close()

    fft = np.fft.fft(y)
    magnitudes = np.abs(fft)
    frequencies = np.linspace(0, sr, len(magnitudes))

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, magnitudes)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

    plt.savefig(f"figures/fft/{title}")
    plt.close()

    stft = librosa.stft(y)
    stft_in_dB = librosa.amplitude_to_db(abs(stft))

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(stft_in_dB, sr=sr, y_axis="linear")
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.savefig(f"figures/spectogram/{title}")
    plt.close()

    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfcc, sr=sr, x_axis="time", y_axis="linear")
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Coefficients")

    plt.savefig(f"figures/mfcc/{title}")
    plt.close()


wav_dir = "./Data/genres_original"


songs = [
    (
        f"{wav_dir}/blues/blues.00040.wav",
        "Blues (Stevie Ray Vaughan - Love Struck Baby)",
    ),
    (
        f"{wav_dir}/classical/classical.00045.wav",
        "Classical (Richard Strauss - Fuge Für Klavier)",
    ),
    (
        f"{wav_dir}/country/country.00055.wav",
        "Country (Vince Gill - Take Your Memory With You)",
    ),
    (f"{wav_dir}/disco/disco.00033.wav", "Disco (The Tymes - You Little Trustmaker)"),
    (f"{wav_dir}/hiphop/hiphop.00008.wav", "Hip hop (Beastie Boys - Slow and Low)"),
    (f"{wav_dir}/jazz/jazz.00081.wav", "Jazz (Tony Williams - Geo Rose)"),
    (f"{wav_dir}/metal/metal.00089.wav", "Metal (Danzig - Apokalips)"),
    (f"{wav_dir}/pop/pop.00039.wav", "Pop (Britney Spears - Deep In My Heart)"),
    (f"{wav_dir}/reggae/reggae.00072.wav", "Reggae (Freddie McGregor - Prophecy)"),
    (f"{wav_dir}/rock/rock.00033.wav", "Rock (The Rolling Stones - Gimme Shelter)"),
]

for song in songs:
    plot_wave_form(song[0], song[1])
