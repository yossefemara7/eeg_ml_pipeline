import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from .config import *
from scipy.signal import spectrogram, gaussian
from .config import *

def compute_gaf(eeg_signal, eeg_min, eeg_max, scale = 1):
    #Get the min and max using the min, max of the eeg 
    # eeg_min, eeg_max = eeg_signal.min(), eeg_signal.max()
    normalized_signal = 2 * (eeg_signal - eeg_min) / (eeg_max - eeg_min) - 1
    normalized_signal = np.nan_to_num(normalized_signal, nan = 0)
    gaf_transformer = GramianAngularField(image_size=int(len(normalized_signal)//scale), method='difference')
    gaf = gaf_transformer.fit_transform(normalized_signal.reshape(1, -1))[0]
    return gaf

def plot_gaf(gaf):
    plt.figure(figsize=(6, 6))
    plt.imshow(gaf, cmap='rainbow', origin='upper')
    plt.colorbar()
    plt.title("Gramian Angular Field (GAF) of EEG Signal")
    plt.xlabel("Time Index")
    plt.ylabel("Time Index")
    plt.show()

def compute_recurrence_plot(eeg_signal):
    pass

def compute_spectrogram(eeg_signal, fs):

    window_length = SPECTOGRAM_WINDOW_LENGTH
    std_dev = SPECTOGRAM_STD_DEV 
    hop_size = SPECTOGRAM_HOP_SIZE
    mfft = SPECTOGRAM_MFFT

    window = gaussian(window_length, std=std_dev, sym=True)

    f, t, Sxx = spectrogram(eeg_signal, fs, window=window, noverlap=window_length-hop_size, nfft=mfft)

    return Sxx[:int(len(Sxx)/4)], f, t

def plot_spectrogram(Sxx, f, t):
    plt.figure(figsize=(10, 6))
    plt.imshow(Sxx, origin='lower', aspect='auto', cmap='viridis', extent=[t.min(), t.max(), f.min(), SPECTROGRAM_MAX_FREQUENCY])
    plt.colorbar(label="Frequency")
    plt.title("EEG Signal Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()
