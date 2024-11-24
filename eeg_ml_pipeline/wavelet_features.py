import pywt
import numpy as np


def get_wavelet_features(eeg_signal):
    from .config import SAMPLING_FREQUENCY, WAVELET_FREQUENCIES
    sampling_rate = SAMPLING_FREQUENCY
    wavelet_features = []
    for channel in eeg_signal:
        frequencies = WAVELET_FREQUENCIES
        scales = [sampling_rate / freq for freq in frequencies]
        coefficients, _ = pywt.cwt(channel, scales, 'cmor')

        # print(len(coefficients))

        mean_coeffs = np.mean(np.abs(coefficients), axis=1)
        std_coeffs = np.std(coefficients, axis=1)
        
        channel_wavelet_features = np.concatenate((mean_coeffs, std_coeffs), axis = 0)
        wavelet_features.append(channel_wavelet_features)

    return wavelet_features
