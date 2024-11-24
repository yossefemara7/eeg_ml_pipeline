import numpy as np
from scipy.signal import welch

def delta_band(data):
    from .config import SAMPLING_FREQUENCY
    fs = SAMPLING_FREQUENCY

    delta_band = (0.5, 4) 
    delta_powers = []
    
    for channel_data in data:
        freqs, psd = welch(channel_data, fs=fs, nperseg=fs*2)
        
        delta_indices = np.logical_and(freqs >= delta_band[0], freqs <= delta_band[1])
        
        delta_power = np.trapz(psd[delta_indices], freqs[delta_indices])
        delta_powers.append(delta_power)
    
    return np.array(delta_powers)

def theta_band(data):
    from .config import SAMPLING_FREQUENCY
    fs = SAMPLING_FREQUENCY

    theta_band = (4, 7) 
    theta_powers = []
    
    for channel_data in data:
        freqs, psd = welch(channel_data, fs=fs, nperseg=fs*2)
        
        theta_indices = np.logical_and(freqs >= theta_band[0], freqs <= theta_band[1])
        
        theta_power = np.trapz(psd[theta_indices], freqs[theta_indices])
        theta_powers.append(theta_power)
    
    return np.array(theta_powers)

def alpha_band(data):
    from .config import SAMPLING_FREQUENCY
    fs = SAMPLING_FREQUENCY

    alpha_band = (8, 13) 
    alpha_powers = []
    
    for channel_data in data:
        freqs, psd = welch(channel_data, fs=fs, nperseg=fs*2)
        
        alpha_indices = np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])
        
        alpha_power = np.trapz(psd[alpha_indices], freqs[alpha_indices])
        alpha_powers.append(alpha_power)
    
    return np.array(alpha_powers)

def beta_band(data):
    from .config import SAMPLING_FREQUENCY
    fs = SAMPLING_FREQUENCY

    beta_band = (14, 30) 
    beta_powers = []
    
    for channel_data in data:
        freqs, psd = welch(channel_data, fs=fs, nperseg=fs*2)
        
        beta_indices = np.logical_and(freqs >= beta_band[0], freqs <= beta_band[1])
        
        beta_power = np.trapz(psd[beta_indices], freqs[beta_indices])
        beta_powers.append(beta_power)
    
    return np.array(beta_powers)

def beta_band_relative(data):
    from .config import SAMPLING_FREQUENCY
    fs = SAMPLING_FREQUENCY

    beta_band = (13, 30)     # Beta frequency range
    relevant_band = (4, 50)  # Total power range, excluding delta
    
    relative_beta_powers = []
    
    for channel_data in data:
        freqs, psd = welch(channel_data, fs=fs, nperseg=fs*2)
        
        beta_indices = np.logical_and(freqs >= beta_band[0], freqs <= beta_band[1])
        relevant_indices = np.logical_and(freqs >= relevant_band[0], freqs <= relevant_band[1])
        
        beta_power = np.trapz(psd[beta_indices], freqs[beta_indices])
        
        total_relevant_power = np.trapz(psd[relevant_indices], freqs[relevant_indices])
        
        relative_beta_power = beta_power / total_relevant_power if total_relevant_power > 0 else 0
        relative_beta_powers.append(relative_beta_power)
    
    return np.array(relative_beta_powers)

def alpha_band_relative(data):
    from .config import SAMPLING_FREQUENCY
    fs = SAMPLING_FREQUENCY

    beta_band = (8, 13)     # Beta frequency range
    relevant_band = (4, 50)  # Total power range, excluding delta
    
    relative_beta_powers = []
    
    for channel_data in data:
        freqs, psd = welch(channel_data, fs=fs, nperseg=fs*2)
        
        beta_indices = np.logical_and(freqs >= beta_band[0], freqs <= beta_band[1])
        relevant_indices = np.logical_and(freqs >= relevant_band[0], freqs <= relevant_band[1])
        
        beta_power = np.trapz(psd[beta_indices], freqs[beta_indices])
        
        total_relevant_power = np.trapz(psd[relevant_indices], freqs[relevant_indices])
        
        relative_beta_power = beta_power / total_relevant_power if total_relevant_power > 0 else 0
        relative_beta_powers.append(relative_beta_power)
    
    return np.array(relative_beta_powers)

def gamma_band(data):
    from .config import SAMPLING_FREQUENCY
    fs = SAMPLING_FREQUENCY

    gamma_band = (0.5, 4) 
    gamma_powers = []
    
    for channel_data in data:
        freqs, psd = welch(channel_data, fs=fs, nperseg=fs*2)
        
        gamma_indices = np.logical_and(freqs >= gamma_band[0], freqs <= gamma_band[1])
        
        gamma_power = np.trapz(psd[gamma_indices], freqs[gamma_indices])
        gamma_powers.append(gamma_power)
    
    return np.array(gamma_powers)