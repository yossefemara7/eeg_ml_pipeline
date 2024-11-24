import numpy as np

def katz_fd(signal):
    katz_fds = []
    for channel in signal:
        N = len(signal)
        L = np.sum(np.abs(np.diff(channel)))
        d = np.max(np.abs(channel - channel[0]))
        katz = np.log(L) / np.log(d)
        katz_fds.append(katz)
    
    return katz_fds

def petrosian_fd(signal):
    petrosian_fds = []
    for channel in signal:
        N = len(channel)
        diff_signal = np.diff(channel)
        N_delta = np.sum(diff_signal[:-1] * diff_signal[1:] < 0)
        petrosian =  np.log(N) / (np.log(N) + np.log(N_delta))
        petrosian_fds.append(petrosian)
    
    return petrosian_fds

if __name__ == '__main__':
    for i in range(1000):
        signal = np.random.rand(10, 1000)
        print("Katz FD:", katz_fd(signal))
        print("Petrosian FD:", petrosian_fd(signal))
