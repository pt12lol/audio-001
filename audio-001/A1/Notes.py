import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def get_sinewave(A, f, phi, t = 1, fs = 44100):
    return A * np.cos(2 * np.pi * f * np.arange(0, t, 1.0 / fs) + phi)

def prepare_plot(filepath, sec_begin = 0, sec_end = -1):
    (fs, x) = wav.read(filepath)
    if sec_end == -1:
        sec_end = x.size / float(fs)
    begin = int(fs * sec_begin)
    end = int(fs * sec_end)
    t = np.arange(x.size) / float(fs)
    plt.plot(t[begin:end], x[begin:end])

def write_sinewave(filepath, A = 20000, f = 440, t = 1, fs = 44100):
    wav.write(
        filepath, fs, np.array(
            get_sinewave(A, f, 0, t, fs),
            dtype='int16'
        )
    )
