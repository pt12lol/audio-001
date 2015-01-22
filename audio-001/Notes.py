import os, random, math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sgnl
import scipy.fftpack as fp



def get_fragment(x, sec_begin = 0, sec_end = -1, fs = 44100):
    begin = get_samples(sec_begin, fs)
    end = get_samples(sec_end if sec_end == -1 else len(x) / float(fs), fs)
    return x[begin:end]

def get_samples(sec, fs = 44100):
    return int(fs * sec)


def write_wav(filepath, x, fs = 44100):
    wav.write(
        filepath, fs,
        np.array(x, dtype='int16')
    )

def write_sine(filepath, A = 20000, f = 440, fs = 44100, t = 1):
    write_wav(filepath, genRealSine(A, f, 0, fs, t), fs)

def read_wav(filepath, sec_begin = 0, sec_end = -1):
    fs, x = wav.read(filepath)
    return fs, get_fragment(x, sec_begin, sec_end, fs)

def read_wavnormalized(filepath, sec_begin = 0, sec_end = -1):
    fs, x = wav.read(filepath)
    return fs, get_fragment(x / float(np.iinfo(np.int16).max), sec_begin, sec_end, fs)

def show_plot(title, x, y):
    y_min = float(min(y))
    y_max = float(max(y))
    y_amp = y_max - y_min
    plt.figure().suptitle(title)
    plt.plot(x, y)
    plt.axis([x[0], x[-1], y_min - 0.3 * y_amp, y_max + 0.3 * y_amp])
    plt.show()

def show_wavplot(filepath, read_func = read_wav, sec_begin = 0, sec_end = -1):
    fs, x = read_func(filepath, sec_begin, sec_end)
    show_plot(filepath, np.arange(0, sec_end, 1./fs), x)



def genRealSine(A = 1, f = 440, phi = 0, fs = 44100, t = 1):
    return A * np.cos(2 * np.pi * f * np.arange(0, t, 1.0 / fs) + phi)

def genComplexSine(A = 1, k = 440, N = 44100):
    n = np.arange(N)
    return A * np.exp(1j * 2 * np.pi * n * k / N)

def genWhiteNoise(A_max = 20000, fs = 44100, t = 1):
    return np.array([random.randint(-A_max, A_max) for i in range(fs * t)])


def dft(x):
    N = len(x)
    nv = np.arange(N)
    kv = np.arange(N)
    s = [np.exp(1j * 2 * np.pi * k * nv / N) for k in kv]
    return np.array([sum(x * np.conjugate(sk)) for sk in s])

def genBufferedDft(x, w = lambda n: 1, dftlen = -1, dft_func = fp.fft):
    N = len(x)
    hM1 = int(math.floor((N + 1) / 2))
    hM2 = int(math.floor(N / 2))
    xw = x * w(N)
    dftbuffer = np.concatenate(
        (xw[hM2:], np.zeros((dftlen if dftlen != -1 else len(x)) - hM1 - hM2), xw[:hM2])
    )
    return dft_func(dftbuffer)

def genMagnitudeSpectrum(x, w = lambda n: 1, dftlen = -1, dft_func = fp.fft):
    N = len(x)
    X = genBufferedDft(x, w, dftlen)
    return 20 * np.log10(np.real(abs(X[:N/2])))

def genPhaseSpectrum(x, w = lambda n: 1, dftlen = -1, dft_func = fp.fft):
    N = len(x)
    X = genBufferedDft(x, w, dftlen)
    return np.unwrap(np.angle(X[:N/2]))


def idft(X):
    N = len(X)
    kv = np.arange(N)
    nv = np.arange(N)
    s = [np.exp(1j * 2 * np.pi * n * kv / N) for n in nv]
    return np.array([sum(X * sn) / N for sn in s])



def analize_dft(source = 'real', amp = 1, freq = 110, hop_num = 15, dftrate = 1, dft_func = fp.fft, idft_func = fp.ifft):

    x = genRealSine(A = amp, f = freq)[::hop_num] if source == 'real' else \
        genComplexSine(A = amp, k = freq)[::hop_num] if source == 'complex' else \
        read_wav(source)[1][::hop_num]

    N = len(x)
    print('len(x) == %s' % N)

    X = abs(dft_func(x))
    mX = genMagnitudeSpectrum(x, dftlen = dftrate * N)
    pX = genPhaseSpectrum(x, dftlen = dftrate * N)
    y = idft_func(X)
    Y = abs(dft_func(y))

    show_plot('Wave x', np.arange(N), np.real(x))
    show_plot('DFT of wave x', np.arange(N), np.real(X))
    show_plot('Magnitude Spectrum of x', np.arange(0, N / (dftrate * 2.0), 1.0 / dftrate), mX)
    show_plot('Phase Spectrum of x', np.arange(0, N / (dftrate * 2.0), 1.0 / dftrate), pX)
    show_plot('Wave y', np.arange(N), np.real(y))
    show_plot('DFT of wave y', np.arange(0, N), np.real(Y))
    
    mX_max = max(mX)
    print('max DFTs of wave x: %s' % [(idx / dftrate, val) for idx, val in enumerate(mX) if val == mX_max])
