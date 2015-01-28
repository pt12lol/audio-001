import os, random, math, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sgnl
import scipy.fftpack as fp
import scipy.signal.windows as windows



# samples management ###########################################################
def get_samples(x, sec_begin = 0, sec_end = -1, fs = 44100):
    begin = get_samplenum(sec_begin, fs)
    end = get_samplenum(sec_end if sec_end != -1 else len(x) / float(fs), fs)
    return x[begin:end]

def get_samplenum(sec, fs = 44100):
    return fs * sec
################################################################################


# files manipulation ###########################################################
def write_wav(filepath, x, fs = 44100):
    wav.write(
        filepath, fs,
        np.array(x, dtype='int16')
    )

def write_wavnormalized(filepath, x, fs = 44100):
    write_wav(filepath, x * np.iinfo(np.int16).max, fs)

def write_sine(filepath, A = 20000, f = 440, fs = 44100, t = 1):
    write_wav(filepath, genRealSine(A, f, 0, fs, t), fs)

def read_wav(filepath, sec_begin = 0, sec_end = -1):
    fs, x = wav.read(filepath)
    return fs, get_samples(x, sec_begin, sec_end, fs)

def read_wavnormalized(filepath, sec_begin = 0, sec_end = -1):
    fs, x = wav.read(filepath)
    return fs, get_samples(x / float(np.iinfo(np.int16).max), sec_begin, sec_end, fs)
################################################################################


# plot drawing #################################################################
def show_plot2d(title, x, y):
    y_min = float(min(y))
    y_max = float(max(y))
    y_amp = y_max - y_min
    plt.figure().suptitle(title)
    plt.plot(x, y)
    plt.axis([x[0], x[-1], y_min - 0.3 * y_amp, y_max + 0.3 * y_amp])
    plt.show()

def show_plot3d(title, x, y, z):
    plt.figure().suptitle(title)
    plt.pcolormesh(x, y, z)
    plt.axis([x[0], x[-1], y[0], y[-1]])
    plt.show()

def show_wavplot(filepath, read_func = read_wav, sec_begin = 0, sec_end = -1):
    fs, x = read_func(filepath, sec_begin, sec_end)
    show_plot2d(filepath, np.arange(0, sec_end, 1./fs), x)
################################################################################


# signal generators ############################################################
def genRealSine(A = 1, f = 440, phi = 0, fs = 44100, t = 1):
    return A * np.cos(2 * np.pi * f * np.arange(0, t, 1.0 / fs) + phi)

def genComplexSine(A = 1, k = 440, N = 44100):
    n = np.arange(N)
    return A * np.exp(2j * np.pi * n * k / N)

def genWhiteNoise(A_max = 20000, fs = 44100, t = 1):
    return np.array([random.randint(-A_max, A_max) for i in range(fs * t)])
################################################################################


# windows helpers ##############################################################
def rectangularWindow(M):
    return np.ones(M)

def genKaiser(beta):
    return lambda M: windows.kaiser(M, beta)
################################################################################


# fourier transform helpers ####################################################
def __fft_acc(x, cutoff):
    N = len(x)
    X_even = fft(x[::2])
    X_odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate(
        [
            X_even + factor[:N / 2] * X_odd,
            X_even + factor[N / 2:] * X_odd
        ]
    )

def dft(x):
    N = len(x)
    nv = np.arange(N)
    kv = np.arange(N)
    s = [np.exp(2j * np.pi * k * nv / N) for k in kv]
    return np.array([sum(x * np.conjugate(sk)) for sk in s])

def fft(x, cutoff = 32):
    N = len(x)
    return dft(x) if N % 2 > 0 or N < cutoff else __fft_acc(x, cutoff)

def genFft(cutoff):
    return lambda x: fft(x, cutoff)
################################################################################


# dft analyzis helpers #########################################################
def genBufferedDft(x, w, N, dft_func = fp.fft):
    M = len(x)
    hM1 = int(math.floor((M + 1) / 2))
    hM2 = int(math.floor(M / 2))
    w_M = w(M)
    w_sum = sum(w_M)
    xw = x * w_M / w_sum
    dft_buffer = np.concatenate(
        (xw[hM2:], np.zeros(N - hM1 - hM2), xw[:hM2])
    )
    return dft_func(dft_buffer)

def genMagnitudeSpectrum_dft(x, w, N, dft_func = fp.fft):
    hN = N / 2
    X = genBufferedDft(x, w, N, dft_func)
    return 20 * np.log10(np.real(abs(X[:hN])))

def genPhaseSpectrum_dft(x, w, N, dft_func = fp.fft):
    hN = N / 2
    X = genBufferedDft(x, w, N, dft_func)
    return np.unwrap(np.angle(X[:hN]))

def genSpectrums_dft(x, w, N, dft_func = fp.fft):
    hN = N / 2
    X = genBufferedDft(x, w, N, dft_func)
    mX = 20 * np.log10(np.real(abs(X[:hN])))
    pX = np.unwrap(np.angle(X[:hN]))
    return mX, pX
################################################################################


# stft analyzis helpers ########################################################
def genSamplesOfStft(x, w, M, H, dft_func = fp.fft):
    hM1 = int(math.floor((M + 1) / 2))
    hM2 = int(math.floor(M / 2))
    x_ext = np.concatenate((np.zeros(hM2), x, np.zeros(hM2)))
    samples_num = (len(x_ext) - 2 * hM1) / H
    lv = np.arange(hM1, hM1 * samples_num, H)
    w_M = w(M)
    w_sum = sum(w_M)
    return np.array([x_ext[l - hM1:l + hM2] * w_M / w_sum for l in lv])

def genMagnitudeSpectrum_stft(x, w, M, N, H, dft_func = fp.fft):
    xlv = genSamplesOfStft(x, w, M, H, dft_func)
    return np.transpose(
        np.array(
            [genMagnitudeSpectrum_dft(xl, w, N, dft_func) for xl in xlv]
        )
    )

def genPhaseSpectrum_stft(x, w, M, N, H, dft_func = fp.fft):
    xlv = genSamplesOfStft(x, w, M, H, dft_func)
    return np.transpose(
        np.array(
            [genPhaseSpectrum_dft(xl, w, N, dft_func) for xl in xlv]
        )
    )

def genSpectrums_stft(x, w, M, N, H, dft_func = fp.fft):
    xlv = genSamplesOfStft(x, w, M, H, dft_func)
    return tuple(
        np.transpose(np.array(X))
        for X in zip(
            *[genSpectrums_dft(xl, w, N, dft_func) for xl in xlv]
        )
    )
################################################################################


# inverse fourier transform helpers ############################################
def __ifft_acc(X, cutoff):
    N = len(X)
    x_even = ifft(X[::2])
    x_odd = ifft(X[1::2])
    factor = np.exp(2j * np.pi * np.arange(N) / N)
    return np.concatenate(
        [
            x_even + factor[:N / 2] * x_odd,
            x_even + factor[N / 2:] * x_odd
        ]
    ) / 2

def idft(X):
    N = len(X)
    kv = np.arange(N)
    nv = np.arange(N)
    s = [np.exp(2j * np.pi * n * kv / N) for n in nv]
    return np.array([sum(X * sn) / N for sn in s])

def ifft(X, cutoff = 32):
    N = len(X)
    return idft(X) if N % 2 > 0 or N < cutoff else __ifft_acc(X, cutoff)

def genIfft(cutoff):
    return lambda X: ifft(X, cutoff)
################################################################################


# dft synthesis helpers ########################################################
def genSignal(mX, pX, w, M, idft_func = fp.ifft):
    N = len(mX) * 2
    hN = N / 2
    hM1 = int(math.floor((M + 1) / 2))
    hM2 = int(math.floor(M / 2))
    X = np.concatenate(
        (
            10 ** (mX / 20) * np.exp(1j * pX),
            np.zeros(1, dtype = complex),
            10 ** (mX[:0:-1] / 20) * np.exp(-1j * pX[:0:-1])
        )
    )
    dft_buffer = np.real(idft_func(X))
    x = np.concatenate(
        (dft_buffer[N - hM2:], dft_buffer[:hM1])
    )
    w_x = np.array([item if item != 0 else np.finfo(float).eps for item in w(M)])
    w_sum = sum(w_x)
    return x * w_sum / w_x
################################################################################



def timer(f):
    start = time.time()
    f()
    return time.time() - start


def dft_test(
    real = True, A = 1, f = 110, H = 10,
    N = 50000, dft_func = fp.fft, idft_func = fp.ifft,
    w = rectangularWindow
):

    x = (
        genRealSine(A = A, f = f) if real else
        genComplexSine(A = A, k = f)
    )[::H]
    
    x_len = len(x)

    X = abs(dft_func(x))
    mX, pX = genSpectrums_dft(x, w, N, dft_func = dft_func)
    y = genSignal(mX, pX, w, x_len, idft_func = idft_func)
    Y = abs(dft_func(y))


def dft_analizeSinusoid(
    real = True, A = 1, f = 110, H = 10,
    N = 50000, dft_func = fp.fft, idft_func = fp.ifft,
    w = rectangularWindow
):

    x = (
        genRealSine(A = A, f = f) if real else
        genComplexSine(A = A, k = f)
    )[::H]

    fs = 44100.0
    x_len = len(x)

    X = abs(dft_func(x))
    mX, pX = genSpectrums_dft(x, w, N, dft_func = dft_func)
    y = genSignal(mX, pX, w, x_len, idft_func = idft_func)
    Y = abs(dft_func(y))

    o_time = np.arange(x_len)
    o_freqSpectrum = np.arange(0, fs / (2.0 * H), fs / (H * N))

    show_plot2d('Wave x', o_time, np.real(x))
    show_plot2d('DFT of wave x', o_time, np.real(X))
    show_plot2d('Magnitude Spectrum of x', o_freqSpectrum, mX)
    show_plot2d('Phase Spectrum of x', o_freqSpectrum, pX)
    show_plot2d('Wave y', o_time, np.real(y))
    show_plot2d('DFT of wave y', o_time, np.real(Y))


def dft_analyzeWav(
    input_filepath, output_filepath = 'test.wav',
    N = 50000, w = rectangularWindow
):

    fs, x = read_wavnormalized(input_filepath)
    x_len = len(x)
    secs = float(x_len) / float(fs)

    X = abs(fp.fft(x))
    mX, pX = genSpectrums_dft(x, w, N * secs, dft_func = fp.fft)
    y = genSignal(mX, pX, w, x_len, idft_func = fp.ifft)
    Y = abs(fp.fft(y))

    o_time = np.arange(x_len) / float(fs)
    o_freqDft = np.arange(x_len) / secs
    o_freqSpectrum = np.arange(0, x_len / 2, float(fs) / float(N)) / secs

    show_plot2d('Wave x', o_time, np.real(x))
    show_plot2d('DFT of wave x', o_freqDft, np.real(X))
    show_plot2d('Magnitude Spectrum of x', o_freqSpectrum, mX)
    show_plot2d('Phase Spectrum of x', o_freqSpectrum, pX)
    show_plot2d('Wave y', o_freqDft, np.real(y))
    show_plot2d('DFT of wave y', o_time, np.real(Y))

    write_wavnormalized(output_filepath, y, fs)


def stft_analyzeWav(
    input_filepath, output_filepath = 'test.wav',
    M = 801, N = 1024, H = 400, w = rectangularWindow
):
    fs, x = read_wavnormalized(input_filepath)
    x_len = len(x)
    secs = float(x_len) / float(fs)
    xmX, xpX = genSpectrums_stft(x, w, M, N, H)
    ox, oy = H * np.arange(xmX.shape[1]) / float(fs), fs * np.arange(N / 2) / N
    show_plot3d('3D magnitude spectrum of x', ox, oy, xmX)
    show_plot3d('3D phase spectrum of x', ox, oy, np.diff(xpX, axis = 1))
