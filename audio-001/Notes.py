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
    x_normalized = get_samples(
        x / float(np.iinfo(np.int16).max), sec_begin, sec_end, fs
    )
    return fs, x_normalized
################################################################################


# plots drawing ################################################################
labels = {
    't': 'Time (s)',
    'A1': 'Amplitude',
    'A2': 'Amplitude (dB)',
    'f': 'Frequency (Hz)',
    'p': 'Phase (rad)'
}

def show_plots2d(title, plots, points, xlabel = '', ylabel = ''):
    xv, yv = zip(*plots)
    y_exclNone = [y[y != np.array(None)] for y in yv]
    y_mins, y_maxs = zip(*
        [(float(min(y)), float(max(y))) for y in y_exclNone]
    )
    y_min = min(y_mins)
    y_max = max(y_maxs)
    y_amp = y_max - y_min
    plt.figure().suptitle(title)
    plt.axis(
        [xv[0][0], xv[0][-1], y_min - 0.3 * y_amp, y_max + 0.3 * y_amp]
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for x, y in plots:
        plt.plot(x, y)
    for x, y in points:
        plt.plot(x, y, 'bo')
    plt.show()

def show_plot3d(title, x, y, z, xlabel = '', ylabel = '', zlabel = ''):
    plt.figure().suptitle(title)
    plt.pcolormesh(x, y, z)
    plt.axis([x[0], x[-1], y[0], y[-1]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar().set_label(zlabel)
    plt.show()

def show_wavplot(filepath, read_func = read_wav, sec_begin = 0, sec_end = -1):
    fs, x = read_func(filepath, sec_begin, sec_end)
    show_plots2d(
        filepath, [(np.arange(0, sec_end, 1./fs), x)], [],
        labels['t'], labels['A1']
    )
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
    wM = w(M)
    w_sum = sum(wM)
    xw = x * wM / w_sum
    dft_buffer = np.concatenate(
        (xw[hM2:], np.zeros(N - hM1 - hM2), xw[:hM2])
    )
    return dft_func(dft_buffer)

def genMagnitudeSpectrum_dft(x, w, N, dft_func = fp.fft):
    hN = N / 2
    X = genBufferedDft(x, w, N, dft_func)
    return 20 * np.log10(abs(X[:hN]))

def genPhaseSpectrum_dft(x, w, N, dft_func = fp.fft):
    hN = N / 2
    X = genBufferedDft(x, w, N, dft_func)
    return np.unwrap(np.angle(X[:hN]))

def genSpectrums_dft(x, w, N, dft_func = fp.fft):
    hN = N / 2
    X = genBufferedDft(x, w, N, dft_func)
    mX = 20 * np.log10(abs(X[:hN]))
    pX = np.unwrap(np.angle(X[:hN]))
    return mX, pX
################################################################################


# stft analyzis helpers ########################################################
def genSamplesOfStft(x, w, M, H, dft_func = fp.fft):
    hM1 = int(math.floor((M + 1) / 2))
    hM2 = int(math.floor(M / 2))
    x_ext = np.concatenate((np.zeros(hM2), x, np.zeros(hM2)))
    lv = np.arange(len(x) / H + 1)
    return np.array([x_ext[l * H:l * H + M] for l in lv])

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


# sinusoidal model #############################################################
def interpPeak(x, ploc, scale = 1):
    lval = x[0]
    pval = x[1]
    rval = x[2]
    iploc = ploc + (lval - rval) / (2.0 * (lval - 2 * pval + rval))
    ipval = pval - (lval - rval) * (iploc - ploc) / 4.0
    return iploc * scale, ipval

def findPeaks(x, t, scale = 1):
    return np.array(
        [
            ((ploc + 1) * scale, pval)
            for ploc, pval in enumerate(x[1:-1])
            if pval > t and pval > x[ploc] and pval > x[ploc + 2]
        ]
    )

def interpPeaks(x, t, scale = 1):
    return np.array(
        [
            interpPeak(x[ploc - 1:ploc + 2], ploc, scale)
            for ploc, peak in findPeaks(x, t)
        ]
    )

def linearInterp(iloc, x, y):
    return zip(iloc, np.interp(iloc, x, y))
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
def genSignal_dft(mX, pX, M, idft_func = fp.ifft):
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
    return np.concatenate(
        (dft_buffer[N - hM2:], dft_buffer[:hM1])
    ) * M
################################################################################


# stft synthesis helpers #######################################################
def genSignal_stft(xmX, xpX, M, H, idft_func = fp.ifft):
    hM1 = int(math.floor((M + 1) / 2))
    hM2 = int(math.floor(M / 2))
    spectrums = np.array(zip(np.transpose(xmX), np.transpose(xpX)))
    x_len = spectrums.shape[0] * H
    samples = np.array(
        [H * genSignal_dft(mX, pX, M, idft_func) for mX, pX in spectrums]
    )
    return sum(
        np.array(
            [
                np.concatenate(
                    (
                        np.zeros(hM2 + i * H),
                        sample,
                        np.zeros(x_len - i * H - (sample.shape[0] / 2) + hM2)
                    )
                ) for i, sample in zip(np.arange(samples.shape[0]), samples)
            ]
        )
    )[hM2:] / M
################################################################################



def timer(f):
    start = time.time()
    f()
    return time.time() - start


def dft_analyzeWav(
    input_filepath = 'test-src.wav',
    output_filepath = 'test-dst.wav',
    N = 65536
):

    fs, x = read_wavnormalized(input_filepath)
    M = len(x)
    secs = float(M) / float(fs)

    X = abs(fp.fft(x))
    mX, pX = genSpectrums_dft(
        x, rectangularWindow, N * secs, dft_func = fp.fft
    )
    y = genSignal_dft(mX, pX, M, idft_func = fp.ifft)
    Y = abs(fp.fft(y))

    o_time = np.arange(M) / float(fs)
    o_freqDft = np.arange(M) / secs
    o_freqSpectrum = float(fs) * np.arange(len(mX)) / float(N)

    show_plots2d(
        'Wave x', [(o_time, x)], [],
        labels['t'], labels['A1']
    )
    show_plots2d(
        'DFT of wave x', [(o_freqDft, X)], [],
        labels['f'], labels['A1']
    )
    show_plots2d(
        'Magnitude Spectrum of x', [(o_freqSpectrum, mX)], [],
        labels['f'], labels['A2']
    )
    show_plots2d(
        'Phase Spectrum of x', [(o_freqSpectrum, pX)], [],
        labels['f'], labels['p']
    )
    show_plots2d(
        'Wave y', [(o_time, y)], [],
        labels['t'], labels['A1']
    )
    show_plots2d(
        'DFT of wave y', [(o_freqDft, Y)], [],
        labels['f'], labels['A1']
    )

    write_wavnormalized(output_filepath, y, fs)


def dft_analyzeWavFragment(
    input_filepath = 'test-src.wav',
    t_start = 0.1, M = 801, N = 1024,
    w = rectangularWindow,
    threshold = -120
):

    wav_obj = read_wavnormalized(input_filepath)
    fs = wav_obj[0]
    sample_start = t_start * fs
    x = wav_obj[1][sample_start:sample_start + M]
    secs = float(M) / float(fs)

    X = abs(fp.fft(x))
    mX, pX = genSpectrums_dft(x, w, N, dft_func = fp.fft)
    y = genSignal_dft(mX, pX, M, idft_func = fp.ifft)
    Y = abs(fp.fft(y))

    o_time = np.arange(M) / float(fs)
    o_freqDft = np.arange(M) / secs
    o_freqSpectrum = float(fs) * np.arange(N / 2) / float(N)
    
    ipmag = interpPeaks(mX, threshold, float(fs) / float(N))
    ipphase = linearInterp(zip(*ipmag)[0], o_freqSpectrum, pX)

    show_plots2d(
        'Wave x', [(o_time, x)], [],
        labels['t'], labels['A1']
    )
    show_plots2d(
        'DFT of wave x', [(o_freqDft, X)], [],
        labels['f'], labels['A1']
    )
    show_plots2d(
        'Magnitude Spectrum of x', [(o_freqSpectrum, mX)], [],
        labels['f'], labels['A2']
    )
    show_plots2d(
        'Phase Spectrum of x', [(o_freqSpectrum, pX)], [],
        labels['f'], labels['p']
    )
    show_plots2d(
        'Magnitude Spectrum of x with its peaks',
        [(o_freqSpectrum, mX)], ipmag,
        labels['f'], labels['A2']
    )
    show_plots2d(
        'Phase Spectrum of x with its peaks',
        [(o_freqSpectrum, pX)], ipphase,
        labels['f'], labels['A2']
    )
    show_plots2d(
        'Wave y', [(o_time, y)], [],
        labels['t'], labels['A1']
    )
    show_plots2d(
        'DFT of wave y', [(o_freqDft, Y)], [],
        labels['f'], labels['A1']
    )


def stft_analyzeWav(
    input_filepath = 'test-src.wav',
    output_filepath = 'test-dst.wav',
    M = 801, N = 2048, H = 250,
    w = rectangularWindow
):

    fs, x = read_wavnormalized(input_filepath)
    
    xmX, xpX = genSpectrums_stft(x, w, M, N, H)
    y = genSignal_stft(xmX, xpX, M, H)

    o_timex = np.arange(len(x)) / float(fs)
    o_timey = np.arange(len(y)) / float(fs)
    ox = H * np.arange(xmX.shape[1]) / float(fs)
    oy = fs * np.arange(xmX.shape[0]) / N
    
    show_plots2d(
        'Wave x', [(o_timex, x)], [],
        labels['t'], labels['A1']
    )
    show_plot3d(
        '3D magnitude spectrum of x', ox, oy, xmX,
        labels['t'], labels['f'], labels['A2']
    )
    show_plot3d(
        '3D phase spectrum of x', ox, oy, np.diff(xpX, axis = 0),
        labels['t'], labels['f'], labels['p']
    )
    show_plots2d(
        'Wave y', [(o_timey, y)], [],
        labels['t'], labels['A1']
    )

    write_wavnormalized(output_filepath, y, fs)
