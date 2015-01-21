import os, random, math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import triang
from scipy.fftpack import fft


def get_fragment(x, sec_begin, sec_end, fs = 44100):
    if sec_end == -1:
        sec_end = len(x) / float(fs)
    begin = get_points(sec_begin, fs)
    end = get_points(sec_end, fs)
    return x[begin:end]

def get_points(sec, fs = 44100):
    return int(fs * sec)


def write_wav(filepath, x, fs = 44100):
    wav.write(
        filepath, fs,
        np.array(x, dtype='int16')
    )

def write_sine(filepath, A = 20000, f = 440, fs = 44100, t = 1):
    wav.write(
        filepath, fs, np.array(
            genRealSine(A, f, 0, fs, t),
            dtype='int16'
        )
    )

def read_wav(filepath, sec_begin = 0, sec_end = -1):
    fs, x = wav.read(filepath)
    return fs, get_fragment(x, sec_begin, sec_end, fs)

def read_wavnormalized(filepath, sec_begin = 0, sec_end = -1):
    fs, x = wav.read(filepath)
    return fs, get_fragment(x / float(np.iinfo(np.int16).max), sec_begin, sec_end, fs)

def show_plot(title, x, ran = None):
    x_min = float(min(x))
    x_max = float(max(x))
    x_amp = x_max - x_min
    if ran == None:
        ran = (0, len(x))
    plt.figure().suptitle(title)
    plt.plot(np.arange(ran[0], ran[1]), x)
    plt.axis([ran[0], ran[1], x_min - 0.3 * x_amp, x_max + 0.3 * x_amp])
    plt.show()

def show_wavplot(filepath, read_func = read_wav, sec_begin = 0, sec_end = -1):
    fs, x = read_func(filepath, sec_begin, sec_end)
    show_plot(filepath, x)


def genRealSine(A = 1, f = 440, phi = 0, fs = 44100, t = 1):
    return A * np.cos(2 * np.pi * f * np.arange(0, t, 1.0 / fs) + phi)

def genComplexSine(A = 1, k = 440, N = 44100):
    n = np.arange(N)
    return A * np.exp(1j * 2 * np.pi * n * k / N)

def genWhiteNoise(A_max = 20000, fs = 44100, t = 1):
    return np.array([random.randint(-A_max, A_max) for i in range(fs * t)])


def DFT(x):
    N = len(x)
    nv = np.arange(-N/2, N/2)
    kv = np.arange(-N/2, N/2)
    s = [np.exp(1j * 2 * np.pi * k * nv / N) for k in kv]
    return np.array([sum(x * np.conjugate(sk)) for sk in s])

def genMagnitudeSpectrum(x):
    return 20 * np.log10(np.real(abs(DFT(x)[len(x)/2:])))

def genPhaseSpectrum(x):
    return np.angle(DFT(x)[len(x)/2:])


def IDFT(X):
    N = len(X)
    kv = np.arange(-N/2, N/2)
    nv = np.arange(-N/2, N/2)
    s = [np.exp(1j * 2 * np.pi * n * kv / N) for n in nv]
    return np.array([sum(X * sn) / N for sn in s])


def analize_DFT(freq = 110, wave_type = 'real', hop_num = 15, Amp = 20000):

    if wave_type == 'real':
        x = genRealSine(A = Amp, f = freq)
    elif wave_type == 'complex':
        x = genComplexSine(A = Amp, k = freq)
    elif wave_type == 'file':
        write_sine('test.wav', A = Amp, f = freq)
        x = read_wav('test.wav')[1]
    x = x[::hop_num]

    N = len(x)
    print('len(x) == %s' % N)

    X = abs(DFT(x))
    y = IDFT(X)
    Y = abs(DFT(y))
    mX = genMagnitudeSpectrum(x)
    pX = genPhaseSpectrum(x)

    show_plot('Wave x', np.real(x))
    show_plot('DFT of wave x', np.real(X), (-N/2, N/2))
    show_plot('Magnitude Spectrum of x', mX)
    show_plot('Phase Spectrum of x', pX)
    show_plot('Wave y', np.real(y))
    show_plot('DFT of wave y', np.real(Y), (-N/2, N/2))
    
    mX_max = max(mX)
    #intdft = [int(i) for i in np.real(mX)]
    #for item in sorted(set(intdft)):
    #    if intdft.count(item) < 10:
    #        print('%s (%s): %s' % (item, intdft.count(item), [idx for idx, val in enumerate(intdft) if val == item]))
    #    else:
    #        print('%s (%s)' % (item, intdft.count(item)))
    print('max DFTs of wave x: %s' % [(idx, val) for idx, val in enumerate(mX) if val == mX_max])


x = triang(15)
N = len(x)
X = fft(fftbuffer)
mX = abs(X[N/2:])
pX = np.angle(X[N/2:])

fftbuffer = np.zeros(15)
fftbuffer[:8] = x[7:]
fftbuffer[8:] = x[:7]
FFTBUFFER = fft(fftbuffer)
mFFTBUFFER = abs(FFTBUFFER[N/2:])
pFFTBUFFER = np.angle(FFTBUFFER[N/2:])

M = 501
hM1 = int(math.floor((M + 1) / 2))
hM2 = int(math.floor(M / 2))
fs, x = read_normalized(os.path.join('Waves', '217543__xserra__orchestra-fragment.wav'))

show_wavplot(os.path.join('Waves', '217543__xserra__orchestra-fragment.wav'), sec_end = 0.01, read_func = read_wav)
