import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def write(filename, freq):
    x = genSine(A = 10000, f = freq)
    x = np.array(x, 'int16')
    wav.write(filename, 44100, x)

def read(filename):
    r, x = wav.read(filename)
    return x


def genSine(A = 1, f = 440, phi = 0, fs = 44100, t = 1):
    return A * np.cos(2 * np.pi * f * np.arange(0, t, 1.0 / fs) + phi)

def genComplexSine(k, N):
    n = np.arange(N)
    return np.exp(1j * 2 * np.pi * n * k / N)

def DFT(x):
    N = len(x)
    nv = np.arange(-N/2, N/2)
    kv = np.arange(-N/2, N/2)
    s = [np.exp(1j * 2 * np.pi * k * nv / N) for k in kv]
    return np.array([sum(x * np.conjugate(sk)) for sk in s])

def IDFT(X):
    N = len(X)
    kv = np.arange(-N/2, N/2)
    nv = np.arange(-N/2, N/2)
    s = [np.exp(1j * 2 * np.pi * n * kv / N) for n in nv]
    return np.array([sum(X * sn) / N for sn in s])


freq = 103

#x = genComplexSine(freq, 1000)
#x = genSine(A = 10000, f = freq)[::10]
#write('test.wav', freq)
x = read('test.wav')[::10]

N = len(x)
print('len(x) == %s' % N)


dft = abs(DFT(x))
idft = IDFT(dft)
dft2 = abs(DFT(idft))


plt.plot(np.real(x))
#plt.axis([0, N, -2, 2])
plt.axis([0, N, -20000, 20000])
plt.show()


plt.plot(np.arange(-N/2, N/2), np.real(dft))
plt.axis([-N/2, N/2, 0, N])
plt.show()


intdft = [int(i) for i in np.real(dft)]
for item in sorted(set(intdft)):
    if intdft.count(item) < 10:
        print '%s (%s): %s' % (item, intdft.count(item), [idx for idx, val in enumerate(intdft) if val == item])
    else:
        print '%s (%s)' % (item, intdft.count(item))

maxdft = max(dft)
print([(idx - N/2, val) for idx, val in enumerate(dft) if val == maxdft])


plt.plot(np.arange(-N/2, N/2), np.real(idft))
#plt.axis([-N/2, N/2, -2, 2])
plt.axis([-N/2, N/2, -20000, 20000])
plt.show()


#plt.plot(np.real(x))
#plt.plot(np.imag(x))
plt.plot(np.arange(-N/2, N/2), np.real(dft))
plt.axis([-N/2, N/2, 0, N])
plt.show()


maxdft = max(dft)
print([(idx - N/2, val) for idx, val in enumerate(dft) if val == maxdft])
