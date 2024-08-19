import acoustics
import numpy as np
import scipy
from matplotlib import pyplot as plt

# generate 4 channel signals
fs = 120
signals = acoustics.make_simple_signals(4, fs*2, center_frequency=10.0/fs)

# apply lags
sample_lags = [0, 10, 20, 30]
signals = np.stack([np.roll(signals[i], sample_lags[i]) for i in range(len(sample_lags))])

# perform fft
SIGNALS = scipy.fft.rfft(signals)
fft_frequencies = scipy.fft.rfftfreq(signals.shape[-1]) * fs

# gcc-phat
REF_SIGNAL = SIGNALS
CROSS_CORRELATIONS = SIGNALS[1] * np.conj(REF_SIGNAL)
GENERALIZED_CROSS_CORRELATIONS = CROSS_CORRELATIONS / np.abs(CROSS_CORRELATIONS)
print(GENERALIZED_CROSS_CORRELATIONS.shape)

# restore time domain gcc
gcc = scipy.fft.irfft(GENERALIZED_CROSS_CORRELATIONS)
print(np.argmax(gcc, axis=-1))