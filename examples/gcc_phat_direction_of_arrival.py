import acoustics
import numpy as np
import scipy

# generate 4 channel signals with different lags
sample_rate, _, _, signals = acoustics.make_simple_signals_fourier(4)

# perform fft
signals_fourier = scipy.fft.rfftn(signals)
# apply time
lags = [0, -100, 200, -300]
print(signals_fourier.shape)