import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy

def spherical_to_cartesian(r, theta, phi):
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)])


def make_simple_signals(n_channels, n_samples, center_frequency=0.1, magnitude=1.0):
    """
    Make simple multichannel time-domain signal.
    :param n_channels:
    :param center_frequency: normalized frequency of the sine wave
    :param n_samples:
    :return:
    """

    # center_freq_hz = center_frequency * fs
    # interval_n = fs * (1/center_freq_hz) = 1/ center_Frequency

    timesteps = range(n_samples)
    # main sine wave
    sin_period = 1.0 / center_frequency
    signals = magnitude * np.sin(np.stack([timesteps for i in range(n_channels)]) / sin_period * 2 * np.pi)
    # noise
    signals += np.random.normal(loc=0.0, scale=magnitude*0.0, size=[n_channels, n_samples])
    return signals



def make_steering_vector(mic_positions, wave_vector, frequencies, speed_of_sound=340, plot_system=False):
    """
    :param mic_positions: microphone positions in meters
    :param wave_vector: vector towards the wave DOA
    :param frequencies: vector of multiples of fundamental frequencies from fft
    :return:
    steering vector of size (n_channels, n_frequencies, 1).
    Last dim 1 is for easier multiplication with the result of stft of size(n_channels, n_frequencies, n_stft)
    """
    wave_vector /= np.linalg.norm(wave_vector)
    distance_differences = np.dot(mic_positions, np.expand_dims(wave_vector, 0).T)
    wavenumbers = np.expand_dims(np.array(2 * np.pi * frequencies / speed_of_sound), 1)
    steering_vector = np.exp(1j * distance_differences * wavenumbers.T)

    # plot
    if plot_system:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.quiver(0, 0, 0, *wave_vector, length=1.0)
        ax.scatter(mic_positions[:, 0], mic_positions[:, 1])
        for i in range(len(mic_positions)):
            ax.quiver(*mic_positions[i], *(-wave_vector), length=distance_differences[i][0])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax.axes.set_zlim3d(-150, 150)
        plt.show()

    return np.squeeze(distance_differences), np.expand_dims(steering_vector, axis=2)

def compute_gcc_phat(signals, lag_range, rfft_args=None):
    """
    Compute gcc-phat between the channel 0 and the other channels.
    :param signals: Array of multichannel samples of size [n_channels, n_samples]
    :return: Array of gcc-phat result of size [n_channels, n_]
    First element is always 0.0 since it is used as the reference signal.
    """

