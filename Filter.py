"""
Created on 15/03/20 15:41
@author: xavierlizarraga

Description:

Usage:

"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import signal


scriptPath = Path(__file__)
scriptDir = scriptPath.resolve().parent


class Filter:

    def __init__(self, sampleRate=44100, filterType='low', cutFrequency=1000., order=2, Q=1, filterClass='butter'):
        self.sampleRate = sampleRate
        self.filterType = filterType
        self.cutFrequency = cutFrequency
        self.order = order
        self.Q = Q
        self.bandWidth = self.cutFrequency / self.Q
        self.filterClass = filterClass
        self.w = 0
        self.h = 0

    def __str__(self):
        return '{0}({1}'.format(self.__class__.__name__, self.__dict__)

    def __call__(self):
        if self.filterType == 'allpass':
            wf = 0.99 if self.Q > 2 else self.Q - 1
            b = [wf, 1]
            a = [1, wf]
        elif self.filterClass == 'butter':
            f0 = self.cutFrequency
            if 'band' in self.filterType:
                f1 = self.cutFrequency * (np.sqrt(1 + 1. / (4. * self.Q ** 2)) - 1. / (2. * self.Q))
                f2 = self.cutFrequency * (np.sqrt(1 + 1. / (4. * self.Q ** 2)) + 1. / (2. * self.Q))
                f0 = [f1, f2]
            print('cutFrequencies: {}'. format(self.cutFrequency))
            b, a = signal.butter(self.order, f0,  self.filterType, analog=True)
        else:
            raise ValueError('filter class is not defined.')
        self.w, self.h = signal.freqz(b, a)
        self.displayTransferFunction()

    def setBandWidth(self, bandwidth):
        self.bandWidth = bandwidth
        self.Q = self.cutFrequency / self.bandWidth

    def displayTransferFunction(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.w * self.sampleRate / (2*np.pi), 20 * np.log10(abs(self.h)), 'b', label='magnitude')
        plt.xscale('log')
        plt.title('{}th order-{}-{} filter - frequency response'.format(self.order, self.filterType, self.filterClass))
        plt.xlabel('Frequency [Hertz]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        ax.grid()
        ax21 = ax.twinx()
        angles = np.unwrap(np.angle(self.h))
        ax21.plot(self.w * self.sampleRate / (2*np.pi), angles, 'g', label='phase')
        ax21.set_ylabel('phase (radians)', color='g')
        ax.axvline(self.cutFrequency, color='red', label='fc')  # cutoff frequency
        ax.legend()
        plt.show()


def main():
    filterType = 'allpass'
    order = 1
    cutFrequency = 100
    Q = 0.1
    _filter = Filter(filterType=filterType, order=order, cutFrequency=cutFrequency, Q=Q)
    print(_filter)
    _filter()
    print('Done!!!')


if __name__ == "__main__":
    main()
