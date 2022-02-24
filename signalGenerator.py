# TODO: get stft and istft function or install librosa in python9
#from librosa.core import stft, istft
#from librosa.util import fix_length
from cgi import test
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.io.wavfile import write as writeWavFile
from scipy.signal.windows import hann, hanning, blackman
import sys
from more_termcolor import colored



##############################################################################


scriptPath = Path(__file__)
scriptDir = scriptPath.resolve().parent
audioDataDir = scriptDir / "data" / "audio"
plotDir = scriptDir / "data" / "plots"


##############################################################################


class SignalGenerator:
    def __init__(
        self,
        signalType,
        duration,
        amplitude=1.0,
        sampleRate=44100,
        frequency=20,
        depth=16,
        nHarmonics=10,
        fadeIn=0.1,
        fadeOut=0.1,
        windowType="hann",
        nCycles=1,
        cycleAttenuation=0,
        silence: float = 1.0,
        logger=None,
    ):
        self.signalType = signalType
        self.duration = duration
        self.amplitude = amplitude
        self.sampleRate = sampleRate
        self.frequency = frequency
        self.depth = depth
        self.nSamples = int(duration * sampleRate)
        self.nHarmonics = nHarmonics
        self.coloredNoises = ["pink_noise", "brown_noise", "blue_noise", "purple_noise"]
        self.timeStamps = np.arange(0, self.duration, 1.0 / self.sampleRate)
        self.fadeIn = fadeIn
        self.fadeOut = fadeOut
        self.fadeInSamples = self.fadeIn * self.sampleRate
        self.fadeOutSamples = self.fadeOut * self.sampleRate
        # TODO - check fadeIn + fadeOut < duration
        self.windowType = windowType
        self.windowIn = 0
        self.windowOut = 0
        self.nCycles = nCycles
        self.silence = silence
        self.cycleAttenuation = cycleAttenuation
        if logger:
            self.logger = logger
        else:
            self.logger = getLogger(level="DEBUG")

    def __str__(self):
        return "{0}({1}".format(self.__class__.__name__, self.__dict__)

    def __reset__(self):
        silenceSamples = int(self.silence * self.sampleRate)
        output = np.zeros(
            shape=(
                int(self.nSamples * self.nCycles + silenceSamples * (self.nCycles - 1)),
            ),
            dtype=np.float64,
        )
        # generate window function for fading
        self.fadeInSamples = int(self.fadeIn * self.sampleRate)
        self.fadeOutSamples = int(self.fadeOut * self.sampleRate)
        assert self.fadeInSamples + self.fadeOutSamples < self.nSamples
        self.windowIn = hann(self.fadeInSamples * 2)[: self.fadeInSamples]
        self.windowOut = hann(self.fadeOutSamples * 2)[
            self.fadeOutSamples * 2 - self.fadeOutSamples :
        ]
        return output, silenceSamples

    def __call__(self):
        output, silenceSamples = self.__reset__()
        testSignal = self.generateTestSignal()
        # apply fades
        # print(self.windowIn.shape)
        if self.fadeInSamples + self.fadeOutSamples < self.nSamples:
            self.logger.debug(
                f"testSignal.shape: {testSignal.shape} self.windowIn.shape: {self.windowIn.shape}"
            )
            testSignal[: self.fadeInSamples] = (
                self.windowIn * testSignal[: self.fadeInSamples]
            )
            self.logger.debug(
                f"testSignal.shape: {testSignal.shape}, self.windowIn.shape: {self.windowIn.shape}"
            )
            self.logger.debug(self.nSamples)
            testSignal[self.nSamples - self.fadeOutSamples :] = (
                self.windowOut * testSignal[self.nSamples - self.fadeOutSamples :]
            )
        else:
            self.logger.warning(
                "Cannot apply fades because the total fade duration is longer than total duration."
            )
        # Apply cycles (repetitions) and silences
        pind = 0
        for n in np.arange(1, self.nCycles + 1):
            pend = int(self.nSamples * n + (n - 1) * silenceSamples)
            # cycle attenuation
            cycleAmplitude = (
                self.amplitude
                if self.cycleAttenuation == 0
                else self.decibelsToGain(
                    self.gainToDecibels(self.amplitude)
                    - (n - 1) * self.cycleAttenuation
                )
            )
            self.logger.debug("testSignal.shape: {}".format(testSignal.shape))
            output[pind:pend] = cycleAmplitude * testSignal
            pind = pend + silenceSamples
        return output

    def generateTestSignal(self):
        self.logger.info(f"Generating {self.signalType} signal")
        if self.signalType == "sine":
            testSignal = self.generateSineWave()
        elif self.signalType == "triangle":
            testSignal = self.generateTriangleWave()
        elif self.signalType == "sawtooth":
            testSignal = self.generateSawToothWave()
        elif self.signalType == "square":
            testSignal = self.generateSquareWave()
        elif self.signalType == "white_noise":
            testSignal = self.generateWhiteNoise()
        # elif any([self.signalType == color for color in self.coloredNoises]):
        #     testSignal = self.generateColoredNoise()
        elif self.signalType == "MLS":
            testSignal = self.generateMLS(self.depth)
        elif self.signalType == "delta":
            testSignal = self.generateDelta()
        elif self.signalType == "pulse":
            testSignal = self.generatePulse()
        elif self.signalType == "level":
            testSignal = self.generateLevelValues(self.depth)
        elif self.signalType == "lin_ramp":
            testSignal = self.generate_lin_ramp()
        elif self.signalType == "log_ramp":
            testSignal = self.generate_log_ramp()
        elif any(
            [self.signalType == signal for signal in ["log_sweep", "inv_log_sweep"]]
        ):
            testSignal = self.generateLogSweepTone(compensated=False)
        elif any(
            [self.signalType == signal for signal in ["lin_sweep", "inv_lin_sweep"]]
        ):
            testSignal = self.generateLinearSweepTone(compensated=False)
        elif any(
            [
                self.signalType == signal
                for signal in ["pink_log_sweep", "inv_pink_log_sweep"]
            ]
        ):
            testSignal = self.generateLogSweepTone(compensated=True)
        elif any(
            [
                self.signalType == signal
                for signal in ["pink_lin_sweep", "inv_pink_lin_sweep"]
            ]
        ):
            testSignal = self.generateLinearSweepTone(compensated=True)
        else:
            raise ValueError("Undefined signalType. Please revise your commands.")
        return testSignal

    def generateSineWave(self):
        return self.generateSine(self.amplitude, self.frequency)

    def generateSine(self, amplitude, frequency):
        return amplitude * np.sin(2.0 * np.pi * frequency * self.timeStamps)

    def generateTriangleWave(self):
        output = np.zeros(shape=(self.nSamples,), dtype=np.float64)
        for i in np.arange(self.nHarmonics):
            n = 2 * i + 1
            sine = (
                ((-1.0) ** i) * (n ** -2.0) * self.generateSine(1, self.frequency * n)
            )
            output += sine
        return 8.0 / np.pi ** 2 * output

    def generateSquareWaveAlias(self):
        pass

    def generateSquareWave(self):
        sine = self.generateSineWave()
        # create square wave
        sine = sine * 100
        sine[sine >= 1] = 1
        sine[sine <= -1] = -1
        # normalize sine between 0 to 1
        square = self.amplitude * sine
        return square

    def generateInvertedSawTooth(self):
        pass

    def generateSawToothWave(self):
        output = np.zeros(shape=(self.nSamples,), dtype=np.float64)
        for i in np.arange(1, self.nHarmonics + 1):
            output += (-(1.0 ** (i - 1))) * self.generateSine(
                1.0 / i, self.frequency * i
            )
        return (2.0 / np.pi) * output

    def generateWhiteNoise(self):
        return np.random.uniform(low=-1, size=len(self.timeStamps))

# TODO: get stft and istft function or install librosa in python9
    # def generateColoredNoise(self):
    #     nFFT = 2048
    #     N = int(self.duration * self.sampleRate)
    #     self.logger.debug("N: {}".format(N))
    #     x = self.generateWhiteNoise()
    #     xPad = fix_length(x, N + nFFT // 2)
    #     X = stft(xPad)
    #     print(X.shape)
    #     k = np.tile(
    #         np.arange(1, X.shape[0] + 1), (X.shape[1], 1)
    #     ).T  # pink and brown noise
    #     print(k, k.shape)
    #     """
    #     k = 1 / k           ---->   brown (-6dB/oct)
    #     k = 1 / sqrt(k)     ---->   pink (-3dB/oct)
    #     k = k               ---->   purplet or violet (+6dB/oct)
    #     k = sqrt(k)         ---->   blue  (+3dB/oct)
    #     """
    #     if self.signalType != "purple_noise":
    #         if self.signalType == "pink_noise":
    #             k = 1 / np.sqrt(k)
    #         elif self.signalType == "brown_noise":
    #             k = 1 / k
    #         elif self.signalType == "blue_noise":
    #             k = np.sqrt(k)
    #     X = X * k
    #     y = istft(X)
    #     self.logger.debug("y.shape: {}".format(y.shape))
    #     y = y[:N]
    #     self.logger.debug("y[:N].shape: {}".format(y.shape))
    #     y = y - np.mean(y)
    #     y = y / np.std(y)
    #     y = y / np.max(np.abs(y))
    #     return y

    def generateMLS(self, N):
        nSamples = int(pow(2, N))
        taps = 4
        tap1 = 1
        tap2 = 2
        tap3 = 4
        tap4 = 15
        if N != 16:
            raise ValueError(
                "Sorry but for now MLS signal is only defined for 16 bits."
            )
        buffer = [1] * N
        samples = np.zeros((nSamples,), dtype=np.float64)
        for i in np.arange(nSamples - 1, 0, -1):
            # feedback bit
            xorbit = buffer[tap1] ^ buffer[tap2]
            # second logic level
            if taps == 4:
                xorbit2 = buffer[tap3] ^ buffer[tap4]
                xorbit = xorbit ^ xorbit2
            for j in np.arange(N - 1, 0, -1):
                temp = buffer[j - 1]
                buffer[j] = temp
            buffer[0] = xorbit
            samples[i] = (-2 * xorbit) + 1.0
        self.nSamples = len(samples)
        return samples

    def generateDelta(self):
        samples = np.zeros((self.nSamples, 1), dtype=np.float64)
        samples[0] = 1.0
        return samples

    def generatePulse(self):
        samples = np.ones((self.nSamples, 1), dtype=np.float64)
        samples[0] = 0.0
        samples[-1] = 0.0
        return samples

    # TODO: fix errors in level signal
    def generateLevelValues(self, depth):
        nAmplitudes = int(pow(2, depth - 1))
        interval = 1.0 / nAmplitudes
        samples = np.zeros((nAmplitudes + 1,), dtype=np.float64)
        for i in np.arange(0, nAmplitudes):
            samples[i] = i * interval
        return samples

    def generate_lin_ramp(self):
        output = np.zeros((self.nSamples,), dtype=np.float64)
        for n in range(1, math.ceil(self.nSamples) - 1):
            output[n] = n / (self.nSamples - 2)
        return output

    def generate_log_ramp(self):
        decibel_range = -(self.gainToDecibels(2.0) * self.depth)
        print(self.nSamples)
        output = np.linspace(decibel_range, 0, self.nSamples, dtype=np.float64)
        print(len(output))
        return self.decibelsToGain(output)

    def generateLogSweepTone(
        self, startFrequency=20.0, stopFrequency=20000.0, compensated=False
    ):
        # convert to log2
        b1 = np.log2(startFrequency)
        b2 = np.log2(stopFrequency)
        # define log2 range
        logBandWidth = b2 - b1
        decibelsRange = -3 * logBandWidth
        # defining step by time resolution
        step = logBandWidth / self.nSamples
        nf = b1 - step  # new frequency
        amp = 1
        phase = 0
        amplitudeRange = self.decibelsToGain(decibelsRange) - amp
        ampStep = np.abs(amplitudeRange) / self.nSamples
        logSweep = np.zeros(shape=(self.nSamples,), dtype=np.float64)
        for i in range(1, self.nSamples):
            f = 2.0 ** nf
            phase = (phase + 2 * np.pi * f / self.sampleRate) % (2 * np.pi)
            logSweep[i] = amp * np.sin(phase)
            nf = nf + step
            if compensated:
                amp = np.power(2, ampStep)
                ampStep = ampStep - step / 2
        if any(
            [
                self.signalType == signal
                for signal in ["inv_log_sweep", "inv_pink_log_sweep"]
            ]
        ):
            # https://dsp.stackexchange.com/questions/41696/calculating-the-inverse-filter-for-the-exponential-sine-sweep-method
            k = np.exp(
                self.timeStamps * np.log(stopFrequency / startFrequency) / self.duration
            )
            logSweep = logSweep[::-1] / k
        return logSweep

    def generateLinearSweepTone(
        self, startFrequency=20.0, stopFrequency=20000.0, compensated=False
    ):
        bandWidth = stopFrequency - startFrequency
        frequencyStep = bandWidth / self.nSamples
        linSweep = np.zeros(shape=(self.nSamples,), dtype=np.float64)
        frequency = startFrequency
        phase = 0
        amp = 1
        for i in range(1, self.nSamples):
            phase = (phase + 2 * np.pi * frequency / self.sampleRate) % (2 * np.pi)
            linSweep[i] = amp * np.sin(phase)
            frequency = frequency + frequencyStep
            if compensated:
                amp = self.decibelsToGain(
                    -3 * (np.log2(frequency) - np.log2(startFrequency))
                )
        if any(
            [
                self.signalType == signal
                for signal in ["inv_lin_sweep", "inv_pink_lin_sweep"]
            ]
        ):
            # https://dsp.stackexchange.com/questions/41696/calculating-the-inverse-filter-for-the-exponential-sine-sweep-method
            k = self.timeStamps * np.log(stopFrequency / startFrequency) / self.duration
            linSweep = linSweep[::-1] / k
        return linSweep

    def decibelsToGain(self, decibels, dBType="dBFS"):
        if dBType == "dBu":
            factor = 0.775
        elif dBType == "dBFS" or dBType == "dBV":
            factor = 1.0
        else:
            raise ValueError(
                "Please define a consistent type of decibels (dBV, dBFS, dBu,...)"
            )
        return np.power(10.0, decibels / 20.0) * factor

    def gainToDecibels(self, gain, dBType="dBFS"):
        if dBType == "dBu":
            factor = 0.775
        elif dBType == "dBFS" or dBType == "dBV":
            factor = 1.0
        else:
            raise ValueError(
                "Please define a consistent type of decibels (dBV, dBFS, dBu,...)"
            )
        return 20.0 * np.log10(gain / factor)

    def generateInverseFilter(self):
        pass

    def generatePinkLogSweepTone(self):
        pass

    def binomial(self, n, k=2):
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


##############################################################################


def checkDirectory(directories):
    for directory in directories:
        if not directory.exists():
            directory.mkdir()


##############################################################################


def getLogger(
    loggerName="logger",
    level="INFO",
    loggerFormat="[%(asctime)s:%(filename)s:%(lineno)s] - %(levelname)s - %(message)s",
):
    logger = logging.getLogger(loggerName)
    if level == "INFO":
        logger.setLevel(logging.INFO)
    elif level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif level == "WARN":
        logger.setLevel(logging.WARN)
    elif level == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        raise ValueError(
            "Please define a valid logger level: INFO, DEBUG, WARN or ERROR"
        )

    formatter = logging.Formatter(loggerFormat)

    logger.handlers = []

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(level)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger


##############################################################################


def wavWrite(filePath, _sampleRate, audioData, bits=16):
    audioData = audioData * (2 ** (bits - 1))
    # TODO: fix 24bit files
    audioData = audioData.astype(np.int16)
    writeWavFile(filePath, _sampleRate, audioData)


##############################################################################


def sine(sample_rate: int, duration: float, amplitude: float, frequency: float = 440.0):
    N = np.arange(0, duration, 1.0 / sample_rate)
    return amplitude * np.sin(2.0 * np.pi * frequency * N)


##############################################################################


# def generateSineTestSignal(sampleRate=16000):
#     testSignal = np.zeros(shape=(1 * sampleRate,), dtype=np.float64)
#     startIndex = 0
#     frequency = 100
#     period = getPeriod(frequency)
#     # get nCycles in 0.25s for 1kHz-sinewave
#     duration, nSamples, _ = redefineDurationToPeriod(0.25, period, sampleRate)
#     testSignal[startIndex:nSamples] = sineGenerator(
#         0.5, frequency, sampleRate, duration
#     )
#     startIndex += nSamples
#     # generate signal and add to testSignal
#     duration, nSamples, _ = redefineDurationToPeriod(0.1, period, sampleRate)
#     testSignal[startIndex : startIndex + nSamples] = sineGenerator(
#         1.0, frequency, sampleRate, duration
#     )
#     startIndex += nSamples
#     # generate signal2
#     frequency = 50
#     period = getPeriod(frequency)
#     duration, nSamples, _ = redefineDurationToPeriod(0.15, period, sampleRate)
#     testSignal[startIndex : startIndex + nSamples] = sineGenerator(
#         1.0, frequency, sampleRate, duration
#     )
#     startIndex += nSamples
#     # generate signal3
#     duration, nSamples, _ = redefineDurationToPeriod(0.25, period, sampleRate)
#     testSignal[startIndex : startIndex + nSamples] = sineGenerator(
#         0.5, frequency, sampleRate, duration
#     )
#     startIndex += nSamples
#     # generate signal4
#     testSignal[startIndex : startIndex + nSamples] = sineGenerator(
#         2.0, frequency, sampleRate, duration
#     )
#     return testSignal, sampleRate


##############################################################################


def sweep_tone(
    sample_rate: int,
    duration: float,
    amplitude: float,
    f1: float = 20,
    f2: float = 20000,
    fade_duration: float = 0.0,
    inverse: bool = False,
)-> np.ndarray:
    """
    Generate a sweep tone signal with specific parameters.

    Args:
        sample_rate (int): Sample rate
        duration (float): Duration of the sweep
        amplitude (float): Amplitude of the sweep
        f1 (float): Start frequency. Defaults to 20Hz
        f2 (float, optional): End frequency. Defaults to 20k.
        fade_duration (float, optional): Fade in and fade out duration. Defaults to 0.0s.
        inverse: (bool, optional): Generate sweep inverse filter. False
    """
    from more_termcolor import cprint

    R = np.log(f2 / f1)
    t = np.arange(0, duration, 1.0 / sample_rate)
    output = np.sin((2.0 * np.pi * f1 * duration / R) * (np.exp(t * R / duration) - 1))
    if inverse:
        k = np.exp(t * R / duration)
        output = output[::-1] / k
    if fade_duration:
        if 2 * fade_duration < duration:
            fade_frequency = 1.0 / fade_duration / 4.0
            fade_window = sine(sample_rate, fade_duration, 1, frequency=fade_frequency)
            output[: len(fade_window)] = output[: len(fade_window)] * fade_window
            output[-len(fade_window) :] = (
                output[-len(fade_window) :] * fade_window[::-1]
            )
        else:
            cprint(
                "WARNING: fading was not applied. The fade duration is longer than the signal length.",
                "yellow",
            )
    return amplitude * output


##############################################################################


def mls(amplitude, N: int):
    """
    Generate Minimum Length Sequence signal for a N-bit

    Args:
        N (int): number of bits

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    nSamples = int(np.power(2, N))
    taps = 4
    tap1 = 1
    tap2 = 2
    tap3 = 4
    tap4 = 15
    if N != 16:
        raise ValueError("Sorry but for now MLS signal is only defined for 16 bits.")
    buffer = [1] * N
    samples = np.zeros((nSamples,), dtype=np.float64)
    for i in np.arange(nSamples - 1, 0, -1):
        # feedback bit
        xorbit = buffer[tap1] ^ buffer[tap2]
        # second logic level
        if taps == 4:
            xorbit2 = buffer[tap3] ^ buffer[tap4]
            xorbit = xorbit ^ xorbit2
        for j in np.arange(N - 1, 0, -1):
            temp = buffer[j - 1]
            buffer[j] = temp
        buffer[0] = xorbit
        samples[i] = (-2 * xorbit) + 1.0
    return amplitude * samples


##############################################################################


if __name__ == "__main__":

    # prepare and define params
    logger = getLogger(level="DEBUG")
    amplitude = 1
    duration = 10
    sample_rate = 44100
    signalType = "lin_ramp"
    fadeIn = 0.0
    fadeOut = 0.0
    checkDirectory([audioDataDir, plotDir])

    # synthesize signal
    signalGenerator = SignalGenerator(
        signalType,
        duration,
        amplitude=amplitude,
        sampleRate=sample_rate,
        nHarmonics=20,
        nCycles=1,
        cycleAttenuation=3,
        logger=logger,
        fadeIn=fadeIn,
        fadeOut=fadeOut,
    )
    testSignal = signalGenerator()

    # display output signal
    plt.plot(testSignal)
    plt.title("Output Signal")
    plt.xlim([0, len(testSignal)])
    plt.xlabel("Sample [n]")
    plt.ylabel("Amplitude [FS]")
    plt.tight_layout()
    plt.show()
    plt.savefig(plotDir / "plot")

    # write output signal
    output_path = audioDataDir / f"{signalType}.wav"
    print(colored(f"Writing generated signal in {output_path}", "green"))
    wavWrite(output_path, sample_rate, testSignal)


##############################################################################

# TODO: add argparser to map parameters
# TODO: fix issues with Pink noise and other signal generators with failures
# TODO - create unittests to check functions
# TODO - add biquad filter class to generate noise filtered versions (HPF, LPF, BPF (3rd/oct), APF, Parametric)
