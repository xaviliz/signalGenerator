# signalGenerator

Generates different test signals such as noises, waveforms and sweep tones:
* sine
* triangle
* sawtooth
* square
* white_noise
* MLS
* delta
* pulse
* level
* lin_ramp
* log_ramp
* log_sweep
* inverse_log_filter
* lin_sweep
* inverse_lin_filter

It also generates train of any kind of signal.

## Installation

For the latest releases install from the GitHub repo
```bash
pip install git+https://github.com/xaviliz/signalGenerator
```

## Usage

#### Generate a sine wave

```python
import signal_generator as sg

signal_generator = sg.SignalGenerator("sine", 20)
sine_20_seg = signal_generator()
```

#### Generate a train of sine waves

```python
signal_generator = sg.SignalGenerator("sine", 1, nCycles=10)
sine_train = signal_generator()
```

#### Generate a linear ramp

```python
amplitude = 1
duration = 10
sample_rate = 44100
signal_type = "lin_ramp"
fade_in = 0.0
fade_out = 0.0
logger = sg.getLogger(level="DEBUG")

# synthesize signal
signal_generator = sg.SignalGenerator(
    signal_type,
    duration,
    amplitude=amplitude,
    sampleRate=sample_rate,
    logger=logger,
    fadeIn=fade_in,
    fadeOut=fade_out,
)
testSignal = signalGenerator()
```

## Dependencies

* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Matplotlib](https://matplotlib.org/)

## Citation

```text
@misc{xlizarraga2022signal-generator,
  author = {Lizarraga, Xavier},
  title = {signal-generator},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/xaviliz/signalGenerator}},
  commit = {623ded5e08eed84b18131d0642068534e4c80154}
}
```
