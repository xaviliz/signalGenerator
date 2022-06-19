# signalGenerator

## Description
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

## Usage

```python
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
```
