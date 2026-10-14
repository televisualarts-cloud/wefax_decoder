# WEFAX Decoder

An HF weather-fax (WEFAX, ITU-R F.343 / WMO) receiver and decoder for
Northwood (GYA), DWD and other international stations.

It listens to a sound-card input, finds each transmission automatically and
decodes it, correcting tuning offset, slant and tears. It then cleans and
sharpens the chart, names it from the broadcast schedules, saves it to disk
and shows it in a browser page. With SDR Console's CAT port it can also
retune the radio through the day to collect the most important charts.

## Features

- Automatic detection of start tone, phasing and stop tone (IOC 576 and 288;
  60, 90, 120 and 240 LPM).
- AFC with drift tracking, a correlation FM discriminator that is robust to
  fades and QRN, and slant correction from the phasing lines.
- A **flywheel** that uses the PC clock to detect and fill audio lost by the
  sound card, plus image-based repair of torn charts.
- Enhancement that removes noise streaks and rebuilds broken chart lines
  without distorting them, with an FSRCNN x2 upscale. Clean charts get only
  light processing.
- Chart identification from the OpenCPN worldwide schedule file, giving file
  names such as `20261011_1800Z_DWD_surface-pressure-analysis.png`.
- SDR Console control over a Kenwood TS-2000 CAT port: frequency read-back,
  tuning, and an **auto-scheduler**. You choose the 1st and 2nd priority
  stations, DX stations fill the gaps, and it learns which frequencies work
  at each time of day.
- A browser page with the following:
  - a live view of the chart as it arrives (green while receiving, orange
    "NO SIGNAL" when a chart is due but not heard, red when idle);
  - a thumbnail strip for each day;
  - Raw and Enhanced views, mouse-wheel zoom, and rotation.

## Install

Python 3.9 or later.

```
pip install -r requirements.txt
```

`onnxruntime` (for the neural upscaler) and `pyserial` (for CAT) are
optional. Without them the decoder uses a Lanczos upscale and cannot do CAT.

## Quick start

1. Tune the receiver in **USB with the dial 1.9 kHz below the published
   frequency**. For Northwood (GYA) that means:

   | Published   | Dial (USB)  |
   |-------------|-------------|
   | 2618.5 kHz  | 2616.6 kHz  |
   | 4610 kHz    | 4608.1 kHz  |
   | 8040 kHz    | 8038.1 kHz  |
   | 11086.5 kHz | 11084.6 kHz |

2. Start the decoder with `python wefax_decoder.py`. The page opens at
   <http://127.0.0.1:8730/>.

3. Choose the sound-card input carrying the receiver audio in the
   **Audio in** box at the top of the page. This could be a virtual cable,
   a USB sound card or a line input. The border turns green when it is
   running, and the IN meter shows the level. The choice is saved.

Each fax appears when its transmission ends.

On Windows, `start_wefax.bat` starts the decoder with CAT on COM4 and the
auto-scheduler on. Edit it to suit your set-up.

**Defaults you may want to change** (at the end of `wefax_decoder.py`, or on
the command line):

- `--device` is the audio input used until you choose one on the page. The
  default is `Line 8 (Virtual Audio Cable) MME`; if that isn't present, the
  system default input is used. `--list-devices` lists the inputs.
- `--qth lat,lon` is your location for the propagation model. The default is Plymouth, UK.

To decode a recording instead of live audio, run
`python wefax_decoder.py --wav recording.wav`.

## Saved files

Everything is written under `./faxes/` (change this with `--out`):

| Path | Contents |
|---|---|
| `raw/<name>.png` | Decoded chart: geometry corrected, no cleaning |
| `enhanced/<name>.png` | Cleaned chart at 2x resolution |
| `thumbs/<name>.jpg` | Thumbnail for the page |
| `stations/<station>/<chart>.png` | The latest copy of each chart, e.g. `stations/Northwood/Surface-Analysis.png`. A newer issue replaces it; a repeat of the same issue only replaces it if it decoded better. The `.json` beside it gives its quality and time |
| `<name>.json` | Decode details: station and chart, IOC, LPM, AFC, clock ppm, quality, tears, etc. |
| `proplog.jsonl` | The auto-scheduler's log of every attempt (SNR, quality) |
| `settings.json` | Your audio input, CAT port and 1st/2nd priority stations |

Audio is not kept unless you add `--save-audio`.

## SDR Console CAT

The decoder reads the tuned frequency from SDR Console, and can set it, over
a Kenwood TS-2000 CAT port:

1. Create a virtual COM port pair with com0com, VSP Manager or Eltima, for
   example COM3 and COM4.
2. In SDR Console, go to Home > Serial Port (CAT) and add one end, such as
   COM3, at 57600 baud.
3. Choose the other end of the pair, for example COM4, in the **CAT port**
   box at the top of the page. The border turns green when SDR Console
   answers, and the choice is saved. You can also give the port at start-up
   with `--cat COM4`; a port chosen on the page takes precedence. Add
   `--cat-rx 2` to follow receiver 2.

`python tools/comtest.py` checks the COM ports if CAT does not respond.
Without CAT, tell the decoder which station you are on with
`--station DWD`, or pick it in the station box on the page.

The schedules come from the OpenCPN weatherfax plugin's worldwide
`WeatherFaxSchedules.xml` in `wefax/data/`, with the Northwood list
corrected. To update them, put a newer copy of the file next to
`wefax_decoder.py`.

## Auto-scheduler

The auto-scheduler needs CAT. Press **AUTO** on the page, or start with
`--cat COM4 --auto`.

- **Priority stations:** choose them with **1st priority** and **2nd priority**
  in the top bar. The default is Northwood then DWD, and the choice is saved.
  Their surface analyses, short-range forecasts and gale warnings are planned
  first, then wave and other charts.
- **Clashes:** the 1st priority station wins over the 2nd.
- **Repeats:** a chart broadcast several times is only taken until one good
  copy has been decoded. If the first try at a priority station's chart was
  poor or not decoded, its next repeat goes ahead of every other chart,
  priority or DX.
- **Gaps:** gaps long enough for a whole chart, plus 75 s before and 45 s
  after, are filled with DX stations.
- **Charts in progress:** once a chart has started, nothing else is planned
  over it, and the scheduler never retunes while a chart is being received,
  even if it runs longer than the schedule says. The only exception is a
  recording that has run over 25 minutes, longer than any single chart, which
  means a stop tone was missed; it is ended so later charts aren't lost.
  If the scheduler does have to leave a chart early, that attempt isn't counted
  against the frequency.
- **Frequency choice:** each slot gets the frequency that has worked best at
  that time of day. With no history yet, it uses a simple ionospheric
  estimate based on path length and sun height at the path mid-point.
- **Learning:** DX combinations of station, frequency and time of day that
  keep failing are retried after 1, 2, 4, 8 and 16 days, then every 28 days,
  so improving propagation is noticed. Old results fade out gradually, so
  seasonal changes are followed.
- **Manual tuning:** using Tune or a chart chip pauses the scheduler for
  20 minutes.
- **No DX:** use `--no-dx` to stop the gap filling.

## The page

- **Live panel** (top left):
  - green while a chart is being received;
  - orange **NO SIGNAL**, with the incoming noise, when a chart is due on the
    tuned frequency but nothing usable is heard;
  - red **IDLE** otherwise.
- **Thumbnail strip:** one UTC day at a time, newest at the top. Use **−1**
  and **+1** to change day.
- **Viewer:**
  - **Raw** and **Enhanced** show the same part of the chart at the same size.
  - The **mouse wheel** zooms and **dragging** pans.
  - **Fit** and **1:1** reset the zoom.
  - **Rotate −90° / +90°** turns the chart. Rotation is remembered for each
    fax in that browser.

## Useful options

| Option | Effect |
|---|---|
| `--no-ml` | Use Lanczos instead of the FSRCNN neural upscaler |
| `--scale 1` | Save the enhanced image at the native 1810 px width |
| `--mode chart`, `--mode greyscale` | Force the finishing mode (`auto` picks one per fax) |
| `--ioc 288`, `--lpm 60` | Override the detected values |
| `--save-audio` | Also keep each transmission as a WAV file |
| `--host 0.0.0.0` | Allow viewing from other devices on your LAN |
| `--port 8730` | Web page port |

`python tools/reenhance.py` re-runs the enhancement on every saved fax with
the current settings. It keeps each previous enhanced image as `*_old.png`.

## How it works

**Detection.** A cheap causal demodulator watches for the start tone (300 Hz
for IOC 576, 675 Hz for IOC 288), the phasing pattern, the 450 Hz stop tone,
and loss of signal. If the stop tone is missed on a back-to-back broadcast,
the next chart's phasing still closes the current chart and starts a new one.
A band-power test keeps a weak or fading fax counted as
present. The whole transmission, with 12 s of pre-roll, is then decoded
offline.

**AFC.** A two-tooth comb is fitted to the 1500 and 2300 Hz lines of the
audio spectrum, then re-fitted in 20 s windows to track drift. Noise is
spectrally flat, so the estimate is not pulled toward the centre at low SNR.

**Demodulation.** Zero-phase I/Q filtering is followed by a correlation
discriminator that averages z[n]·z*[n−1] before taking the angle. Samples in
fades and crashes therefore carry little weight. Clicks, fades and QRN
crashes are flagged and later filled from the lines above and below.

**Line timing and slant.** The phasing pulse is tracked line by line. A
robust fit gives the LPM, the exact line period and the line start. The
period is refined by maximising vertical-edge sharpness, and a smooth
residual follows sound-card clock drift. Leftover phasing lines at the top
and stop-tone lines at the bottom are trimmed. If there is no stop tone,
trailing noise is cut off where the fax tones disappear.

**Tears and the flywheel.** Lost audio samples shift every later line
sideways.
- **Flywheel:** each sound-card buffer is time-stamped by the PC clock as it
  arrives. Buffers can be late but never early, so the lowest arrival times
  trace the true clock. A lasting step in that trace shows how much audio was
  lost and where, and the gap is filled with silence before decoding.
- **Image check:** any remaining jump is found by comparing blocks of lines.
  It is only corrected if the shift persists further down the chart.

**Enhancement.**
1. Single-line noise streaks are removed. A streak is a deviation from the
   lines above and below that is coherent along the line.
2. Clean charts then just get levels and the upscale.
3. On noisy charts, 24 oriented line detectors rebuild broken isobars and
   fronts. A mark only counts as a line if it responds along its direction
   and not across it, so text and noise are left alone.
4. The background gets non-local-means denoising, and the result is upscaled
   2x with FSRCNN.

## Testing without a radio

```
python tools/make_test_signal.py test.wav --snr 12 --ppm 150 --offset 40
python wefax_decoder.py --wav test.wav
```

This synthesises a chart with a tuning offset, drift, clock error, noise, QRN
crashes and fades, and also writes the true image to `test_truth.npy`.

## Project layout

```
wefax_decoder.py          main program: capture, decode, save, web server
start_wefax.bat           Windows launcher
wefax/
  frontend.py             live capture, tone detection, flywheel time-stamps
  dsp.py                  filters, demodulators, AFC comb, tone features
  decode.py               offline decoder: AFC, phasing, slant, gaps, edges
  realign.py              tear detection and repair
  enhance.py              streak removal, line rebuilding, upscaling
  preview.py              live preview image
  schedule.py             broadcast schedules and chart identification
  cat.py                  Kenwood TS-2000 CAT for SDR Console
  autosched.py            auto-scheduler, propagation model, learning log
  server.py               web server and live updates
  web/index.html          the page
  data/                   OpenCPN WeatherFaxSchedules.xml (GPL-3.0, data)
  models/                 FSRCNN x2 ONNX weights (Apache-2.0)
tools/
  make_test_signal.py     synthetic test transmissions
  reenhance.py            re-run enhancement on saved faxes
  comtest.py              COM port / CAT diagnostic
```

## Licence

The code in this project is released under the MIT License (see `LICENSE`).

Two bundled third-party files keep their own licences:

- `wefax/data/WeatherFaxSchedules.xml` is from the OpenCPN weatherfax plugin
  ([rgleason/weatherfax_pi](https://github.com/rgleason/weatherfax_pi)) and
  stays under GPL-3.0 (see `wefax/data/WeatherFaxSchedules_LICENSE.txt`). It
  is a data file read at run time, not part of the code. If you redistribute
  it, the GPL-3.0 terms apply to that file. You can also leave it out and
  supply your own copy next to `wefax_decoder.py`.
- `wefax/models/fsrcnn_x2.onnx` contains the FSRCNN x2 weights from
  [Saafke/FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow),
  Apache-2.0 (see `wefax/models/FSRCNN_LICENSE.txt`). They were converted to
  ONNX and checked against the original graph.
