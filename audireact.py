# audireact_v5_sampleaccurate.py
"""
Sample-accurate auditory reaction-time experiment.

Single sounddevice OutputStream plays continuous noise and mixes
precomputed stereo tone buffers on demand. The callback provides
a device DAC timestamp which we map to time.perf_counter() so
we obtain an *actual* onset timestamp for each tone.

CSV includes planned scheduled timestamp (UTC), actual onset (UTC),
RT in seconds, and hit/miss/false_positive status.
"""
import os
import time
import datetime
import random
import csv
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from psychopy import core, event, visual, gui

# ----------------- Participant Info -----------------
info = {"Participant ID": ""}
dlg = gui.DlgFromDict(info, title="Participant Info")
if not dlg.OK:
    core.quit()
participant_id = info["Participant ID"].strip() or "test"

# ----------------- Parameters -----------------
SR = 44100
TOTAL_NOISE_DURATION = 900.0   # seconds
TONE_DURATION = 0.2            # seconds
TONE_FREQ = 1000.0             # Hz
FIXED_VOLUME = 0.10
INTERVAL_OPTIONS = [2.0, 4.0, 5.0, 7.0]
RANDOM_SEED = 12345
BLOCKSIZE = 256                # callback block size
FB_DUR = 0.2                   # visual feedback duration
MAX_RT = 1.5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------- Balanced schedule (equal L/R and balanced intervals) -----------------
def compute_balanced_schedule(total_noise_duration=TOTAL_NOISE_DURATION,
                              tone_duration=TONE_DURATION,
                              interval_options=INTERVAL_OPTIONS):
    sides = ["left", "right"]
    # Build cycles of (side, interval) so each side gets every interval once per cycle
    cycle_pairs = [(s, it) for s in sides for it in interval_options]

    side_seq = []
    interval_seq = []
    cum = 0.0

    while True:
        random.shuffle(cycle_pairs)
        progressed = False
        for side, inter in cycle_pairs:
            projected = cum + inter + tone_duration
            if projected > total_noise_duration:
                # cannot add more
                return side_seq, interval_seq
            side_seq.append(side)
            interval_seq.append(inter)
            cum += inter
            progressed = True
        if not progressed:
            return side_seq, interval_seq

# compute schedule
combo_sides, combo_intervals = compute_balanced_schedule()
n_trials = len(combo_sides)
print(f"Planned trials: {n_trials}")

# scheduled_times relative to start (seconds)
scheduled_times = []
cum = 0.0
for inter in combo_intervals:
    cum += inter
    scheduled_times.append(cum)

# ----------------- Precompute tone buffers (stereo) -----------------
t = np.arange(int(round(TONE_DURATION * SR))) / SR
tone_mono = np.sin(2 * np.pi * TONE_FREQ * t) * FIXED_VOLUME
# make stereo float32 arrays
tone_left = np.column_stack((tone_mono.astype("float32"), np.zeros_like(tone_mono, dtype="float32")))
tone_right = np.column_stack((np.zeros_like(tone_mono, dtype="float32"), tone_mono.astype("float32")))

# ----------------- Noise bed (stereo) -----------------
noise_amp = 0.02
noise_samples = np.random.uniform(-1.0, 1.0, int(SR * TOTAL_NOISE_DURATION)).astype("float32") * noise_amp
fade_len = int(0.05 * SR)
if fade_len > 0:
    fade = np.linspace(0.0, 1.0, fade_len).astype("float32")
    noise_samples[:fade_len] *= fade
    noise_samples[-fade_len:] *= fade[::-1]
noise_stereo = np.column_stack((noise_samples, noise_samples)).astype("float32")
noise_len = noise_stereo.shape[0]

# ----------------- Shared state for callback <-> main thread -----------------
state_lock = threading.Lock()

# streaming state
noise_cursor = 0  # position in noise buffer (samples)
device_start_dac = None
device_start_perf = None

# tone playback slot (None when idle); will hold numpy array stereo
tone_slot = None
tone_cursor = 0

# for reporting actual onset back to main thread
last_onset_perf = None
last_onset_event = threading.Event()  # set by callback when onset computed

audio_started_flag = False

# ----------------- Mapping helpers (device DAC <-> perf) -----------------
def perf_to_ntp_str(perf_ts, perf_anchor_perf, perf_anchor_ntp):
    """Convert a perf_counter timestamp to UTC string using anchor pair."""
    # perf_anchor_perf -> perf time when we captured perf_anchor_ntp (unix)
    offset_sec = perf_ts - perf_anchor_perf
    unix_ts = perf_anchor_ntp + offset_sec
    return datetime.datetime.utcfromtimestamp(unix_ts).strftime("%Y-%m-%d %H:%M:%S.%f")

# We'll capture NTP offset once at start to provide UTC strings (optional)
def sync_clock_offset():
    try:
        import ntplib
        c = ntplib.NTPClient()
        r = c.request("pool.ntp.org", version=3, timeout=3)
        offset = r.tx_time - time.time()
        return offset
    except Exception:
        return 0.0

clock_offset = sync_clock_offset()
PERF0 = time.perf_counter()
NTP0 = time.time() + clock_offset

def perf_to_ntpstr(perf_ts):
    unix_ts = NTP0 + (perf_ts - PERF0)
    return datetime.datetime.utcfromtimestamp(unix_ts).strftime("%Y-%m-%d %H:%M:%S.%f")

# ----------------- Audio callback (single stream: noise + optional tone mix) -----------------
def audio_callback(outdata, frames, time_info, status):
    """
    outdata: (frames, channels)
    time_info contains 'outputBufferDacTime' (device clock seconds for first sample)
    """
    global noise_cursor, device_start_dac, device_start_perf
    global tone_slot, tone_cursor, last_onset_perf, audio_started_flag

    # initialize device/perf anchor at first callback
    dac_ts = None
    try:
        if isinstance(time_info, dict):
            dac_ts = time_info.get("outputBufferDacTime") or time_info.get("output_buffer_dac_time")
        else:
            dac_ts = getattr(time_info, "outputBufferDacTime", None)
    except Exception:
        dac_ts = None

    if dac_ts is None:
        # if time_info did not supply DAC time, still proceed but we lose sample-accurate mapping
        # fill with zeros to be safe
        outdata[:] = np.zeros((frames, 2), dtype="float32")
        return

    if device_start_dac is None:
        device_start_dac = float(dac_ts)
        device_start_perf = time.perf_counter()
        audio_started_flag = True

    # write noise chunk (looping)
    end = noise_cursor + frames
    if end <= noise_len:
        out_chunk = noise_stereo[noise_cursor:end].copy()
        noise_cursor = end
        if noise_cursor >= noise_len:
            noise_cursor = noise_cursor % noise_len
    else:
        # wrap-around
        part1 = noise_stereo[noise_cursor:noise_len]
        part2 = noise_stereo[0:end - noise_len]
        out_chunk = np.vstack((part1, part2)).copy()
        noise_cursor = end - noise_len

    # mix in tone if present
    with state_lock:
        local_tone = tone_slot
        local_tone_cursor = tone_cursor

    if local_tone is not None:
        # how many tone samples available
        tone_remaining = local_tone.shape[0] - local_tone_cursor
        to_write = min(frames, tone_remaining)
        # add tone fragment into out_chunk
        out_chunk[:to_write] += local_tone[local_tone_cursor:local_tone_cursor + to_write]
        # update shared cursor and possibly clear tone
        with state_lock:
            tone_cursor += to_write
            # compute onset perf at the first block where tone_cursor was zero (i.e., local_tone_cursor == 0)
            if local_tone_cursor == 0:
                # device DAC time corresponds to the first sample of out_chunk (this callback)
                # compute perf timestamp for first sample in out_chunk:
                onset_perf = device_start_perf + (float(dac_ts) - float(device_start_dac))
                last_onset_perf = onset_perf
                last_onset_event.set()
            if tone_cursor >= local_tone.shape[0]:
                tone_slot = None
                tone_cursor = 0

    # guard against clipping
    np.clip(out_chunk, -1.0, 1.0, out=out_chunk)
    outdata[:] = out_chunk

# ----------------- Set up stream -----------------
stream = sd.OutputStream(samplerate=SR, channels=2, callback=audio_callback, blocksize=BLOCKSIZE, dtype="float32")

# ----------------- PsychoPy Setup -----------------
win = visual.Window(fullscr=False, color="black", units="pix", allowGUI=False)
fixation = visual.TextStim(win, text="+", color="white", height=50)
instr = visual.TextStim(
    win,
    text="Press SPACE as quickly as possible when you hear a tone.\n\nPress ENTER to start.",
    color="white",
    height=30,
    wrapWidth=800,
    pos=(0, 0),
    alignText="center",
)
instr.draw()
win.flip()

# wait for ENTER
while True:
    keys = event.getKeys()
    if "return" in keys or "enter" in keys:
        break
    elif "escape" in keys:
        core.quit()
    core.wait(0.01)

fixation.draw()
win.flip()

# ----------------- Results & CSV -----------------
results = []
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, f"audireact_{participant_id}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv")

# ----------------- Start audio and run trials -----------------
try:
    stream.start()
    # wait for audio to actually start and provide a device/perf anchor
    t0_wait = time.perf_counter()
    while not audio_started_flag and (time.perf_counter() - t0_wait) < 2.0:
        time.sleep(0.001)

    if not audio_started_flag:
        # fallback: set anchors so mapping functions don't break (less accurate)
        device_start_perf = time.perf_counter()
        device_start_dac = 0.0
        print("WARNING: device DAC timestamp not available; timing less accurate.")
    else:
        print("Audio callback supplied device/perf anchor.")

    start_perf = device_start_perf  # anchor for scheduling planned offsets

    print("Playback started — running trials.")

    for i, scheduled_offset in enumerate(scheduled_times):
        trial_number = i + 1
        side = combo_sides[i]
        planned_perf = start_perf + scheduled_offset

        # WAIT before scheduled onset: allow false positives
        while True:
            now = time.perf_counter()
            if now >= planned_perf:
                break
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                ts = time.perf_counter()
                ts_ntp = perf_to_ntpstr(ts)
                results.append([trial_number, "none", "", ts_ntp, "", "", "false_positive"])
                fixation.height = 70
                fixation.draw()
                win.flip()
                core.wait(FB_DUR)
                fixation.height = 50
                fixation.draw()
                win.flip()
                event.clearEvents()
            # micro-sleep to keep CPU reasonable
            time.sleep(0.0008)

        # At planned time: schedule tone by placing it into tone_slot
        with state_lock:
            if side == "left":
                tone_slot = tone_left.copy()
            else:
                tone_slot = tone_right.copy()
            tone_cursor = 0
            last_onset_event.clear()
            last_onset_perf = None

        # busy-wait *only* until callback sets last_onset_perf (should happen imminently)
        # However, to ensure we don't deadlock, we wait up to a small timeout
        onset_timeout = 1.0  # seconds
        onset_happened = last_onset_event.wait(timeout=onset_timeout)
        if not onset_happened:
            # very unusual: callback didn't report onset in time
            actual_onset_perf = None
            print(f"Warning: no onset reported for trial {trial_number}")
        else:
            with state_lock:
                actual_onset_perf = last_onset_perf

        scheduled_ts_ntp = perf_to_ntpstr(planned_perf)
        actual_onset_ts_ntp = perf_to_ntpstr(actual_onset_perf) if actual_onset_perf is not None else ""

        # collect RT within MAX_RT seconds after actual_onset_perf (if available), else use planned
        rt = None
        response_perf = None
        if actual_onset_perf is not None:
            rt_deadline = actual_onset_perf + MAX_RT
            while time.perf_counter() < rt_deadline:
                keys = event.getKeys()
                if "escape" in keys:
                    raise KeyboardInterrupt
                if "space" in keys:
                    response_perf = time.perf_counter()
                    rt = response_perf - actual_onset_perf
                    # feedback
                    fixation.height = 70
                    fixation.draw()
                    win.flip()
                    core.wait(FB_DUR)
                    fixation.height = 50
                    fixation.draw()
                    win.flip()
                    break
                time.sleep(0.0008)
        else:
            # fallback: use planned perf for RT window
            rt_deadline = planned_perf + MAX_RT
            while time.perf_counter() < rt_deadline:
                keys = event.getKeys()
                if "escape" in keys:
                    raise KeyboardInterrupt
                if "space" in keys:
                    response_perf = time.perf_counter()
                    rt = response_perf - planned_perf
                    fixation.height = 70
                    fixation.draw()
                    win.flip()
                    core.wait(FB_DUR)
                    fixation.height = 50
                    fixation.draw()
                    win.flip()
                    break
                time.sleep(0.0008)

        if rt is None:
            resp_status = "miss"
            response_ts_ntp = ""
        else:
            resp_status = "hit"
            response_ts_ntp = perf_to_ntpstr(response_perf)

        # save result row: trial, side, scheduled_ts, actual_onset_ts, response_ts, RT, resp_status
        results.append([
            trial_number,
            side,
            scheduled_ts_ntp,
            actual_onset_ts_ntp,
            response_ts_ntp,
            "" if rt is None else f"{rt:.6f}",
            resp_status
        ])

        print(f"Trial {trial_number}/{n_trials} | Side:{side} | {resp_status} | RT={'' if rt is None else f'{rt:.4f}s'}")

    print("All trials complete. Allowing trailing audio to play briefly.")
    time.sleep(0.2)

except KeyboardInterrupt:
    print("Experiment terminated early by user (ESC).")

finally:
    try:
        stream.stop()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass

    # Save results
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial",
            "side",
            "scheduled_timestamp_utc",
            "actual_onset_timestamp_utc",
            "response_timestamp_utc",
            "RT_seconds",
            "resp_status"
        ])
        w.writerows(results)

    print(f"Data saved to: {csv_path}")
    win.close()
    core.quit()
