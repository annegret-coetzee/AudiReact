import os
import time
import datetime
import random
import csv
import threading
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
RAMP_DUR = 0.005  # 5 ms
RAMP_SAMPLES = int(RAMP_DUR * SR)
NOISE_ATTEN = 0.3


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

# ----------------- Stable NTP-anchored timestamping -----------------

def sync_clock_offset():
    """
    One-shot NTP offset estimation.
    Used ONLY to establish a UTC anchor at experiment start.
    """
    try:
        import ntplib
        c = ntplib.NTPClient()
        r = c.request("pool.ntp.org", version=3, timeout=3)
        return r.tx_time - time.time()
    except Exception:
        return 0.0


# Capture a SINGLE anchor pair at experiment start
NTP_OFFSET = sync_clock_offset()

PERF_ANCHOR = time.perf_counter()
UTC_ANCHOR = time.time() + NTP_OFFSET


def perf_to_utc_timestamp(perf_ts):
    """
    Convert a perf_counter timestamp to a UTC datetime string
    using a fixed anchor. Immune to system clock jumps.
    """
    unix_ts = UTC_ANCHOR + (perf_ts - PERF_ANCHOR)
    return datetime.datetime.utcfromtimestamp(unix_ts).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )


# ----------------- Audio callback (single stream: noise + optional tone mix) -----------------
def audio_callback(outdata, frames, time_info, status):
    global noise_cursor, device_start_dac, device_start_perf
    global tone_slot, tone_cursor, last_onset_perf, audio_started_flag

    # --- DAC timestamp handling ---
    dac_ts = None
    try:
        if isinstance(time_info, dict):
            dac_ts = time_info.get("outputBufferDacTime")
        else:
            dac_ts = getattr(time_info, "outputBufferDacTime", None)
    except Exception:
        pass

    if dac_ts is None:
        outdata[:] = 0
        return

    if device_start_dac is None:
        device_start_dac = float(dac_ts)
        device_start_perf = time.perf_counter()
        audio_started_flag = True

    # --- noise bed ---
    end = noise_cursor + frames
    if end <= noise_len:
        out_chunk = noise_stereo[noise_cursor:end].copy()
        noise_cursor = end % noise_len
    else:
        part1 = noise_stereo[noise_cursor:]
        part2 = noise_stereo[:end - noise_len]
        out_chunk = np.vstack((part1, part2)).copy()
        noise_cursor = end - noise_len

    # --- copy shared tone state ---
    with state_lock:
        local_tone = tone_slot
        local_tone_cursor = tone_cursor

    # --- tone + cosine-ramped noise dip ---
    if local_tone is not None:
        tone_remaining = local_tone.shape[0] - local_tone_cursor
        to_write = min(frames, tone_remaining)

        idx = np.arange(local_tone_cursor, local_tone_cursor + to_write)
        atten = np.full(to_write, NOISE_ATTEN, dtype=np.float32)

        onset = idx < RAMP_SAMPLES
        atten[onset] = 1 - (1 - NOISE_ATTEN) * 0.5 * (
            1 - np.cos(np.pi * idx[onset] / RAMP_SAMPLES)
        )

        offset = idx > (local_tone.shape[0] - RAMP_SAMPLES)
        tail = idx[offset] - (local_tone.shape[0] - RAMP_SAMPLES)
        atten[offset] = 1 - (1 - NOISE_ATTEN) * 0.5 * (
            1 + np.cos(np.pi * tail / RAMP_SAMPLES)
        )

        out_chunk[:to_write] *= atten[:, None]
        out_chunk[:to_write] += local_tone[local_tone_cursor:local_tone_cursor + to_write]

        with state_lock:
            tone_cursor += to_write
            if local_tone_cursor == 0:
                onset_perf = device_start_perf + (float(dac_ts) - float(device_start_dac))
                last_onset_perf = onset_perf
                last_onset_event.set()
            if tone_cursor >= local_tone.shape[0]:
                tone_slot = None
                tone_cursor = 0

    # --- safety ---
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

# Save anchors for audit/debug
anchor_info = {
    "perf_anchor": PERF_ANCHOR,
    "utc_anchor": UTC_ANCHOR,
    "ntp_offset": NTP_OFFSET
}

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
        print("WARNING: device DAC timestamp not available; timing less accurate.")
    else:
        print("Audio callback supplied device/perf anchor.")

    start_perf = device_start_perf  # anchor for scheduling planned offsets
    print("Playback started — running trials.")

    for i, scheduled_offset in enumerate(scheduled_times):
        trial_number = i + 1
        side = combo_sides[i]
        planned_perf = start_perf + scheduled_offset
        scheduled_offset_s = scheduled_offset  # already relative to start_perf

        # WAIT until planned onset
        while time.perf_counter() < planned_perf:
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                ts = time.perf_counter()
                ts_ntp = perf_to_utc_timestamp(ts)
                results.append([trial_number, "none", "", ts_ntp, "", "", "false_positive", None, None, None])
                fixation.height = 70
                fixation.draw()
                win.flip()
                core.wait(FB_DUR)
                fixation.height = 50
                fixation.draw()
                win.flip()
                event.clearEvents()
            time.sleep(0.0008)

        # schedule tone
        with state_lock:
            tone_slot = tone_left.copy() if side == "left" else tone_right.copy()
            tone_cursor = 0
            last_onset_event.clear()
            last_onset_perf = None

        # wait for onset callback
        onset_timeout = 1.0
        if last_onset_event.wait(timeout=onset_timeout):
            with state_lock:
                actual_onset_perf = last_onset_perf
            actual_onset_offset_s = actual_onset_perf - start_perf
        else:
            actual_onset_perf = None
            actual_onset_offset_s = None

        scheduled_ts_utc = perf_to_utc_timestamp(planned_perf)
        actual_onset_ts_utc = perf_to_utc_timestamp(actual_onset_perf) if actual_onset_perf else ""

        # --- Collect response & RT ---
        response_perf = None
        rt = None
        rt_anchor = actual_onset_perf if actual_onset_perf is not None else planned_perf
        rt_deadline = rt_anchor + MAX_RT

        while time.perf_counter() < rt_deadline:
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                response_perf = time.perf_counter()
                rt = response_perf - rt_anchor
                fixation.height = 70
                fixation.draw()
                win.flip()
                core.wait(FB_DUR)
                fixation.height = 50
                fixation.draw()
                win.flip()
                break
            time.sleep(0.0008)

        if response_perf is not None:
            response_offset_s = response_perf - start_perf
            resp_status = "hit"
            response_ts_utc = perf_to_utc_timestamp(response_perf)
        else:
            response_offset_s = None
            resp_status = "miss"
            response_ts_utc = ""

        results.append([
            trial_number,
            side,
            scheduled_ts_utc,
            actual_onset_ts_utc,
            response_ts_utc,
            "" if rt is None else f"{rt:.6f}",
            resp_status,
            scheduled_offset_s,
            actual_onset_offset_s,
            response_offset_s
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

    # ----------------- Save CSV -----------------
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial",
            "side",
            "scheduled_timestamp_utc",
            "actual_onset_timestamp_utc",
            "response_timestamp_utc",
            "RT_seconds",
            "resp_status",
            "scheduled_offset_s",
            "actual_onset_offset_s",
            "response_offset_s"
        ])
        w.writerows(results)

    print(f"Data saved to: {csv_path}")
    win.close()
    core.quit()
