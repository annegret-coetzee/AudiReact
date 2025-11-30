# audireact_v3_fixed.py
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import datetime
from psychopy import core, event, visual, gui
from audigen import ensure_sound_file
import os
import random
import csv

# ----------------- Participant Info -----------------
info = {"Participant ID": ""}
dlg = gui.DlgFromDict(info, title="Participant Info")
if not dlg.OK:
    core.quit()
participant_id = info["Participant ID"].strip() or "test"

# ----------------- Track Loading -----------------
base_dir = os.path.dirname(__file__)
track_path = os.path.join(base_dir, "audigen.wav")
track_exists = ensure_sound_file(track_path)

# handle different possible return values from ensure_sound_file
if not track_exists:
    raise FileNotFoundError(f"Track not found or ensure_sound_file failed: {track_path}")
# if ensure_sound_file returns a path string, accept that
if isinstance(track_exists, str):
    track_path = track_exists

data, sr = sf.read(track_path, dtype="float32")
if data.ndim == 1:
    data = data[:, np.newaxis]  # make 2D (samples, channels)
n_samples = data.shape[0]
n_channels = data.shape[1]

# ----------------- NTP Sync -----------------
def sync_clock_offset():
    try:
        import ntplib
        c = ntplib.NTPClient()
        r = c.request("pool.ntp.org", version=3, timeout=3)
        offset = r.tx_time - time.time()
        print(f"NTP clock offset: {offset:.6f} s")
        return offset
    except Exception as e:
        print("NTP sync failed (ntplib missing or network issue). Using local system time.")
        return 0.0

clock_offset = sync_clock_offset()
PERF0 = time.perf_counter()
NTP0 = time.time() + clock_offset

def perf_to_ntp(perf_ts):
    """Convert a perf_counter timestamp to unix UTC seconds (float)."""
    return NTP0 + (perf_ts - PERF0)

def ntp_str_from_perf(perf_ts):
    unix_ts = perf_to_ntp(perf_ts)
    return datetime.datetime.utcfromtimestamp(unix_ts).strftime("%Y-%m-%d %H:%M:%S.%f")

# ----------------- Experiment Parameters -----------------
total_noise_duration = 600.0  # seconds
tone_duration = 0.2
max_rt = 1.5
fb_dur = 0.2
volume_levels = [0.05, 0.10, 0.15]
interval_options = [2.0, 4.0, 5.0, 7.0]
sides_list = ["left", "right"]
RANDOM_SEED = 12345

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------- Trial Schedule -----------------
combos = [(v, s) for v in volume_levels for s in sides_list]

def compute_trial_schedule():
    # Try increasing the number of repeats until schedule doesn't fit
    max_repeat = 0
    final_intervals = []

    for N in range(1, 1000):
        n_trials = len(combos) * N
        sampled_intervals = [random.choice(interval_options) for _ in range(n_trials)]
        # account for tone_duration for each trial
        total_time = sum(sampled_intervals) + (n_trials * tone_duration)
        if total_time <= total_noise_duration:
            max_repeat = N
            final_intervals = sampled_intervals.copy()
        else:
            break

    if max_repeat <= 0:
        max_repeat = 1
        final_intervals = [random.choice(interval_options) for _ in range(len(combos) * max_repeat)]

    combo_pool = combos * max_repeat
    random.shuffle(combo_pool)

    if len(final_intervals) != len(combo_pool):
        final_intervals = [random.choice(interval_options) for _ in combo_pool]

    cumulative = 0.0
    scheduled_times = []
    for inter in final_intervals:
        cumulative += inter
        scheduled_times.append(cumulative)

    # make sure scheduled times fit within total_noise_duration (consider tone halfway)
    n_fit = sum(1 for t in scheduled_times if t + (tone_duration / 2.0) <= total_noise_duration)
    if n_fit < len(combo_pool):
        combo_pool = combo_pool[:n_fit]
        scheduled_times = scheduled_times[:n_fit]

    return combo_pool, scheduled_times

combo_pool, scheduled_times = compute_trial_schedule()
n_trials = len(combo_pool)
print(f"Experiment will have {n_trials} trials.")

# ----------------- Option: overlay generated tones into the audio buffer -----------------
# If audigen.wav already contains the target tones at correct moments, set OVERLAY_TONES=False.
OVERLAY_TONES = False

if OVERLAY_TONES:
    # Create a copy of the audio buffer to mix tones into (do not modify original file)
    mix = data.copy()
    def generate_tone(freq=1000.0, dur=0.2, sr=sr, amplitude=0.1, side="left"):
        t = np.arange(int(np.round(dur * sr))) / sr
        tone = np.sin(2 * np.pi * freq * t) * amplitude
        # build stereo panned version matching number of channels
        if n_channels == 1:
            return tone[:, np.newaxis]
        else:
            # panning: left-> (1,0), right->(0,1)
            if side == "left":
                stereo = np.stack([tone, np.zeros_like(tone)], axis=1)
            elif side == "right":
                stereo = np.stack([np.zeros_like(tone), tone], axis=1)
            else:  # center
                stereo = np.stack([tone, tone], axis=1)[:,:n_channels]
            # if file has >2 channels, tile or truncate
            if n_channels > 2:
                # duplicate first two channels across remaining channels
                extra = np.tile(stereo[:, :2], (1, n_channels//2))[:, :n_channels]
                return extra
            return stereo

    # Choose frequency for the tone (set as needed)
    tone_freq = 1000.0
    for idx, scheduled_offset in enumerate(scheduled_times):
        start_sample = int(round(scheduled_offset * sr))
        tone_samples = int(round(tone_duration * sr))
        side = combo_pool[idx][1]
        volume = combo_pool[idx][0]
        tone_sig = generate_tone(freq=tone_freq, dur=tone_duration, sr=sr, amplitude=volume, side=side)
        end_sample = start_sample + tone_samples
        if start_sample >= n_samples:
            continue
        if end_sample > n_samples:
            # truncate tone if it would overrun file buffer
            tone_sig = tone_sig[: n_samples - start_sample]
            end_sample = n_samples
        # mix: add tone into mix buffer (beware clipping)
        if mix.shape[1] == tone_sig.shape[1]:
            mix[start_sample:end_sample] += tone_sig
        else:
            # handle channel mismatch by repeating tone across channels
            tone_rep = np.tile(tone_sig, (1, mix.shape[1] // tone_sig.shape[1] + 1))[:, :mix.shape[1]]
            mix[start_sample:end_sample] += tone_rep

    # guard against clipping: normalize if needed
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        print(f"Warning: audio peak {peak:.3f} > 1.0, normalizing to prevent clipping.")
        mix = mix / peak

    # replace data with mix for playback
    data = mix.copy()
    n_samples = data.shape[0]

# ----------------- PsychoPy Setup -----------------
win = visual.Window(fullscr=False, color="black", units="pix", allowGUI=False)
fixation = visual.TextStim(win, text="+", color="white", height=50)
instr = visual.TextStim(
    win,
    text="Press SPACE as quickly as possible when you hear a tone.\nPress ENTER to start.",
    color="white",
    height=30,
    wrapWidth=600,
    pos=(0, -60),
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

# ----------------- SoundDevice callback/state -----------------
start_device_ts = None     # device clock time returned by callback (outputBufferDacTime)
start_perf = None          # perf_counter time corresponding to the first callback
audio_started_flag = False
play_cursor = 0

def audio_callback(outdata, frames, time_info, status):
    """
    Write audio frames and capture the first-sample device timestamp.
    time_info may contain 'outputBufferDacTime' or 'output_buffer_dac_time'
    depending on platform/version; we attempt both.
    """
    global start_device_ts, start_perf, audio_started_flag, play_cursor

    # capture once
    if not audio_started_flag:
        ts = None
        if isinstance(time_info, dict):
            ts = time_info.get("outputBufferDacTime") or time_info.get("output_buffer_dac_time")
        else:
            try:
                ts = getattr(time_info, "outputBufferDacTime", None)
            except Exception:
                ts = None

        if isinstance(ts, (float, int)):
            start_device_ts = float(ts)
            start_perf = time.perf_counter()
            audio_started_flag = True

    # supply audio chunk
    start = int(play_cursor)
    end = start + frames
    chunk = data[start:end]

    if chunk.shape[0] < frames:
        # final partial block: fill available frames then pad with zeros
        outdata[: chunk.shape[0]] = chunk
        outdata[chunk.shape[0] :] = 0.0
        play_cursor += chunk.shape[0]
        raise sd.CallbackStop()
    else:
        outdata[:] = chunk
        play_cursor += frames

# ----------------- Results and persistence -----------------
results = []
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, f"audireact_{participant_id}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv")

# ----------------- Playback stream -----------------
stream = sd.OutputStream(samplerate=sr, channels=data.shape[1], callback=audio_callback, blocksize=256)

try:
    stream.start()
    # wait briefly for callback to capture device timestamp
    t_start_wait = time.perf_counter()
    timeout = 2.0  # seconds
    while not audio_started_flag and (time.perf_counter() - t_start_wait) < timeout:
        time.sleep(0.001)

    if audio_started_flag:
        print(f"Audio callback captured device timestamp: {start_device_ts}")
        print(f"Corresponding perf timestamp: {start_perf:.6f}")
        print("NTP start:", ntp_str_from_perf(start_perf))
    else:
        # fallback: use current perf time as start anchor
        start_perf = time.perf_counter()
        print("WARNING: audio callback did not provide device timestamp within timeout.")
        print("Falling back to perf_counter anchor:", start_perf)
        print("NTP start (fallback):", ntp_str_from_perf(start_perf))

    print("Playback started — trials running")

    # main trial loop (use perf times anchored to start_perf)
    for i, scheduled_offset in enumerate(scheduled_times):
        trial_number = i + 1
        volume, side = combo_pool[i]  # note: combo_pool items are (volume, side)

        target_perf = start_perf + scheduled_offset
        # wait until target_perf, but allow ESC early termination and false-positive presses
        while True:
            now = time.perf_counter()
            if now >= target_perf:
                break
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt

            if "space" in keys:
                # false positive (pressed before scheduled onset)
                ts = time.perf_counter()
                ts_ntp = ntp_str_from_perf(ts)
                # keep columns consistent with final CSV format:
                # trial, side, volume, scheduled_timestamp_utc, response_timestamp_utc, RT_seconds, resp_status
                results.append([trial_number, "none", "", "", ts_ntp, "", "false_positive"])

                fixation.height = 70
                fixation.draw()
                win.flip()
                core.wait(fb_dur)
                fixation.height = 50
                fixation.draw()
                win.flip()

                event.clearEvents()  # clear the SPACE press completely

        # At scheduled onset (device-anchored), compute NTP timestamp from perf
        scheduled_perf = target_perf
        scheduled_ts_ntp = ntp_str_from_perf(scheduled_perf)
        print(f"Trial {trial_number} scheduled at NTP {scheduled_ts_ntp}")

        # Collect RT: record perf timestamp when space pressed
        rt = None
        response_perf = None
        rt_deadline = scheduled_perf + max_rt
        while time.perf_counter() < rt_deadline:
            keys = event.getKeys()
            if "escape" in keys:
                raise KeyboardInterrupt
            if "space" in keys:
                response_perf = time.perf_counter()
                rt = response_perf - scheduled_perf
                # visual feedback
                fixation.height = 70
                fixation.draw()
                win.flip()
                core.wait(fb_dur)
                fixation.height = 50
                fixation.draw()
                win.flip()
                break
            time.sleep(0.001)

        if rt is None:
            resp_status = "miss"
            response_ts_ntp = ""
        else:
            resp_status = "hit"
            response_ts_ntp = ntp_str_from_perf(response_perf)

        results.append([
            trial_number,
            side,
            f"{volume:.3f}",
            scheduled_ts_ntp,
            response_ts_ntp,
            "" if rt is None else f"{rt:.6f}",
            resp_status,
        ])
        print(f"Trial {trial_number}/{n_trials} | Side:{side} Vol:{volume:.3f} | {resp_status} | RT={'' if rt is None else f'{rt:.4f}s'}")

    print("All trials complete.")
    # allow stream to finish naturally
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

    # Save results (always)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "side",
            "volume",
            "scheduled_timestamp_utc",
            "response_timestamp_utc",
            "RT_seconds",
            "resp_status",
        ])
        writer.writerows(results)

    print(f"Data saved to: {csv_path}")

    win.close()
    core.quit()
