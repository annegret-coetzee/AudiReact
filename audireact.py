# ==============================================================================
# Auditory reaction-time experiment (pre-generated track)
# with NTP timestamps
# ==============================================================================

from psychopy import prefs, visual, core, event, sound, gui
import os, csv, datetime, time, random, numpy as np


# ---------------------- NTP sync for grounded timestamps ----------------------
def sync_clock_offset():
    """
    Try to obtain an offset between system time and NTP time.
    Returns offset in seconds: (NTP_UTC - local_system_time).
    If anything fails (no internet, ntplib missing, etc.), returns 0.0.
    """
    try:
        import ntplib
        c = ntplib.NTPClient()
        r = c.request("pool.ntp.org", version=3, timeout=3)
        offset = r.tx_time - time.time()
        print(f"NTP clock offset: {offset:.6f} seconds (UTC - local)")
        return offset
    except Exception as e:
        print(f"Warning: NTP sync failed ({e}); using local system time only.")
        return 0.0

# Compute once at experiment start
clock_offset = sync_clock_offset()

def get_ntp_timestamp():
    """
    Return a NTP-grounded UTC timestamp string with milliseconds.
    If NTP failed, this is just system time in UTC.
    """
    true_unix = time.time() + clock_offset
    return datetime.datetime.utcfromtimestamp(true_unix).strftime("%Y-%m-%d %H:%M:%S.%f")


#----------------------------- Audio Device Setup ------------------------------
import sounddevice as sd

# Use system default output device
default_output_index = sd.default.device[1]  # [input, output], output is index 1
device_info = sd.query_devices(default_output_index)
device_name = device_info['name']

# Feed PsychoPy PTB backend
prefs.hardware['audioDevice'] = device_name
prefs.hardware['audiolib'] = ['PTB']
print(f"Using system default audio device: {device_name}")


# ------------------------- Experiment parameters ------------------------------
total_noise_duration = 600.0   # 10 minutes of white noise (seconds)
freq = 440.0                   # tone freq (Hz)
tone_duration = 0.2            # tone length (s)
max_rt = 1.5                   # response window (s)
fb_dur = 0.2                   # neutral feedback flash (s)

# Intensity and interval choices
volume_levels = [0.05, 0.10, 0.15]    
interval_options = [2.0, 3.0, 4.0, 5.0] 

# Combinations (volume x side) repeated equally
sides_list = ["left", "right"]
combos = [(v, s) for v in volume_levels for s in sides_list]  # 6 combos

#Experiment-wide random seed (reproducible soundtrack)
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------------Participant ID & File Setup---------------------------
info = {"Participant ID": ""}
dlg = gui.DlgFromDict(info, title="Participant Info")
if not dlg.OK:
    print("Experiment cancelled at participant ID entry.")
    core.quit()

participant_id = info["Participant ID"].strip()
if not participant_id:
    participant_id = "test"
    print("No ID entered — using 'test'")

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir,"data",participant_id)
os.makedirs(data_dir, exist_ok=True)

track_path = os.path.join(base_dir, "audireact_track.wav")

# Auto-generate if missing
if not os.path.exists(track_path):
    from audireact_track import generate_track
    generate_track(track_path)

csv_path = os.path.join(data_dir,f"audireact_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
print(f"Data will be saved to: {csv_path}")



# ---------------- Compute trial repeats and schedule intervals ----------------
def compute_trial_schedule():
    max_repeat = 0
    final_intervals = []
    
    for N in range(1, 1000):  # safety upper bound
        n_trials = len(combos) * N
        # sample intervals reproducibly for this N
        sampled_intervals = [random.choice(interval_options) for _ in range(n_trials)]
        total_time = sum(sampled_intervals) + tone_duration
        if total_time <= total_noise_duration:
            max_repeat = N
            final_intervals = sampled_intervals.copy()
        else:
            break

    if max_repeat <= 0:
        max_repeat = 1
        n_trials = len(combos) * max_repeat
        final_intervals = [random.choice(interval_options) for _ in range(len(combos)*max_repeat)]
    
    # build combo pool
    combo_pool = combos * max_repeat
    # shuffle combo pool reproducibly
    random.shuffle(combo_pool)
    
    # assign intervals according to shuffled combo_pool length
    if len(final_intervals) != len(combo_pool):
        # safety check
        final_intervals = [random.choice(interval_options) for _ in combo_pool]

    # compute cumulative scheduled times
    cumulative = 0.0
    scheduled_times = []
    for inter in final_intervals:
        cumulative += inter
        scheduled_times.append(cumulative)
    
    # trim any trials that would exceed total_noise_duration
    n_fit = sum(1 for t in scheduled_times if t + (tone_duration / 2.0) <= total_noise_duration)
    if n_fit < len(combo_pool):
        print(f"Note: only {n_fit} of {len(combo_pool)} trials fit within {total_noise_duration}s. Trimming.")
        combo_pool = combo_pool[:n_fit]
        final_intervals = final_intervals[:n_fit]
        scheduled_times = scheduled_times[:n_fit]
    
    return combo_pool, final_intervals, scheduled_times, max_repeat

# Call the function once at experiment start
combo_pool, intervals_assigned, scheduled_times, repeats = compute_trial_schedule()
n_trials = len(combo_pool)
print(f"Experiment will have {n_trials} trials.")


# ----------------------------- Visual setup -----------------------------------
win = visual.Window(fullscr=False, color="black", units="pix", allowGUI=False)
fixation = visual.TextStim(win, text="+", color="white", height=50)
instr_header = visual.TextStim(win, text="INSTRUCTIONS", color="white", height=48, pos=(0, 200))
instr_body = visual.TextStim(win, text=(
    "Participant: Press SPACE as quickly as possible when you hear a tone.\n\n"
    "The track is fixed and will run for 10 minutes. You can stop anytime with ESC. \n\n"
    "Press ENTER and the experiment will start shortly."
), color="white", height=30, wrapWidth=600, pos=(0, -60), alignText='center')
end_text = visual.TextStim(win, text="Experiment complete.\nThank you!", color="white", height=40)

instr_header.draw()
instr_body.draw()
win.flip()

# Allow ESC to quit at instruction page; ENTER to begin
while True:
    keys = event.getKeys()
    if 'escape' in keys:
        print("Experiment aborted at intro screen.")
        win.close()
        core.quit()
    if 'return' in keys:
        break
    core.wait(0.01)

fixation.draw()
win.flip()


# ---------- Load pre-generated track safely ----------
track_path = os.path.join(base_dir, "audireact_track.wav")
if not os.path.exists(track_path):
    try:
        from audireact_track import generate_track
        generate_track(track_path)
    except Exception as e:
        print(f"Error generating track: {e}")
        win.close()
        core.quit()

# Verify the WAV file exists and is readable
if not os.path.exists(track_path):
    print(f"Track file still missing: {track_path}")
    win.close()
    core.quit()

# Attempt to load the track
try:
    track = sound.Sound(track_path)
except Exception as e:
    print(f"Failed to load track: {e}")
    win.close()
    core.quit()

# Attempt to play track
try:
    track.play()
except Exception as e:
    print(f"Failed to play track: {e}")
    win.close()
    core.quit()

print("Track started successfully. Press ESC to quit at any time.")


# ---------- Main trial loop ----------
results = []
main_clock = core.Clock()
start_time_local = time.time()
start_ntp = get_ntp_timestamp()
print(f"Track started at local time {datetime.datetime.fromtimestamp(start_time_local)} (NTP-UTC: {start_ntp})")

try:
    for i in range(n_trials):
        trial_number = i+1
        side = combo_pool[i][1]
        volume = combo_pool[i][0]
        interval = intervals_assigned[i]
        scheduled_time = scheduled_times[i]

        # Wait until scheduled time
        while main_clock.getTime() < scheduled_time:
            keys = event.getKeys()
            if 'escape' in keys:
                raise KeyboardInterrupt
            if 'space' in keys:
                ts_fp = get_ntp_timestamp()
                results.append([trial_number,"none","","",ts_fp,"","", "false_positive"])
                flash = fixation
                flash.height = 70
                flash.draw()
                win.flip()
                core.wait(fb_dur)
                flash.height = 50
                flash.draw()
                win.flip()
            core.wait(0.01)

        # Record NTP timestamp exactly at tone playback
        scheduled_ts_ntp = get_ntp_timestamp()
        # In this case, the tone is already part of the pre-generated WAV, so we only log the timestamp

        rt_clock = core.Clock()
        rt = None
        resp_status = "hit"
        response_ts_ntp = ""

        while rt_clock.getTime() < max_rt and rt is None:
            resp = event.getKeys(keyList=['space','escape'],timeStamped=rt_clock)
            if resp:
                key, rt = resp[0]
                if key=='escape':
                    raise KeyboardInterrupt
                response_ts_ntp = get_ntp_timestamp()

        if rt is None:
            resp_status = "miss"
        else:
            flash = fixation
            flash.height = 70
            flash.draw()
            win.flip()
            core.wait(fb_dur)
            flash.height = 50
            flash.draw()
            win.flip()

        rt_for_csv = "" if rt is None else f"{rt:.6f}"
        results.append([
            trial_number,
            side,
            f"{volume:.3f}",
            f"{interval:.3f}",
            scheduled_ts_ntp,
            response_ts_ntp,
            rt_for_csv,
            resp_status
        ])
        print(f"Trial {trial_number}/{n_trials} | Side:{side} Vol:{volume:.3f} Int:{interval:.1f}s | {resp_status} | RT={rt_for_csv}")

    print("All trials complete!")
    track.stop()
    end_text.draw()
    win.flip()
    core.wait(4)

except KeyboardInterrupt:
    print("Experiment aborted early by ESC key.")
    track.stop()
    win.close()
    core.quit()

finally:
    with open(csv_path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial','side','volume','interval_s',
            'scheduled_timestamp_utc','response_timestamp_utc',
            'RT_seconds','resp_status'
        ])
        writer.writerows(results)
    print(f"Data saved to: {csv_path}")

    win.close()
    core.quit()
