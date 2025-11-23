# ============================================
# Generate full Auditory Reaction-Time track
# ============================================

def generate_track(track_path="audireact_track.wav"):
    import numpy as np
    import random
    import soundfile as sf

    # ---------- Parameters ----------
    RANDOM_SEED = 12345
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    sr = 44100  # sampling rate
    total_noise_duration = 600.0  # 10 minutes
    freq = 440.0
    tone_duration = 0.2
    volume_levels = [0.05, 0.10, 0.15]
    interval_options = [2.0, 3.0, 4.0, 5.0]
    sides_list = ["left", "right"]

    # ---------- Generate combos ----------
    combos = [(v, s) for v in volume_levels for s in sides_list]

    # ---------- Compute max repeats and scheduled intervals ----------
    def compute_trial_schedule():
        max_repeat = 0
        final_intervals = []

        for N in range(1, 1000):
            n_trials = len(combos) * N
            sampled_intervals = [random.choice(interval_options) for _ in range(n_trials)]
            total_time = sum(sampled_intervals) + tone_duration
            if total_time <= total_noise_duration:
                max_repeat = N
                final_intervals = sampled_intervals.copy()
            else:
                break

        if max_repeat <= 0:
            max_repeat = 1
            final_intervals = [random.choice(interval_options) for _ in range(len(combos) * max_repeat)]
            print("Warning: could not fit combos into noise duration; using repeats=1")

        combo_pool = combos * max_repeat
        random.shuffle(combo_pool)

        if len(final_intervals) != len(combo_pool):
            final_intervals = [random.choice(interval_options) for _ in combo_pool]

        cumulative = 0.0
        scheduled_times = []
        for inter in final_intervals:
            cumulative += inter
            scheduled_times.append(cumulative)

        n_fit = sum(1 for t in scheduled_times if t + (tone_duration/2) <= total_noise_duration)
        if n_fit < len(combo_pool):
            print(f"Note: only {n_fit} of {len(combo_pool)} trials fit in {total_noise_duration}s. Trimming.")
            combo_pool = combo_pool[:n_fit]
            final_intervals = final_intervals[:n_fit]
            scheduled_times = scheduled_times[:n_fit]

        return combo_pool, final_intervals, scheduled_times

    combo_pool, intervals_assigned, scheduled_times = compute_trial_schedule()
    n_trials = len(combo_pool)
    print(f"Generating track with {n_trials} tones.")

    # ---------- Generate base tone ----------
    t_wave = np.linspace(0, tone_duration, int(sr * tone_duration), endpoint=False)
    base_tone_wave = np.sin(2 * np.pi * freq * t_wave)

    # ---------- Generate white noise ----------
    amplitude_noise = 0.02
    noise_samples = np.random.uniform(-1,1,int(sr*total_noise_duration)) * amplitude_noise
    fade_len = int(sr*0.05)
    fade = np.linspace(0,1,fade_len)
    noise_samples[:fade_len] *= fade
    noise_samples[-fade_len:] *= fade[::-1]
    stereo_noise = np.column_stack((noise_samples, noise_samples))  # stereo

    # ---------- Overlay tones ----------
    track_samples = stereo_noise.copy()

    for i, (vol, side) in enumerate(combo_pool):
        start_sample = int(scheduled_times[i] * sr)
        tone_wave = base_tone_wave * vol
        left_ch = np.column_stack((tone_wave, np.zeros_like(tone_wave)))
        right_ch = np.column_stack((np.zeros_like(tone_wave), tone_wave))
        tone_array = left_ch if side=="left" else right_ch
        end_sample = start_sample + tone_array.shape[0]
        if end_sample > track_samples.shape[0]:
            end_sample = track_samples.shape[0]
            tone_array = tone_array[:end_sample-start_sample]
        track_samples[start_sample:end_sample] += tone_array

    # Clip to [-1,1] to avoid distortion
    track_samples = np.clip(track_samples, -1.0, 1.0)

    # ---------- Save as WAV ----------
    sf.write(track_path, track_samples, sr)
    print(f"Track saved as {track_path}")
