# ============================================
# Auditory reaction-time experiment
# ============================================

if __name__ == "__main__":
    from psychopy import visual, core, event, sound, gui, prefs, monitors
    import random, csv, time, os, datetime
    import numpy as np
    import sounddevice as sd
    
    device_list = sd.query_devices()

    # Extract output devices
    output_devices = [
        d['name'] for d in device_list
        if d.get('max_output_channels') > 0
    ]
    # Safety fallback
    if not output_devices:
        output_devices = ["Default device only"]
  
    # ---------- Participant ID setup ----------
    # Create GUI dialog for participant info
    info = {
        "Participant ID": "",
        "Audio Device": ["Select Audio Device"] + output_devices
    }
    
    dlg = gui.DlgFromDict(
        info,
        title="Participant & Device Setup"
    )

    # If user cancels, quit cleanly
    if not dlg.OK:
        print("Experiment cancelled at participant ID entry.")
        core.quit()

    participant_id = info["Participant ID"].strip()
    if not participant_id:
        participant_id = "test"
        print("No ID entered — using 'test'")
        
    selected_device = info["Audio Device"]
    
    # If they left the placeholder, just warn and use default
    if selected_device == "Select headphones":
        print("⚠ No audio device selected — using system default.")
    else:
        prefs.hardware['audioDevice'] = selected_device
        print(f"Using audio device: {selected_device}")
        
        prefs.hardware['audioDevice'] = selected_device
        prefs.hardware['audiolib'] = ['PTB']  # force PTB backend

    # ---------- File setup ----------
    # Define main experiment folder
    base_dir = os.path.dirname(__file__)

    # Create data directory and participant subfolder
    data_dir = os.path.join(base_dir, "data", participant_id)
    os.makedirs(data_dir, exist_ok=True)

    # Build filename with timestamp
    filename = f"audireact_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    csv_path = os.path.join(data_dir, filename)

    print(f"Data will be saved to: {csv_path}")

    # ---------- Experiment parameters ----------
    n_trials        = 6             # total number of tones
    amplitude       = 0.1           # background noise loudness
    sr              = 44100         # sample rate
    noise_duration  = 60.0          # seconds of noise
    max_rt          = 1.5           # seconds; after this, count as miss
    fb_dur          = 0.2           # seconds; feedback flash duration
    iti_range       = (1.0, 1.5)    # seconds; random ITI
    freq            = 440.0         # Hz (A4)
    tone_duration   = 0.2           # seconds of tone
    
    # ---------- Visual setup ----------
    win             =   visual.Window(fullscr=False, color="black", units="pix", allowGUI=False)
    fixation        =   visual.TextStim(win, text="+", color="white", height=50)
    header          =   visual.TextStim(win, text="INSTRUCTIONS", color="white", height=50, pos=(0, 200))
    body_text       =   (
        """Researcher:
    Press ENTER to deliver the next tone.
    
    Participant:
    Press SPACE as quickly as possible
    when you hear the tone.
                                 
    Press ENTER to start.
    Press ESC to exit."""
    )
    intro_body      =   visual.TextStim(win, text=body_text, color="white", height=30, wrapWidth=800, pos=(0, -80))
    end_text        =   visual.TextStim(win, text="Experiment complete.\nThank you!", color="white", height=40)
    
    header.draw()
    intro_body.draw()
    win.flip()

    # Allow ESC before starting
    while True:
        keys = event.getKeys()
        if 'escape' in keys:
            print("Experiment aborted at intro screen.")
            win.close()
            core.quit()
        if 'return' in keys:
            break
        core.wait(0.01)

    # ---------- Sound setup ----------
    # Generate white noise (random samples between -1 and 1)
    noise_samples = np.random.uniform(-1, 1, int(sr * noise_duration)) * amplitude
    
    # Add smooth fade in/out to prevent clicks
    fade_len = int(sr * 0.05)  # 50 ms fade
    fade = np.linspace(0, 1, fade_len)
    noise_samples[:fade_len] *= fade
    noise_samples[-fade_len:] *= fade[::-1]
    
    # Convert to stereo (identical in both ears)
    stereo_noise = np.column_stack((noise_samples, noise_samples))

    # Create Sound object and play continuously (for very long)
    noise = sound.Sound(stereo_noise, stereo=True, sampleRate=sr)
    noise.play(loops=9999)

    # Generate waveform for 0.2s sine wave
    t = np.linspace(0, tone_duration, int(sr * tone_duration), endpoint=False)
    tone_wave = np.sin(2 * np.pi * freq * t) * amplitude
    
    # Create stereo arrays: left-only and right-only
    left_channel = np.column_stack((tone_wave, np.zeros_like(tone_wave)))   # sound in left only
    right_channel = np.column_stack((np.zeros_like(tone_wave), tone_wave))  # sound in right only

    # Convert to PsychoPy Sound objects
    tone_left = sound.Sound(left_channel, stereo=True, sampleRate=sr)
    tone_right = sound.Sound(right_channel, stereo=True, sampleRate=sr)

    # ---------- Trial setup ----------
    sides       = ['left', 'right'] * (n_trials // 2)
    random.shuffle(sides)
    results     = []  # trial, side, RT, resp_status, timestamp
    clock       = core.Clock()

    # ---------- Neutral feedback ----------
    def flash_neutral(dur=fb_dur):
        fixation.height = 70
        fixation.draw()
        win.flip()
        core.wait(dur)
        fixation.height = 50
        fixation.draw()
        win.flip()

    # ---------- Main loop ----------
    try:
        for trial_number, side in enumerate(sides, start=1):

            fixation.draw()
            win.flip()

            # Wait for researcher to trigger the next tone
            while True:
                keys = event.getKeys()
                if 'space' in keys:  # false positive before tone
                    results.append([trial_number, "none", "", "false_positive", time.strftime("%H:%M:%S")])
                    print(f"False positive detected before trial {trial_number}")
                    flash_neutral()
                if 'escape' in keys:
                    raise KeyboardInterrupt
                if 'return' in keys:
                    break
                core.wait(0.01)

            # Play tone and start RT clock
            clock.reset()
            (tone_left if side == 'left' else tone_right).play()

            # Wait for participant response or until max_rt
            rt = None
            resp_status = "hit"
            while clock.getTime() < max_rt and rt is None:
                resp = event.getKeys(keyList=['space', 'escape'], timeStamped=clock)
                if resp:
                    key, rt = resp[0]
                    if key == 'escape':
                        raise KeyboardInterrupt

            if rt is None:
                resp_status = "miss"
            else:
                flash_neutral()

            rt_for_csv = "" if rt is None else f"{rt:.6f}"
            results.append([trial_number, side, rt_for_csv, resp_status, time.strftime("%H:%M:%S")])
            print(f"Trial {trial_number}/{n_trials} | Side: {side} | {resp_status} | {rt_for_csv}")

            core.wait(random.uniform(*iti_range))

        print("All trials complete!")
        noise.stop()
        end_text.draw()
        win.flip()
        core.wait(4)

    except KeyboardInterrupt:
        print("Experiment aborted early by ESC key.")
        core.quit()

    finally:
        # ---------- Save results ----------
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial', 'side', 'RT_seconds', 'resp_status', 'timestamp'])
            writer.writerows(results)
        print(f"Data saved to: {csv_path}")

        noise.stop()
        win.close()
        core.quit()
    

