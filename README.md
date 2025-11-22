# Auditory Reaction-Time Experiment

This program implements the auditory component of our **Cognitive Science study on respiration and attention**.  
It measures **reaction times (RTs)** to lateralized auditory tones (left vs. right ear) presented during continuous white noise, while participants’ breathing is recorded externally via a breathing belt, to be incorporated later.

---

## Purpose and Design

The experiment investigates how respiratory phase modulates auditory perception and motor response timing.
Participants hear a 10-minute track of continuous white noise punctuated by short tones of different volumes and lateralizations. They press the space bar as quickly as possible when they detect a tone.

	-Independent variable: lateralization and volume of auditory tone
	-Dependent variable: reaction time (RT) to the tone
	-Data quality note: Wired headphones are recommended to avoid audio latency

The program automatically records misses and false positives, and each timestamp is NTP-grounded for precise cross-device synchronization.

---

## Implementation

- **Platform:** PsychoPy (Python)
- **Trial configuration:** 10-minute pre-generated auditory track with reproducible tone sequence
- **Tone variations:** 
	- Three volume levels (low, medium, high)
	- Two lateralizations (left/right ear)
	- All combinations repeated equally
- **Randomized but reproducible sequence:** uses fixed random seed for cross-participant consistency
- **Inter-tone intervals::** randomly drawn from four possible durations
- **Continuous white noise:** Maintains engagement and masks ambient sounds  
- **Visual feedback:** Neutral “+” flash confirms button press without introducing affective bias
- **Misses and false positives:**
	- Missed tones are recorded as “miss”
	- Button presses before tones are recorded as “false positive”
- **Audio output:** automatically uses system default; ensures reliable delivery   
- **Cross-platform:** Runs in PsychoPy on Windows, macOS, and Linux  

---

## Data output
Participant responses are logged automatically under /data/<ParticipantID>/.

Each CSV row corresponds to a tone or a false-positive response, with the following fields:

| Column                    | Description                                                                     |
| ------------------------- | ------------------------------------------------------------------------------- |
| `trial`                   | Trial number (sequence of tones in the 10-min track)                            |
| `side`                    | Ear in which the tone was played (`left`/`right`) or `none` for false positives |
| `volume`                  | Tone volume (0.05, 0.10, 0.15)                                                  |
| `interval_s`              | Inter-tone interval preceding this tone (seconds)                               |
| `scheduled_timestamp_utc` | NTP-grounded timestamp when tone was scheduled/played                           |
| `response_timestamp_utc`  | NTP-grounded timestamp of participant’s space-bar press                         |
| `RT_seconds`              | Reaction time in seconds (blank if missed)                                      |
| `resp_status`             | Response status: `hit`, `miss`, or `false_positive`                             |


Examples:
Hit: participant responded to the tone within the allowed window.
Miss: participant did not respond within the response window.
False positive: participant pressed the space bar before the next tone.

Using NTP-grounded timestamps ensures synchronization with external devices such as a breathing belt, allowing analysis of RTs relative to respiratory phase.
---

## Folder Structure

AudiReact/
├── audireact.py # Main experiment script
├── audireact_track.py     # Generates pre-computed auditory track if missing
├── audireact_track.wav    # Pre-generated stimulus track (auto-created)
├── data/ # Participant data output (auto-created)
├── requirements.txt # Dependencies (for reference)
└── README.md # This file

---

## How to Run the Experiment

1. Install [PsychoPy](https://www.psychopy.org/download.html).  
2. Open **Coder view** (not Builder).  
3. Load `audireact.py`.  
4. Press **Run ▶**.  

At startup, the program will:
- Check if audireact_track.wav exists; if missing, it generates the 10-minute track automatically.
- Use the system default audio output.
- Prompt for Participant ID (blank defaults to test).

**Note:** Wired headphones are recommended to minimize audio latency.
**Note:** Timestamps are NTP-grounded for synchronization with external devices (e.g., breathing belt).
**Note:** The tone sequence is reproducible, allowing cross-participant comparison.
---

## Future Development

- Integration with real-time breathing-belt input for phase-locked tone delivery
- Adjustable parameters for tone frequency, volume levels, and inter-tone intervals
- Optional GUI for live monitoring and experiment setup

Tested with PsychoPy 2024.2 on Windows 11.