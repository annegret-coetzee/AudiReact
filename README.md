# Auditory Reaction-Time Experiment

This program implements the auditory component of our **Cognitive Science study on respiration and attention**.  
It measures **reaction times (RTs)** to lateralized auditory tones (left vs. right ear) presented during continuous white noise, while participants’ breathing is recorded externally via a breathing belt, to be incorporated later.

---

## Purpose and Design

The experiment investigates how respiratory phase modulates auditory perception and motor response timing.
Participants hear continuous white noise punctuated by short 0.10-volume tones presented to either the left or right ear. They press the space bar as quickly as possible when they detect a tone.

	-**Independent variable:** lateralization (left vs. right)
	-**Control:** all tones use identical frequency and volume
	-**Good practice:** wired headphones and belt recommended to minimize system audio latency

The program:
	- Schedules each tone using high-resolution wall-clock time (perf_counter)
	- Logs reactions with UTC-synchronized timestamps for cross-device alignment
	- Records hits, misses, and false positives

---

## Implementation
### Core technical design
	- The experiment now uses a single audio output stream with real-time mixing:
	- Continuous white noise is generated in the audio callback.
	- Tone waveforms are injected at precise sample positions when scheduled.
	- A fixed random seed ensures reproducible trial order.
	
### Stimulus configuration
Tones:
	- 440 Hz
	- 0.2 s duration
	- Base volume fixed at 0.10
	- Presented either to left or right channel

Intervals:
	- Four possible durations, balanced across left and right

Lateralization:
	- Equal number of left and right tones

Scheduling:
	- All intervals and sides are pre-randomized with balancing
	- Sequence is reproducible across participants

### Visual feedback
Pressing space shows a brief neutral “+” flash to confirm detection without affecting arousal.

### Error classification
	- Hit: response within the allowed window
	- Miss: no response before the next tone
	- False positive: button press when no tone is active

## Data Output
Each participant’s data is stored in: */data/<ParticipantID>/*

A CSV file logs one row per tone and one row per false-positive event.

| Column                    | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| `trial`                   | Trial index (1…N)                                        |
| `side`                    | `left` / `right` for tones, `none` for false positives   |                                                        |
| `interval_s`              | Inter-tone interval preceding the tone                   |
| `scheduled_timestamp_utc` | UTC timestamp when the tone **was scheduled and output** |
| `response_timestamp_utc`  | UTC timestamp of the participant’s keypress              |
| `RT_seconds`              | Reaction time in seconds (hit only)                      |
| `resp_status`             | `hit`, `miss`, or `false_positive`                       |

All timestamps are synchronized to system UTC, allowing future alignment with breathing-belt data.

## Folder Structure
AudiReact/
├── audireact.py          # Main experiment (live scheduling architecture)
├── data/                 # Output directory (auto-created)
└── README.md             # This file

*(Note: The old pre-generated track system has been removed.)*

## How to Run the Experiment

1. Install [PsychoPy](https://www.psychopy.org/download.html).  
2. Open **Coder view**.  
3. Load `audireact.py`.  
4. Press **Run ▶**.  

At startup, the script will:
	- Prompt for Participant ID
	- Pre-generate the balanced trial structure
	- Start the continuous noise stream
	- Schedule tones precisely based on real time
	
**Recommendation**: use wired headphones for accurate left–right localization and stable latency.
---

## Future Development

- Integration with real-time breathing-belt input for phase-locked tone delivery
- Direct coupling of tones to respiratory phase
- Configurable parameters for intervals and balancing
- Optional GUI for live monitoring and experiment setup

Tested with PsychoPy 2024.2 on Windows 11.