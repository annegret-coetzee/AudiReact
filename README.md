# Auditory Reaction-Time Experiment

This program implements the auditory component of our **Cognitive Science study on respiration and attention**.  
It measures **reaction times (RTs)** to lateralized auditory tones (left vs. right ear) presented during continuous white noise, while participants’ breathing is either recorded externally or will later be integrated via a breathing belt.

---

## Purpose and Design

The experiment examines how **respiratory phase** modulates **auditory perception and motor response timing**.

Participants hear continuous white noise through headphones and press the **space bar** as quickly as possible when a short tone appears in either ear.

- **Independent variable:** breathing phase (to be implemented / externally recorded)  
- **Dependent variable:** reaction time (RT) to the auditory tone  

The current version allows the researcher to **manually trigger stimuli** (ENTER key), ensuring reliable data collection even before the real-time breathing-belt integration is completed.

---

## Implementation

- **Platform:** PsychoPy (Python)
- **Stimulus control:** Manual tone triggering by the researcher
- **Trial configuration:** Currently **6 trials** for debugging; adjustable via n_trials
- **Tone distribution:** The left/right tones are are evenly balanced and then randomized
- **Stereo tone generation:** True left-only and right-only sine waves
- **Continuous white noise:** Maintains engagement and masks ambient sounds  
- **Visual feedback:** Neutral “+” flash confirms button press without introducing affective bias 
- **Audio-device selection:** At startup, the GUI includes a dropdown menu listing all available output devices, allowing the user to explicitly select their headphones. This prevents cases where PsychoPy defaults to the wrong output device, which can suppress or misroute the tones.
- **Data logging:** Each participant’s data (trial, side, RT, response status, timestamp) are stored automatically under `/data/<ParticipantID>/`  
  - If the **Participant ID** field is left empty, the program defaults to `data/test/`  
- **Cross-platform:** Runs in PsychoPy on Windows, macOS, and Linux  

---

## Folder Structure

AudiReact/
├── audireact.py # Main experiment script
├── data/ # Participant data output (auto-created)
├── requirements.txt # Dependencies (for reference)
└── README.md # This file

---

## How to Run the Experiment

1. Install [PsychoPy](https://www.psychopy.org/download.html).  
2. Open **Coder view** (not Builder).  
3. Load `audireact.py`.  
4. Press **Run ▶**.  

A dialog will appear asking for:
- Audio Device (dropdown list of available outputs)
- Participant ID
	-Leaving the ID blank results in files being saved under data/test/.

> Note: Running the script outside of PsychoPy (e.g., in a standalone virtual environment) was tested but led to dependency conflicts between PsychoPy, NumPy, and many others. The PsychoPy application remains the most stable and portable option.
> Note: The number of trials are set to 6 for testing purposes
---

## Future Development

- Integration with breathing-belt input for **real-time phase-locked tone delivery**  
- Parameterization of tone frequency, intensity, and inter-trial interval  
- Optional GUI for experiment setup and live monitoring  


*Tested with PsychoPy 2024.2 on Windows 11.*