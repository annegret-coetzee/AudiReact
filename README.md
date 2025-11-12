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
- **Trial configuration:** Currently fixed at **6 trials** for development and debugging purposes; this value can easily be adjusted in the code (`n_trials` variable).  
- **Tone distribution:** The left/right tones are **randomized but evenly balanced** across trials to avoid side bias.    
- **Stereo tone generation:** Separate left/right sine waves ensure precise lateralization  
- **Continuous white noise:** Maintains engagement and masks ambient sounds  
- **Visual feedback:** Neutral “+” flash confirms button press without introducing affective bias  
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

When prompted, either enter a **Participant ID** (e.g., `P01`).  
If left blank, data are saved under `data/test/`.  

> Note: Running the script outside of PsychoPy (e.g., in a standalone virtual environment) was tested but led to dependency conflicts between PsychoPy, NumPy, and many others. The PsychoPy application remains the most stable and portable option.
> Note: The number of trials are set to 6 for testing purposes
---

## Future Development

- Integration with breathing-belt input for **real-time phase-locked tone delivery**  
- Parameterization of tone frequency, intensity, and inter-trial interval  
- Optional GUI for experiment setup and live monitoring  


*Tested with PsychoPy 2024.2 on Windows 10.*